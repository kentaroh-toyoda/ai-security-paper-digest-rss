#!/usr/bin/env python3
"""
Attack Extractor for AI Security Papers

This module extracts attack techniques, prompts, and methodologies from academic papers
and prepares them for implementation in promptfoo red teaming configurations.

Usage:
    python attack_extractor.py --paper-id <arxiv_id>
    python attack_extractor.py --all-papers
"""

import os
import re
import json
import argparse
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv

from utils.llm import (
    create_openrouter_client,
    make_rate_limited_request,
    clean_and_extract_json,
    calculate_cost,
    format_cost,
    check_rate_limit_status,
    DEFAULT_MODEL
)
from utils.qdrant import (
    init_qdrant_client,
    get_paper_by_url,
    get_paper_by_arxiv_id,
    get_all_papers,
    update_paper_attacks,
    update_paper_attacks_by_arxiv_id
)

# Load environment variables
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

class AttackExtractor:
    """Extracts attack techniques from academic papers."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = create_openrouter_client(api_key)
        
    def extract_attacks_from_paper(self, paper_data: Dict[str, Any], model: str = DEFAULT_MODEL) -> Dict[str, Any]:
        """Extract attacks from a structured paper.
        
        Args:
            paper_data: Dictionary containing structured paper data
            model: LLM model to use for extraction
            
        Returns:
            Dictionary containing extracted attacks and metadata
        """
        print(f"\n🔍 Analyzing paper: {paper_data.get('title', 'Unknown')}")
        
        # Prepare paper content for analysis
        paper_content = self._prepare_paper_content(paper_data)
        
        # Extract attacks using LLM
        attack_analysis = self._analyze_for_attacks(paper_content, model)
        
        # Process and structure the results
        structured_attacks = self._structure_attack_results(attack_analysis, paper_data)
        
        return structured_attacks
    
    def _prepare_paper_content(self, paper_data: Dict[str, Any]) -> str:
        """Prepare paper content for attack analysis."""
        sections = paper_data.get('sections', {})
        
        content_parts = []
        
        # Add title and abstract
        if paper_data.get('title'):
            content_parts.append(f"Title: {paper_data['title']}")
        
        if sections.get('abstract'):
            content_parts.append(f"Abstract: {sections['abstract']}")
        
        # Add key sections that might contain attacks
        key_sections = ['introduction', 'methodology', 'results', 'discussion', 'conclusion']
        
        for section_name in key_sections:
            if sections.get(section_name):
                content_parts.append(f"{section_name.title()}: {sections[section_name]}")
        
        return "\n\n".join(content_parts)
    
    def _analyze_for_attacks(self, paper_content: str, model: str) -> Dict[str, Any]:
        """Analyze paper content for attack techniques using LLM."""
        
        system_prompt = """You are an AI security researcher analyzing academic papers for implementable attack techniques.

Your task is to extract concrete attack methods that can be implemented in red teaming tools like promptfoo.

Look for:
1. **Direct Attacks**: Specific prompts, jailbreaks, or attack sequences
2. **Attack Methodologies**: Step-by-step procedures for attacks
3. **Prompt Injection Techniques**: Ways to manipulate LLM behavior
4. **Jailbreaking Methods**: Techniques to bypass safety measures
5. **Adversarial Prompting**: Methods to elicit harmful responses
6. **Social Engineering Attacks**: Role-playing or persona-based attacks
7. **Multi-turn Attack Sequences**: Complex conversation-based attacks

For each attack found, extract:
- Attack name/type
- Concrete examples or prompts
- Step-by-step methodology
- Target vulnerabilities
- Success conditions
- Difficulty level (easy/medium/hard)

If no concrete attacks are found, analyze what type of paper this is and why no attacks were extractable.

Output as JSON with this structure:
{
  "attacks_found": true/false,
  "paper_classification": "direct_attacks|theoretical|defensive|survey|evaluation|other",
  "attack_count": number,
  "attacks": [
    {
      "name": "Attack name",
      "type": "jailbreak|prompt_injection|adversarial|social_engineering|multi_turn|other",
      "description": "Brief description",
      "methodology": "Step-by-step approach",
      "example_prompts": ["prompt1", "prompt2"],
      "target_vulnerabilities": ["vulnerability1", "vulnerability2"],
      "success_criteria": "How to measure success",
      "difficulty": "easy|medium|hard",
      "promptfoo_category": "harmful|intent|jailbreak|custom"
    }
  ],
  "theoretical_concepts": [
    {
      "concept": "Theoretical attack concept",
      "description": "Description of the concept",
      "implementation_potential": "low|medium|high"
    }
  ],
  "evaluation_criteria": [
    {
      "criterion": "Evaluation method",
      "description": "How it could be used for testing"
    }
  ],
  "defensive_insights": [
    {
      "defense": "Defensive measure mentioned",
      "implied_attacks": ["attacks this defense counters"]
    }
  ],
  "paper_summary": "Brief summary of paper's contribution to AI security",
  "implementation_recommendations": "Recommendations for implementing found attacks",
  "references_to_check": ["Referenced papers that might contain attacks"]
}"""

        user_prompt = f"Please analyze this academic paper for implementable attack techniques:\n\n{paper_content}"
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1
        }
        
        try:
            response = make_rate_limited_request(
                "https://openrouter.ai/api/v1/chat/completions",
                self.headers,
                payload
            )
            
            result_data = response.json()
            result = result_data["choices"][0]["message"]["content"]
            
            # Extract token usage and cost
            usage = result_data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
            cost = calculate_cost(input_tokens, output_tokens, model)
            
            print(f"💰 Attack extraction cost: {format_cost(cost)} ({total_tokens} tokens)")
            
            # Parse the JSON response
            attack_data = clean_and_extract_json(result)
            attack_data['_metadata'] = {
                'tokens_used': total_tokens,
                'cost': cost,
                'model': model,
                'extraction_date': datetime.now().isoformat()
            }
            
            return attack_data
            
        except Exception as e:
            print(f"❌ Error analyzing paper for attacks: {str(e)}")
            return {
                "attacks_found": False,
                "paper_classification": "error",
                "attack_count": 0,
                "attacks": [],
                "error": str(e)
            }
    
    def _structure_attack_results(self, attack_analysis: Dict[str, Any], paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Structure the attack analysis results with paper metadata."""
        
        structured_result = {
            "paper_info": {
                "arxiv_id": paper_data.get('arxiv_id'),
                "title": paper_data.get('title'),
                "authors": paper_data.get('authors', []),
                "date": paper_data.get('date'),
                "url": paper_data.get('url')
            },
            "extraction_info": {
                "extraction_date": datetime.now().isoformat(),
                "extractor_version": "1.0",
                "model_used": attack_analysis.get('_metadata', {}).get('model', 'unknown'),
                "tokens_used": attack_analysis.get('_metadata', {}).get('tokens_used', 0),
                "cost": attack_analysis.get('_metadata', {}).get('cost', 0.0)
            },
            "analysis_results": attack_analysis
        }
        
        # Log results
        attacks_found = attack_analysis.get('attacks_found', False)
        attack_count = attack_analysis.get('attack_count', 0)
        paper_classification = attack_analysis.get('paper_classification', 'unknown')
        
        if attacks_found and attack_count > 0:
            print(f"✅ Found {attack_count} implementable attacks")
            for attack in attack_analysis.get('attacks', []):
                print(f"  - {attack.get('name', 'Unnamed')}: {attack.get('type', 'unknown')} ({attack.get('difficulty', 'unknown')} difficulty)")
        else:
            print(f"ℹ️  No direct attacks found - Paper classified as: {paper_classification}")
            
            # Show alternative value
            theoretical_count = len(attack_analysis.get('theoretical_concepts', []))
            evaluation_count = len(attack_analysis.get('evaluation_criteria', []))
            defensive_count = len(attack_analysis.get('defensive_insights', []))
            
            if theoretical_count > 0:
                print(f"💡 Found {theoretical_count} theoretical concepts for potential implementation")
            if evaluation_count > 0:
                print(f"📊 Found {evaluation_count} evaluation criteria for attack testing")
            if defensive_count > 0:
                print(f"🛡️  Found {defensive_count} defensive insights revealing attack vectors")
        
        return structured_result
    
    def process_paper_by_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Process a single paper for attack extraction using URL.
        
        Args:
            url: URL of the paper to process
            
        Returns:
            Dictionary containing attack extraction results, or None if paper not found
        """
        # Initialize Qdrant client
        try:
            client = init_qdrant_client()
        except Exception as e:
            print(f"❌ Error initializing Qdrant: {str(e)}")
            return None
        
        # Get paper from database
        paper_data = get_paper_by_url(client, url)
        if not paper_data:
            print(f"❌ Paper {url} not found in database. Please run paper_read.py first.")
            return None
        
        # Extract attacks
        attack_results = self.extract_attacks_from_paper(paper_data)
        
        # Update paper in database with attack information
        try:
            update_paper_attacks(client, url, attack_results)
            print(f"✅ Updated paper {url} with attack analysis")
        except Exception as e:
            print(f"⚠️  Warning: Could not update paper in database: {str(e)}")
        
        return attack_results

    def process_paper(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """Process a single paper for attack extraction (deprecated - use process_paper_by_url instead).
        
        Args:
            arxiv_id: ArXiv ID of the paper to process
            
        Returns:
            Dictionary containing attack extraction results, or None if paper not found
        """
        # Convert to URL and use the URL-based function
        url = f"https://arxiv.org/abs/{arxiv_id}"
        return self.process_paper_by_url(url)
    
    def process_all_papers(self) -> List[Dict[str, Any]]:
        """Process all papers in the database for attack extraction.
        
        Returns:
            List of attack extraction results
        """
        # Initialize Qdrant client
        try:
            client = init_qdrant_client()
        except Exception as e:
            print(f"❌ Error initializing Qdrant: {str(e)}")
            return []
        
        # Get all papers
        papers = get_all_papers(client)
        if not papers:
            print("ℹ️  No papers found in database")
            return []
        
        print(f"📚 Processing {len(papers)} papers for attack extraction...")
        
        results = []
        for i, paper in enumerate(papers, 1):
            print(f"\n[{i}/{len(papers)}] Processing: {paper.get('title', 'Unknown')}")
            
            # Skip if already processed (has attack analysis)
            if paper.get('attack_analysis'):
                print("✅ Already processed - skipping")
                continue
            
            try:
                attack_results = self.extract_attacks_from_paper(paper)
                results.append(attack_results)
                
                # Update paper in database
                arxiv_id = paper.get('arxiv_id')
                if arxiv_id:
                    update_paper_attacks(client, arxiv_id, attack_results)
                    
            except Exception as e:
                print(f"❌ Error processing paper: {str(e)}")
                continue
        
        return results


def main():
    """Main function to handle command line arguments and process papers."""
    parser = argparse.ArgumentParser(
        description="Extract attack techniques from AI security papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python attack_extractor.py --paper-id 2411.14133
  python attack_extractor.py --all-papers
  python attack_extractor.py --paper-id 2411.14133 --output attacks.json
        """
    )
    
    parser.add_argument(
        "--paper-id",
        help="ArXiv ID of specific paper to process"
    )
    
    parser.add_argument(
        "--all-papers",
        action="store_true",
        help="Process all papers in the database"
    )
    
    parser.add_argument(
        "--output",
        help="Output file to save results (JSON format)"
    )
    
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LLM model to use for extraction (default: {DEFAULT_MODEL})"
    )
    
    args = parser.parse_args()
    
    # Check for required environment variables
    if not OPENROUTER_API_KEY:
        print("❌ OPENROUTER_API_KEY is not set. Please update your .env file.")
        exit(1)
    
    # Validate arguments
    if not args.paper_id and not args.all_papers:
        print("❌ Please specify either --paper-id or --all-papers")
        exit(1)
    
    if args.paper_id and args.all_papers:
        print("❌ Please specify either --paper-id or --all-papers, not both")
        exit(1)
    
    print("🚀 Attack Extractor - Analyzing papers for implementable attacks")
    
    # Check rate limit status
    print("\n📊 Checking rate limit status...")
    check_rate_limit_status()
    
    # Initialize extractor
    extractor = AttackExtractor(OPENROUTER_API_KEY)
    
    # Process papers
    results = []
    
    if args.paper_id:
        print(f"\n🔍 Processing single paper: {args.paper_id}")
        result = extractor.process_paper(args.paper_id)
        if result:
            results.append(result)
    
    elif args.all_papers:
        print(f"\n📚 Processing all papers in database")
        results = extractor.process_all_papers()
    
    # Save results if output file specified
    if args.output and results:
        try:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")
        except Exception as e:
            print(f"❌ Error saving results: {str(e)}")
    
    # Summary
    if results:
        total_papers = len(results)
        papers_with_attacks = sum(1 for r in results if r.get('analysis_results', {}).get('attacks_found', False))
        total_attacks = sum(r.get('analysis_results', {}).get('attack_count', 0) for r in results)
        
        print(f"\n📊 Summary:")
        print(f"  Papers processed: {total_papers}")
        print(f"  Papers with attacks: {papers_with_attacks}")
        print(f"  Total attacks found: {total_attacks}")
        print(f"  Papers without attacks: {total_papers - papers_with_attacks}")
        
        if total_attacks > 0:
            print(f"\n✅ Ready to generate promptfoo configurations!")
            print(f"💡 Next step: Run promptfoo_generator.py to create test configurations")
        else:
            print(f"\nℹ️  No direct attacks found, but papers may contain valuable insights")
            print(f"💡 Check theoretical concepts and evaluation criteria for manual implementation")
    else:
        print(f"\n❌ No results generated")


if __name__ == "__main__":
    main()
