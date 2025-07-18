#!/usr/bin/env python3
"""
Test Script for Attack Implementation System

This script demonstrates the complete workflow of the attack implementation system
using a sample paper or existing papers in the database.

Usage:
    python test_attack_system.py
    python test_attack_system.py --demo-paper https://arxiv.org/abs/2411.14133
"""

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

from attack_extractor import AttackExtractor
from promptfoo_generator import PromptfooGenerator
from utils.qdrant import init_qdrant_client, get_all_papers

# Load environment variables
load_dotenv()

def test_complete_workflow(demo_paper_url=None):
    """Test the complete attack implementation workflow."""
    
    print("🚀 Testing AI Security Paper Attack Implementation System")
    print("=" * 60)
    
    # Check environment variables
    required_vars = ["OPENROUTER_API_KEY", "QDRANT_API_URL", "QDRANT_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please update your .env file with the required credentials.")
        return False
    
    print("✅ Environment variables configured")
    
    # Test database connection
    try:
        client = init_qdrant_client()
        papers = get_all_papers(client)
        print(f"✅ Database connection successful - Found {len(papers)} papers")
    except Exception as e:
        print(f"❌ Database connection failed: {str(e)}")
        return False
    
    # If demo paper URL provided, process it first
    if demo_paper_url:
        print(f"\n📥 Processing demo paper: {demo_paper_url}")
        try:
            import subprocess
            result = subprocess.run([
                "python", "store_paper.py", "--url", demo_paper_url
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Demo paper processed successfully")
            else:
                print(f"⚠️  Demo paper processing had issues: {result.stderr}")
        except Exception as e:
            print(f"❌ Error processing demo paper: {str(e)}")
    
    # Test attack extraction
    print(f"\n🔍 Testing Attack Extraction")
    print("-" * 30)
    
    try:
        extractor = AttackExtractor(os.getenv("OPENROUTER_API_KEY"))
        
        if len(papers) == 0:
            print("ℹ️  No papers in database to test with")
            return True
        
        # Test with first paper
        test_paper = papers[0]
        print(f"Testing with paper: {test_paper.get('title', 'Unknown')}")
        
        # Extract attacks
        attack_results = extractor.extract_attacks_from_paper(test_paper)
        
        if attack_results:
            analysis = attack_results.get('analysis_results', {})
            attacks_found = analysis.get('attacks_found', False)
            attack_count = analysis.get('attack_count', 0)
            
            print(f"✅ Attack extraction completed")
            print(f"   Attacks found: {attacks_found}")
            print(f"   Attack count: {attack_count}")
            print(f"   Paper classification: {analysis.get('paper_classification', 'unknown')}")
            
            if attacks_found:
                print(f"   Attack types found:")
                for attack in analysis.get('attacks', []):
                    print(f"     - {attack.get('name', 'Unnamed')}: {attack.get('type', 'unknown')}")
            else:
                theoretical_count = len(analysis.get('theoretical_concepts', []))
                evaluation_count = len(analysis.get('evaluation_criteria', []))
                defensive_count = len(analysis.get('defensive_insights', []))
                print(f"   Alternative value:")
                print(f"     - Theoretical concepts: {theoretical_count}")
                print(f"     - Evaluation criteria: {evaluation_count}")
                print(f"     - Defensive insights: {defensive_count}")
        else:
            print("❌ Attack extraction failed")
            return False
            
    except Exception as e:
        print(f"❌ Attack extraction error: {str(e)}")
        return False
    
    # Test promptfoo configuration generation
    print(f"\n🔧 Testing Promptfoo Configuration Generation")
    print("-" * 45)
    
    try:
        generator = PromptfooGenerator("./test_promptfoo_configs")
        
        # Generate config for the test paper
        config_path = generator.generate_config_for_paper(test_paper)
        
        if config_path:
            print(f"✅ Configuration generated: {config_path}")
            
            # Check if files were created
            config_dir = Path("./test_promptfoo_configs")
            configs = list((config_dir / "configs").glob("*.yaml"))
            strategies = list((config_dir / "strategies").glob("*.js"))
            
            print(f"   Generated files:")
            print(f"     - Config files: {len(configs)}")
            print(f"     - Strategy files: {len(strategies)}")
            
            # Show file contents preview
            if configs:
                print(f"\n📄 Sample configuration preview:")
                with open(configs[0], 'r') as f:
                    lines = f.readlines()[:10]  # First 10 lines
                    for line in lines:
                        print(f"     {line.rstrip()}")
                    if len(f.readlines()) > 10:
                        print(f"     ... (truncated)")
            
            if strategies:
                print(f"\n📄 Sample strategy preview:")
                with open(strategies[0], 'r') as f:
                    lines = f.readlines()[:15]  # First 15 lines
                    for line in lines:
                        print(f"     {line.rstrip()}")
                    if len(f.readlines()) > 15:
                        print(f"     ... (truncated)")
        else:
            print("ℹ️  No configuration generated (no attacks found)")
            
    except Exception as e:
        print(f"❌ Configuration generation error: {str(e)}")
        return False
    
    # Test batch processing (if multiple papers)
    if len(papers) > 1:
        print(f"\n📚 Testing Batch Processing")
        print("-" * 25)
        
        try:
            # Test with first 3 papers to avoid excessive API calls
            test_papers = papers[:3]
            print(f"Testing batch processing with {len(test_papers)} papers...")
            
            batch_results = []
            for i, paper in enumerate(test_papers, 1):
                print(f"[{i}/{len(test_papers)}] Processing: {paper.get('title', 'Unknown')[:50]}...")
                try:
                    result = extractor.extract_attacks_from_paper(paper)
                    batch_results.append(result)
                except Exception as e:
                    print(f"   ❌ Error: {str(e)}")
                    continue
            
            print(f"✅ Batch processing completed")
            print(f"   Papers processed: {len(batch_results)}")
            
            # Generate batch configs
            batch_configs = []
            for result in batch_results:
                if result and result.get('analysis_results', {}).get('attacks_found', False):
                    config_path = generator.generate_config_for_paper(result.get('paper_info', {}))
                    if config_path:
                        batch_configs.append(config_path)
            
            if len(batch_configs) > 1:
                master_config = generator.generate_master_config(batch_configs)
                print(f"✅ Master configuration generated: {master_config}")
            
        except Exception as e:
            print(f"❌ Batch processing error: {str(e)}")
    
    # Summary and next steps
    print(f"\n📊 Test Summary")
    print("=" * 15)
    print("✅ All core components tested successfully!")
    print()
    print("🎯 Next Steps:")
    print("1. Review generated configurations in ./test_promptfoo_configs/")
    print("2. Install promptfoo: npm install -g promptfoo")
    print("3. Modify target models in config files as needed")
    print("4. Run tests: promptfoo eval -c <config_file>")
    print("5. View results: promptfoo view")
    print()
    print("📖 For detailed usage instructions, see README_ATTACK_IMPLEMENTATION.md")
    
    return True


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test the AI Security Paper Attack Implementation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_attack_system.py
  python test_attack_system.py --demo-paper https://arxiv.org/abs/2411.14133
        """
    )
    
    parser.add_argument(
        "--demo-paper",
        help="URL of a demo paper to process first (optional)"
    )
    
    args = parser.parse_args()
    
    # Run the test
    success = test_complete_workflow(args.demo_paper)
    
    if success:
        print("\n🎉 All tests passed! The attack implementation system is ready to use.")
        exit(0)
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        exit(1)


if __name__ == "__main__":
    main()
