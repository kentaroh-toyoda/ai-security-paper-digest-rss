# utils/gpt.py

import os
import json
import openai
from openai import OpenAI
from dotenv import load_dotenv
from typing import Tuple, Dict, Any
from datetime import datetime

# Load environment variables
load_dotenv()

# Default configuration
DEFAULT_MODEL = "gpt-4.1"
DEFAULT_TEMPERATURE = 0.1


def get_gpt_config():
    """Get GPT configuration from environment variables with defaults."""
    return {
        "model": os.getenv("GPT_MODEL", DEFAULT_MODEL),
        "temperature": float(os.getenv("GPT_TEMPERATURE", DEFAULT_TEMPERATURE))
    }


def generate_search_keywords(topic: str, api_key: str) -> str:
    """Generate optimized search keywords using GPT.

    Args:
        topic: The topic to generate search keywords for
        api_key: OpenAI API key

    Returns:
        str: Optimized search query string
    """
    client = OpenAI(api_key=api_key)

    system_prompt = """You are an expert in academic paper search. Your task is to generate optimized search keywords for finding relevant papers in OpenAlex.

Consider:
1. Technical terms and jargon in the field
2. Alternative phrasings and synonyms
3. Related concepts and methodologies
4. Common abbreviations and acronyms

Generate a search query that:
- Uses OR operators to combine related terms
- Uses AND operators to ensure relevance
- Includes quotation marks for exact phrases
- Excludes irrelevant terms with NOT
- Is optimized for academic paper search

Example input: "LLM red teaming"
Example output: "large language model" AND ("red teaming" OR "jailbreaking" OR "adversarial prompting") AND (security OR safety) NOT (medical OR healthcare)

Respond with ONLY the search query, no explanations."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": topic}
            ],
            temperature=DEFAULT_TEMPERATURE
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error generating search keywords: {str(e)}")
        return topic


def assess_relevance_and_tags(text: str, api_key: str, temperature: float = 0.1, model: str = "gpt-4.1") -> Tuple[Dict[str, Any], int]:
    """Assess if a paper is relevant and extract tags using GPT."""
    client = OpenAI(api_key=api_key)

    system_prompt = """You are an expert in AI security and safety. Your task is to assess if a paper is relevant to AI security, safety, or red teaming, and extract relevant tags.

IMPORTANT: Be strict in your assessment. Only mark papers as relevant if they DIRECTLY address AI security, safety, or red teaming topics.

Relevance criteria (paper MUST focus on one or more of these):
1. AI Security & Red Teaming:
   - LLM red teaming and jailbreaking
   - Prompt injection and adversarial prompting
   - Model extraction and stealing
   - Data poisoning and backdoor attacks
   - Privacy attacks (membership inference, model inversion)
   - Security vulnerabilities in AI systems

2. AI Safety & Alignment:
   - Robustness against adversarial examples
   - Alignment with human values
   - Preventing harmful outputs
   - Safety evaluation and testing
   - Risk assessment and mitigation

3. AI Governance & Policy:
   - Security standards and best practices
   - Regulatory compliance
   - Security audits and certifications
   - Incident response and monitoring

NOT RELEVANT (examples):
- General AI/ML papers without security focus
- Papers about AI applications without security implications
- Papers about AI ethics without security aspects
- Papers about AI performance or efficiency without security context
- Papers about AI explainability without security focus

Paper Types to Identify (choose ONE that best fits):
- Research Paper: Original research with novel contributions
- Survey/Review: Comprehensive overview of existing work
- Benchmarking: Evaluation and comparison of methods/systems
- Position Paper: Opinion or perspective on a topic
- Other: Any other type not covered above

If the paper is relevant:
1. Provide a brief summary (2-4 bullet points)
2. Extract 3-5 relevant tags
3. Rate relevance from 1-5 (5 being most relevant)
4. Provide a brief reason for the relevance rating
5. Identify the paper type (choose ONE that best fits)

If the paper is not relevant, simply respond with {"relevant": false}.

Respond in JSON format:
{
    "relevant": true/false,
    "summary": ["point 1", "point 2", ...],
    "tags": ["tag1", "tag2", ...],
    "relevance_score": 1-5,
    "reason": "brief explanation",
    "paper_type": "type"
}"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=temperature
        )
        result = response.choices[0].message.content
        token_count = response.usage.total_tokens

        # Parse the response as JSON
        try:
            result_dict = json.loads(result)
            return result_dict, token_count
        except json.JSONDecodeError:
            print(f"❌ Failed to parse GPT response as JSON: {result}")
            return {"relevant": False}, token_count

    except Exception as e:
        print(f"❌ Error calling GPT API: {str(e)}")
        return {"relevant": False}, 0


def assess_paper_quality(metadata: dict, api_key: str, return_usage=False):
    """Assess paper quality using available metadata.

    Args:
        metadata: Dictionary containing paper metadata:
            - title: Paper title
            - abstract: Paper abstract
            - cited_by_count: Number of citations
            - publication_type: Type of publication
            - source: Publication source/venue
            - code_url: URL to code repository if available
            - date: Publication date (YYYY-MM-DD)
        api_key: OpenAI API key
        return_usage: Whether to return token usage information

    Returns:
        dict: Quality assessment results
    """
    client = OpenAI(api_key=api_key)
    config = get_gpt_config()

    # Calculate paper age in months
    pub_date = datetime.strptime(metadata['date'], "%Y-%m-%d")
    now = datetime.now()
    age_months = (now.year - pub_date.year) * 12 + (now.month - pub_date.month)

    # Adjust citation context based on age
    citation_context = ""
    if age_months < 3:
        citation_context = "This is a very recent paper (less than 3 months old), so citation count is not yet meaningful."
    elif age_months < 6:
        citation_context = "This is a recent paper (3-6 months old), so citation count should be considered with caution."
    elif age_months < 12:
        citation_context = "This paper is 6-12 months old, so citation count is starting to become meaningful."
    else:
        citation_context = f"This paper is {age_months} months old, so citation count is a good indicator of impact."

    system_prompt = """You are an AI reviewer. Evaluate a research paper based on its metadata and abstract.
Consider the following factors:
1. Clarity: How well-written and clear is the paper based on the title and abstract?
2. Novelty: How novel is the work based on the title, abstract, and publication venue?
3. Significance: How significant is the work based on citations (considering paper age) and venue?
4. Try-worthiness: Is this paper worth implementing or experimenting with?

Also identify any code repository links if available."""

    user_prompt = f"""Paper Metadata:
Title: {metadata['title']}
Abstract: {metadata['abstract']}
Publication Date: {metadata['date']}
Citations: {metadata['cited_by_count']} ({citation_context})
Publication Type: {metadata['publication_type']}
Source/Venue: {metadata['source']}
Code URL: {metadata['code_url']}

Please assess the paper and output JSON:
{{
  "Clarity": score from 1 to 5,
  "Novelty": score from 1 to 5,
  "Significance": score from 1 to 5,
  "Try-worthiness": true or false,
  "Justification": "brief explanation of the scores",
  "Code repository": "GitHub URL or similar, if found"
}}"""

    response = client.chat.completions.create(
        model=config["model"],
        temperature=config["temperature"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    content = response.choices[0].message.content

    # Print the raw response for debugging
    print(f"\nGPT Quality Assessment for {metadata['title']}:")
    print(content)

    try:
        # Try to parse as JSON first
        parsed = json.loads(content)
    except json.JSONDecodeError:
        try:
            # Fallback to eval if JSON parsing fails
            parsed = eval(content)
        except Exception as e:
            print(f"❌ Error parsing GPT output: {e}")
            parsed = {
                "Clarity": 1,
                "Novelty": 1,
                "Significance": 1,
                "Try-worthiness": False,
                "Justification": f"Error parsing response: {str(e)}",
                "Code repository": metadata.get("code_url", "")
            }

    if return_usage:
        return parsed, response.usage
    return parsed
