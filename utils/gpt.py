# utils/gpt.py

import os
import json
import openai
from openai import OpenAI
from dotenv import load_dotenv

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


def assess_relevance_and_tags(text, api_key, return_usage=False):
    client = OpenAI(api_key=api_key)
    config = get_gpt_config()

    system_prompt = (
        "You are an AI assistant that filters research papers. "
        "Given the title, abstract, and URL of a paper, determine whether it's relevant to AI safety, security, "
        "or responsible AI development, including topics like LLM red teaming, adversarial attacks, "
        "data poisoning, robustness, or any other aspects of making AI systems more reliable and safe. "
        "If relevant, provide a summary and generate appropriate tags (e.g., 'red teaming', 'robustness', 'LLM')."
    )
    user_prompt = f"""Here is a research paper.

{text}

Please output JSON with the following format:

{{
  "relevant": true or false,
  "summary": [bullet points about the key ideas],
  "tags": [short tags for topic classification],
  "relevance_score": integer from 1 (not very relevant) to 5 (very relevant),
  "reason": "brief explanation of why this paper is considered relevant or not"
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
    print(f"\nGPT Response for {text.split('Title: ')[1].split('\n')[0]}:")
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
                "relevant": False,
                "summary": ["❌ Failed to parse GPT output."],
                "tags": ["error"],
                "relevance_score": 1,
                "reason": f"Error parsing response: {str(e)}"
            }

    if return_usage:
        return parsed, response.usage
    return parsed


def assess_paper_quality(title, fulltext_html, api_key, return_usage=False):
    client = OpenAI(api_key=api_key)
    config = get_gpt_config()

    system_prompt = (
        "You are an AI reviewer. Evaluate a research paper based on its clarity, novelty, and significance. "
        "Also assess if it is worth trying in practice (try-worthiness), and extract justification and any code repository links."
    )
    user_prompt = f"""
Title: {title}

Please assess the following full paper (HTML version below) and output JSON:

{{
  "Clarity": score from 1 to 5,
  "Novelty": score from 1 to 5,
  "Significance": score from 1 to 5,
  "Try-worthiness": true or false,
  "Justification": "why this score was given",
  "Code repository": "GitHub URL or similar, if found"
}}

HTML Full Paper:
==================
{fulltext_html[:12000]}  <!-- Truncate if too long -->
"""

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
    print(f"\nGPT Quality Assessment for {title}:")
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
                "Code repository": ""
            }

    if return_usage:
        return parsed, response.usage
    return parsed
