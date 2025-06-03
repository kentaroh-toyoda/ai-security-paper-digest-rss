# utils/gpt.py

import os
import json
import openai
from openai import OpenAI
from dotenv import load_dotenv
from typing import Tuple, Dict, Any

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


def assess_relevance_and_tags(text: str, api_key: str, temperature: float = 0.1, model: str = "gpt-4.1") -> Tuple[Dict[str, Any], int]:
    """Assess if a paper is relevant and extract tags using GPT."""
    client = OpenAI(api_key=api_key)

    system_prompt = """You are an expert in AI security and safety. Your task is to assess if a paper is relevant to AI security, safety, or red teaming, and extract relevant tags.

Relevance criteria:
- Papers about AI security, safety, or red teaming
- Papers about adversarial attacks on AI systems
- Papers about jailbreaking or prompt injection
- Papers about AI alignment or robustness
- Papers about privacy or security in AI systems
- Papers about AI governance or policy
- Papers about AI ethics or responsible AI
- Papers about AI risk assessment or threat modeling

If the paper is relevant:
1. Provide a brief summary (2-4 bullet points)
2. Extract 3-5 relevant tags
3. Rate relevance from 1-5 (5 being most relevant)
4. Provide a brief reason for the relevance rating

If the paper is not relevant, simply respond with {"relevant": false}.

Respond in JSON format:
{
    "relevant": true/false,
    "summary": ["point 1", "point 2", ...],
    "tags": ["tag1", "tag2", ...],
    "relevance_score": 1-5,
    "reason": "brief explanation"
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
