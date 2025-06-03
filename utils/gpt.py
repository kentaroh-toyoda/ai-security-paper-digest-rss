# utils/gpt.py

import openai

def assess_relevance_and_tags(text, api_key, return_usage=False):
    openai.api_key = api_key
    system_prompt = (
        "You are an AI assistant that filters research papers. "
        "Given the title, abstract, and URL of a paper, determine whether it's relevant to AI security, "
        "such as LLM red teaming, adversarial attacks, data poisoning, or robustness. "
        "If relevant, provide a summary and generate appropriate tags (e.g., 'red teaming', 'robustness', 'LLM')."
    )
    user_prompt = f"""Here is a research paper.

{text}

Please output JSON with the following format:

{{
  "relevant": true or false,
  "summary": [bullet points about the key ideas],
  "tags": [short tags for topic classification],
  "relevance_score": integer from 1 (not very relevant) to 5 (very relevant)
}}"""

    response = openai.ChatCompletion.create(
        model="gpt-4.1",
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    content = response["choices"][0]["message"]["content"]
    try:
        parsed = eval(content)
    except Exception:
        parsed = {
            "relevant": False,
            "summary": ["❌ Failed to parse GPT output."],
            "tags": ["error"],
            "relevance_score": 1
        }
    if return_usage:
        return parsed, response["usage"]
    return parsed

def assess_paper_quality(title, fulltext_html, api_key, return_usage=False):
    openai.api_key = api_key
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

    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    content = response["choices"][0]["message"]["content"]
    try:
        parsed = eval(content)
    except Exception:
        parsed = {
            "Clarity": 1,
            "Novelty": 1,
            "Significance": 1,
            "Try-worthiness": False,
            "Justification": "❌ Failed to parse GPT output.",
            "Code repository": ""
        }
    if return_usage:
        return parsed, response["usage"]
    return parsed