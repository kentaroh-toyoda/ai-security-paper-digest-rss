import os
import json
import requests
import openai
from dotenv import load_dotenv

load_dotenv()

AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SEARCH_TERM = "LLM red teaming"
MIN_RELEVANCE = 1
GPT_CACHE_FILE = ".gpt_filter_cache.json"
gpt_cost_per_1k = 0.01

total_tokens_used = 0
headers = {
    "Authorization": f"Bearer {AIRTABLE_TOKEN}",
    "Content-Type": "application/json",
}
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Load existing cache
if os.path.exists(GPT_CACHE_FILE):
    with open(GPT_CACHE_FILE, "r") as f:
        gpt_filter_cache = json.load(f)
else:
    gpt_filter_cache = {}

def fetch_all_records():
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"
    all_records = []
    offset = None

    while True:
        params = {"pageSize": 100}
        if offset:
            params["offset"] = offset

        res = requests.get(url, headers=headers, params=params)
        if not res.ok:
            print("❌ Airtable error:", res.text)
            break

        data = res.json()
        all_records.extend(data.get("records", []))
        offset = data.get("offset")

        if not offset:
            break

    return all_records

def gpt_filter(paper, topic):
    global total_tokens_used
    cache_key = paper['url']
    if cache_key in gpt_filter_cache:
        return gpt_filter_cache[cache_key]

    prompt = f"""
You are an expert assistant helping filter research papers relevant to the topic: "{topic}".

Below is a research paper with its title, abstract summary, and tags. Based on the topic above, decide if the paper is related. Only respond with strict JSON:

{{
  "relevant": true or false,
  "reason": "one-line explanation"
}}

---
Title: {paper['title']}
Summary: {paper['summary']}
Tags: {paper['tags']}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        total_tokens_used += response.usage.total_tokens
        result = json.loads(response.choices[0].message.content.strip())
        gpt_filter_cache[cache_key] = result
        return result
    except Exception as e:
        print(f"❌ GPT error for paper: {paper['title']}\n{e}")
        return {"relevant": False, "reason": "error"}

def search_relevant_papers(records, topic, min_relevance):
    matches = []
    for r in records:
        fields = r.get("fields", {})
        relevance = fields.get("Relevance", 0)
        if not isinstance(relevance, (int, float)) or relevance < min_relevance:
            continue

        paper = {
            "title": fields.get("Title", ""),
            "summary": fields.get("Summary", ""),
            "tags": fields.get("Tags", ""),
            "relevance": relevance,
            "date": fields.get("Date"),
            "authors": fields.get("Authors"),
            "url": fields.get("URL"),
        }

        gpt_result = gpt_filter(paper, topic)
        if gpt_result.get("relevant"):
            paper["reason"] = gpt_result.get("reason")
            matches.append(paper)

    return matches

if __name__ == "__main__":
    all_papers = fetch_all_records()
    filtered = search_relevant_papers(all_papers, SEARCH_TERM, MIN_RELEVANCE)
    print(f"\n🔎 Found {len(filtered)} papers GPT-matched for topic '{SEARCH_TERM}'\n")
    for p in filtered:
        print(f"- {p['title']} (Relevance: {p['relevance']})\n  {p['reason']}\n  {p['url']}")

    print(f"\n🧾 Total GPT tokens used: {total_tokens_used:,}")
    print(f"💵 Estimated cost: ${total_tokens_used / 1000 * gpt_cost_per_1k:.4f} USD")

    # Save updated cache
    with open(GPT_CACHE_FILE, "w") as f:
        json.dump(gpt_filter_cache, f, indent=2)