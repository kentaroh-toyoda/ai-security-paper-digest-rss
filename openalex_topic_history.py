import os
import time
import json
import requests
import openai
import argparse
from datetime import datetime
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME")
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL")

START_DATE = os.getenv("START_DATE", "2022-01-01")  # format YYYY-MM-DD
MAX_PAGES = int(os.getenv("MAX_PAGES", 10))

HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_TOKEN}",
    "Content-Type": "application/json",
}

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# --- GPT TAGGING FUNCTION ---
def summarize_and_tag(title, abstract, link, topic):
    prompt = f"""
You are an AI assistant filtering research papers for relevance to AI security, focused on the topic: {topic}.

Evaluate the following paper and return strict JSON:
{{
  "relevant": true or false,
  "summary": ["..."],
  "tags": ["..."],
  "relevance": 1 to 5
}}

---
Title: {title}
Abstract: {abstract}
URL: {link}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"❌ GPT error on: {title}\n{e}")
        return {"relevant": False}

# --- Check if paper already exists by title ---
def paper_exists(title):
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"
    params = {"filterByFormula": f"{{Title}}='{title.replace("'", "\\'")}'"}
    res = requests.get(url, headers=HEADERS, params=params)
    if res.ok:
        records = res.json().get("records", [])
        return len(records) > 0
    return False

# --- AIRTABLE PUSH ---
def send_to_airtable(entry, topic):
    if paper_exists(entry['title']):
        print(f"⚠️ Duplicate title, skipping: {entry['title']}")
        return

    data = {
        "fields": {
            "Title": entry['title'],
            "URL": entry['url'],
            "Summary": "\n".join(entry['summary']),
            "Tags": ", ".join(entry['tags']),
            "Authors": ", ".join(entry['authors']),
            "Relevance": entry['relevance'],
            "Date": entry['date'],
            "Topic": topic,
            "Source": "OpenAlex"
        }
    }
    response = requests.post(f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}", headers=HEADERS, json=data)
    if response.ok:
        print(f"✅ Added: {entry['title']}")
    else:
        print(f"❌ Failed to add: {entry['title']}\n{response.text}")

# --- FETCH LOOP ---
def fetch_openalex_results(topic):
    base_url = "https://api.openalex.org/works"
    params = {
        "search": topic,
        "filter": f"from_publication_date:{START_DATE}",
        "per-page": 50,
        "sort": "publication_date:desc"
    }

    page = 1
    processed = 0

    while page <= MAX_PAGES:
        print(f"🔎 Fetching OpenAlex page {page} for topic: '{topic}'...")
        res = requests.get(base_url, params={**params, "page": page})
        if not res.ok:
            print("❌ Request failed:", res.text)
            break

        data = res.json()
        works = data.get("results", [])
        if not works:
            print("No more results.")
            break

        for work in works:
            title = work.get("title", "")
            abstract = work.get("abstract_inverted_index", {})
            abstract_text = " ".join(abstract.keys()) if isinstance(abstract, dict) else ""
            authors = [a['author']['display_name'] for a in work.get("authorships", [])]
            url = work.get("primary_location", {}).get("landing_page_url", work.get("id"))
            date = work.get("publication_date", "")

            result = summarize_and_tag(title, abstract_text, url, topic)
            if result.get("relevant"):
                send_to_airtable({
                    "title": title,
                    "url": url,
                    "summary": result['summary'],
                    "tags": result['tags'],
                    "relevance": result['relevance'],
                    "authors": authors,
                    "date": date
                }, topic)
            else:
                print(f"🚫 Not relevant: {title}")
            time.sleep(1)

        page += 1
        time.sleep(2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, required=True, help="Research topic to query OpenAlex for")
    args = parser.parse_args()

    fetch_openalex_results(args.topic)
