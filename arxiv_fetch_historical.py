import os
import time
import json
import feedparser
import openai
import requests
from datetime import datetime, UTC
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME")

CUTOFF_DATE = os.getenv("CUTOFF_DATE", "2022-01-01")  # customizable cutoff
MAX_RESULTS = 1000
CHUNK_SIZE = 100
LAST_INDEX_FILE = ".last_index"
ARXIV_QUERY = (
    "(all:\"AI security\" OR all:adversarial OR all:\"prompt injection\" OR all:red teaming) "
    "AND (cat:cs.CR OR cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV OR cat:eess.AS OR cat:stat.ML)"
)

# --- INIT ---
client = openai.OpenAI(api_key=OPENAI_API_KEY)
CUTOFF_DATE = datetime.strptime(CUTOFF_DATE, "%Y-%m-%d")

# --- GPT FUNCTION ---
def summarize_and_tag(title, summary, link):
    prompt = f"""
You are an AI assistant filtering arXiv papers for relevance to AI security.

Evaluate the following paper. Only return `relevant: true` if the paper clearly addresses **AI security**, including but not limited to adversarial attacks, prompt injection, jailbreaks, LLM misuse, red teaming, model theft, evasion, poisoning, or robustness under attack.

Avoid false positives. If the paper is not clearly about these topics, set `relevant: false`.

Additionally, return a numeric relevance level between 1 (barely relevant) and 5 (extremely relevant).

Return JSON only in the following format:
{{
  "relevant": true or false,
  "summary": ["..."],
  "tags": ["..."],
  "relevance": 1 to 5
}}

Title: {title}
Abstract: {summary}
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

# --- AIRTABLE PUSH ---
def send_to_airtable(entry):
    title = entry['title']
    check_url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}?filterByFormula=URL='{entry['url']}'"
    headers = {
        "Authorization": f"Bearer {AIRTABLE_TOKEN}",
        "Content-Type": "application/json",
    }
    check = requests.get(check_url, headers=headers)
    if check.ok and check.json().get("records"):
        print(f"⚠️ Already in Airtable: {title}")
        return

    data = {
        "fields": {
            "Title": title,
            "URL": entry['url'],
            "Summary": "\n".join(entry['summary']),
            "Tags": ", ".join(entry['tags']),
            "Authors": ", ".join(entry['authors']),
            "Relevance": entry['relevance'],
            "Date": entry['date']
        }
    }
    res = requests.post(f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}", headers=headers, json=data)
    if res.ok:
        print(f"✅ Added: {title}")
    else:
        print(f"❌ Failed to add: {title}\n{res.text}")

# --- FETCH LOOP ---
def fetch_and_process():
    base_url = "http://export.arxiv.org/api/query"
    start = 0
    if os.path.exists(LAST_INDEX_FILE):
        with open(LAST_INDEX_FILE, "r") as f:
            start = int(f.read().strip())

    processed = 0

    while processed < MAX_RESULTS:
        print(f"\n🔎 Fetching {start}–{start+CHUNK_SIZE}...")
        query = f"search_query={quote_plus(ARXIV_QUERY)}&start={start}&max_results={CHUNK_SIZE}&sortBy=submittedDate&sortOrder=descending"
        url = f"{base_url}?{query}"
        feed = feedparser.parse(url)

        if not feed.entries:
            print("No more results.")
            break

        for entry in feed.entries:
            published = datetime.strptime(entry.published[:10], "%Y-%m-%d")
            if published < CUTOFF_DATE:
                print(f"🛑 Reached cutoff date: {entry.published}")
                return

            authors = [a.name for a in entry.authors] if hasattr(entry, 'authors') else []
            result = summarize_and_tag(entry.title, entry.summary, entry.link)

            if result.get("relevant"):
                send_to_airtable({
                    "title": entry.title,
                    "url": entry.link,
                    "summary": result['summary'],
                    "tags": result['tags'],
                    "relevance": result['relevance'],
                    "authors": authors,
                    "date": published.strftime("%Y-%m-%d")
                })
            else:
                print(f"🚫 Not relevant: {entry.title}")
            time.sleep(1)

        start += CHUNK_SIZE
        processed += len(feed.entries)

        # Save progress
        with open(LAST_INDEX_FILE, "w") as f:
            f.write(str(start))

        time.sleep(3)  # Respect arXiv rate limit

if __name__ == "__main__":
    fetch_and_process()
