# daily_ai_security_digest.py

import os
import time
import json
import requests
import openai
import feedparser
from datetime import datetime, timedelta, timezone
from feedgen.feed import FeedGenerator
from dotenv import load_dotenv
from dateutil import parser

load_dotenv()

# --- CONFIGURATION ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME")
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL")

HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_TOKEN}",
    "Content-Type": "application/json",
}

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# --- GPT TAGGING FUNCTION ---
def summarize_and_tag(title, abstract, link):
    prompt = f"""
You are an AI assistant filtering research papers for relevance to AI security.

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

# --- FETCH FROM OPENALEX ---
def fetch_openalex_today():
    base_url = "https://api.openalex.org/works"
    since = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
    params = {
        "filter": f"from_publication_date:{since}",
        "per-page": 50,
        "sort": "publication_date:desc",
        "mailto": OPENALEX_EMAIL,
    }
    res = requests.get(base_url, params=params)
    if res.ok:
        return res.json().get("results", [])
    else:
        print("❌ OpenAlex request failed:", res.text)
        return []

# --- FETCH FROM MULTIPLE ARXIV FEEDS ---
def fetch_arxiv_today():
    feeds = [
        "https://export.arxiv.org/rss/cs.AI",
        "https://export.arxiv.org/rss/cs.LG",
        "https://export.arxiv.org/rss/cs.CV",
        "https://export.arxiv.org/rss/cs.CL"
    ]
    recent = []
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    for feed_url in feeds:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries:
            pub = parser.parse(entry.published)
            if pub > yesterday:
                recent.append(entry)
    return recent

# --- Check for existing papers in Airtable ---
def paper_exists_by_id(url):
    id_fragments = [url.split("/")[-1].strip()]
    if "arxiv.org" in url:
        if "abs" in url:
            id_fragments.append(url.split("abs/")[-1].strip())
        elif "pdf" in url:
            id_fragments.append(url.split("pdf/")[-1].replace(".pdf", "").strip())
    airtable_url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"

    for fragment in id_fragments:
        filter_formula = f"SEARCH('{fragment}', {{URL}})"
        params = {"filterByFormula": filter_formula}
        res = requests.get(airtable_url, headers=HEADERS, params=params)
        if res.ok:
            records = res.json().get("records", [])
            if len(records) > 0:
                return True
    return False

# --- Send Paper to Airtable ---
def send_to_airtable(entry):
    if paper_exists_by_id(entry['url']):
        print(f"⚠️ Already exists in Airtable, skipping: {entry['title']}")
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
            "Source": entry['source']
        }
    }
    response = requests.post(
        f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}",
        headers=HEADERS,
        json=data
    )
    if response.ok:
        print(f"✅ Added to Airtable: {entry['title']}")
    else:
        print(f"❌ Airtable push failed for: {entry['title']}\n{response.text}")

# --- Generate RSS Feed ---
def generate_rss(papers):
    fg = FeedGenerator()
    fg.title("AI Security Paper Digest")
    fg.link(href="https://kentaroh-toyoda.github.io/ai-security-paper-digest-rss/rss.xml")
    fg.description("Daily digest of AI security research papers")
    fg.language("en")

    for entry in papers:
        fe = fg.add_entry()
        fe.title(entry['title'])
        fe.link(href=entry['url'])
        fe.description("\n".join(entry['summary']) + f"\n\nTags: {', '.join(entry['tags'])}")
        pub_date = parser.isoparse(entry['date']).astimezone(timezone.utc)
        fe.pubDate(pub_date)

    fg.rss_file("rss.xml")
    print("✅ RSS file generated.")

# --- RUN SCRIPT ---
if __name__ == "__main__":
    print("🔍 Fetching today's OpenAlex and arXiv papers...")
    all_papers = []
    oa_papers = fetch_openalex_today()
    for work in oa_papers:
        title = work.get("title", "")
        abstract = work.get("abstract_inverted_index", {})
        abstract_text = " ".join(abstract.keys()) if isinstance(abstract, dict) else ""
        authors = [a['author']['display_name'] for a in work.get("authorships", [])]
        url = work.get("primary_location", {}).get("landing_page_url", work.get("id"))
        date = work.get("publication_date", datetime.now(timezone.utc).isoformat())

        if paper_exists_by_id(url):
            print(f"⚠️ Duplicate skipped: {title}")
            continue

        result = summarize_and_tag(title, abstract_text, url)
        if result.get("relevant"):
            print(f"✅ Relevant (OpenAlex): {title}")
            paper_entry = {
                "title": title,
                "summary": result['summary'],
                "tags": result['tags'],
                "url": url,
                "date": date,
                "authors": authors,
                "relevance": result['relevance'],
                "source": "OpenAlex"
            }
            all_papers.append(paper_entry)
            send_to_airtable(paper_entry)
        else:
            print(f"🚫 Not relevant: {title}")
        time.sleep(1)

    arxiv_papers = fetch_arxiv_today()
    for entry in arxiv_papers:
        title = entry.title
        abstract = entry.summary
        url = entry.link
        date = parser.parse(entry.published).isoformat()
        authors = [a.name for a in entry.authors] if hasattr(entry, 'authors') else []

        if paper_exists_by_id(url):
            print(f"⚠️ Duplicate skipped: {title}")
            continue

        result = summarize_and_tag(title, abstract, url)
        if result.get("relevant"):
            print(f"✅ Relevant (arXiv): {title}")
            paper_entry = {
                "title": title,
                "summary": result['summary'],
                "tags": result['tags'],
                "url": url,
                "date": date,
                "authors": authors,
                "relevance": result['relevance'],
                "source": "arXiv"
            }
            all_papers.append(paper_entry)
            send_to_airtable(paper_entry)
        else:
            print(f"🚫 Not relevant: {title}")
        time.sleep(1)

    if all_papers:
        generate_rss(all_papers)
    else:
        print("ℹ️ No relevant papers found today.")
