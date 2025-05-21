import os
import time
import json
import feedparser
import openai
import requests
from datetime import datetime, UTC
from feedgen.feed import FeedGenerator
from dotenv import load_dotenv

load_dotenv()

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME")

# Config
ARXIV_FEEDS = [
    "https://export.arxiv.org/rss/cs.AI",
    "https://export.arxiv.org/rss/cs.LG",
    "https://export.arxiv.org/rss/cs.CL",
    "https://export.arxiv.org/rss/cs.CR",
    "https://export.arxiv.org/rss/cs.CV",
    "https://export.arxiv.org/rss/eess.AS",
    "https://export.arxiv.org/rss/stat.ML",
]
MAX_FETCH_PER_FEED = 100
MAX_TOTAL_PROCESSED = 1000
RSS_OUTPUT_PATH = "rss.xml"

# Keywords for pre-filtering
KEYWORDS = [
    "adversarial", "attack", "robust", "defense", "jailbreak", 
    "poisoning", "red teaming", "spoofing", "deepfake", "steganography",
    "prompt injection", "model extraction", "exfiltration", "security", 
    "privacy", "tamper", "evasion", "backdoor"
]

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def fetch_arxiv_entries(feed_url, limit=50):
    feed = feedparser.parse(feed_url)
    time.sleep(3)  # Respect arXiv rate limit
    return feed.entries[:limit]

def is_potentially_relevant(entry, keywords):
    text = f"{entry.title} {entry.summary}".lower()
    return any(keyword in text for keyword in keywords)

def summarize_and_tag(entry):
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

Title: {entry.title}
Abstract: {entry.summary}
URL: {entry.link}
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    content = response.choices[0].message.content.strip()

    try:
        result = json.loads(content)
        return result
    except Exception as e:
        print(f"\n❌ Failed to parse response for: {entry.title}")
        print("Raw GPT output:\n", content)
        print("Error:", str(e))
        return {"relevant": False}

def write_rss_feed(entries, output_path="rss.xml"):
    fg = FeedGenerator()
    fg.title("AI Security Paper Digest")
    fg.link(href="https://kentaroh-toyoda.github.io/ai-security-paper-digest-rss/rss.xml")
    fg.description("Summarized AI security papers from arXiv")

    for paper in entries:
        fe = fg.add_entry()
        fe.title(paper['title'])
        fe.link(href=paper['url'])
        summary_html = "<br/>".join(f"• {s}" for s in paper["summary"])
        tags_html = f"Tags: {', '.join(paper['tags'])}" if paper['tags'] else ""
        authors_html = f"Authors: {', '.join(paper['authors'])}" if paper.get('authors') else ""
        relevance_html = f"Relevance: {paper.get('relevance', '?')} / 5"
        link_html = f"<br/><br/><a href='{paper['url']}'>Read on arXiv</a>"
        description = f"{summary_html}<br/><br/>{tags_html}<br/>{authors_html}<br/>{relevance_html}{link_html}"
        fe.description(description)
        fe.pubDate(datetime.now(UTC))

    fg.rss_file(output_path)

def send_to_airtable(title, url, summary, tags, authors, relevance):
    check_url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}?filterByFormula=URL='{url}'"
    headers = {
        "Authorization": f"Bearer {AIRTABLE_TOKEN}",
        "Content-Type": "application/json",
    }
    check = requests.get(check_url, headers=headers)
    if check.ok and check.json().get("records"):
        print(f"⚠️ Skipping duplicate entry in Airtable: {title}")
        return

    data = {
        "fields": {
            "Title": title,
            "URL": url,
            "Summary": "\n".join(summary),
            "Tags": ", ".join(tags),
            "Authors": authors,
            "Relevance": relevance,
            "Date": datetime.now(UTC).strftime("%Y-%m-%d")
        }
    }
    response = requests.post(f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}", headers=headers, json=data)
    if not response.ok:
        print(f"❌ Airtable push failed for: {title}\n{response.text}")
    else:
        print(f"✅ Airtable entry added: {title}")

def main():
    all_entries = []
    for feed_url in ARXIV_FEEDS:
        entries = fetch_arxiv_entries(feed_url, MAX_FETCH_PER_FEED)
        all_entries.extend(entries)

    seen = set()
    unique_entries = []
    for entry in all_entries:
        if entry.title not in seen:
            seen.add(entry.title)
            unique_entries.append(entry)

    relevant_candidates = [e for e in unique_entries if is_potentially_relevant(e, KEYWORDS)]

    processed = 0
    rss_items = []
    for entry in relevant_candidates:
        if processed >= MAX_TOTAL_PROCESSED:
            break

        print(f"Processing: {entry.title}")
        result = summarize_and_tag(entry)

        if result.get("relevant"):
            authors = [author.name for author in entry.authors] if hasattr(entry, "authors") else []
            rss_items.append({
                "title": entry.title,
                "url": entry.link,
                "summary": result["summary"],
                "tags": result["tags"],
                "authors": authors,
                "relevance": result.get("relevance", "?")
            })
            send_to_airtable(
                title=entry.title,
                url=entry.link,
                summary=result["summary"],
                tags=result["tags"],
                authors=", ".join(authors),
                relevance=result.get("relevance", "?")
            )
            processed += 1
        else:
            print("Not relevant to AI security.")

    if rss_items:
        write_rss_feed(rss_items, RSS_OUTPUT_PATH)
        print(f"\n✅ RSS feed written to: {RSS_OUTPUT_PATH}")
    else:
        print("\nNo relevant papers found for RSS today.")

if __name__ == "__main__":
    main()
