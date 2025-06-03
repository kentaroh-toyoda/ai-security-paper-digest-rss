# update_rss.py

import os
import time
import feedparser
import requests
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from feedgen.feed import FeedGenerator
from utils.gpt import assess_relevance_and_tags
from utils.baserow import insert_to_baserow, paper_exists_in_baserow

load_dotenv()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASEROW_API_TOKEN = os.getenv("BASEROW_API_TOKEN")
BASEROW_TABLE_ID = os.getenv("BASEROW_TABLE_ID")
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL")
RSS_FEED_URL = os.getenv("RSS_FEED_URL")

# Constants
OPENALEX_URL = "https://api.openalex.org/works"
ARXIV_FEEDS = [
    "https://export.arxiv.org/rss/cs.AI",
    "https://export.arxiv.org/rss/cs.LG",
    "https://export.arxiv.org/rss/cs.CL",
    "https://export.arxiv.org/rss/cs.CV",
]

# Token usage tracking
total_tokens = 0

def fetch_openalex_24h():
    since = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
    url = f"{OPENALEX_URL}?filter=from_publication_date:{since}&per-page=100&mailto={OPENALEX_EMAIL}"
    resp = requests.get(url)
    return resp.json().get("results", [])

def fetch_arxiv():
    entries = []
    for feed_url in ARXIV_FEEDS:
        parsed = feedparser.parse(feed_url)
        entries.extend(parsed.entries)
    return entries

def build_rss_feed(relevant_papers):
    fg = FeedGenerator()
    fg.title("AI Security Paper Digest")
    fg.link(href=RSS_FEED_URL)
    fg.description("Curated papers on AI security from OpenAlex and ArXiv")

    for paper in relevant_papers:
        fe = fg.add_entry()
        fe.title(paper["Title"])
        fe.link(href=paper["URL"])
        fe.description("\n".join(paper["Summary"]))
        fe.author({"name": paper["Authors"]})
        fe.pubDate(datetime.fromisoformat(paper["Date"]).astimezone(timezone.utc))
        fe.guid(paper["URL"])

    fg.rss_file("rss.xml")

def process_papers(raw_papers, source):
    global total_tokens
    relevant = []

    for paper in raw_papers:
        # Extract info based on source
        if source == "openalex":
            title = paper.get("title", "")
            url = paper.get("id", "")
            abstract = paper.get("abstract", "") or ""
            authors = ", ".join([a.get("author", {}).get("display_name", "") for a in paper.get("authorships", [])])
            date = paper.get("publication_date", datetime.now(timezone.utc).date().isoformat())
        else:
            title = paper.title
            url = paper.link
            abstract = paper.summary if hasattr(paper, 'summary') else ""
            authors = paper.author if hasattr(paper, 'author') else "Unknown"
            date = datetime.now(timezone.utc).date().isoformat()

        if not title or not url:
            continue

        if paper_exists_in_baserow(url, BASEROW_API_TOKEN, BASEROW_TABLE_ID):
            print(f"⏭️ Already exists: {title}")
            continue

        fulltext = f"Title: {title}\nAbstract: {abstract}\nURL: {url}"
        result, token_count = assess_relevance_and_tags(fulltext, OPENAI_API_KEY, temperature=0.1, model="gpt-4.1")
        total_tokens += token_count

        if not result["relevant"]:
            print(f"🚫 Not relevant: {title}")
            continue

        print(f"✅ Relevant: {title}")
        row = {
            "Title": title,
            "URL": url,
            "Summary": result["summary"],
            "Tags": ", ".join(result["tags"]),
            "Authors": authors,
            "Date": date,
            "Relevance Score": result["relevance_score"]
        }

        insert_to_baserow(row, BASEROW_API_TOKEN, BASEROW_TABLE_ID)
        relevant.append(row)
        time.sleep(1.5)  # prevent rate limiting

    return relevant

def estimate_cost(tokens):
    cost_per_token = 0.01 / 1000  # GPT-4.1 input pricing ($0.01/1K tokens)
    return round(tokens * cost_per_token, 4)

def main():
    print("🔄 Fetching OpenAlex papers (past 24h)...")
    openalex_papers = fetch_openalex_24h()

    print("🔄 Fetching ArXiv RSS feeds...")
    arxiv_papers = fetch_arxiv()

    print("🧠 Filtering OpenAlex papers...")
    openalex_results = process_papers(openalex_papers, source="openalex")

    print("🧠 Filtering ArXiv papers...")
    arxiv_results = process_papers(arxiv_papers, source="arxiv")

    all_results = openalex_results + arxiv_results

    print(f"📝 Generating RSS feed with {len(all_results)} papers...")
    build_rss_feed(all_results)

    cost = estimate_cost(total_tokens)
    print(f"✅ Done. Total tokens used: {total_tokens}, Estimated cost: ${cost:.4f}")

if __name__ == "__main__":
    main()
