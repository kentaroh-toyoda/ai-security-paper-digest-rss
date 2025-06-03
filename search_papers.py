import os
import time
import json
import requests
import argparse
from datetime import datetime
from dotenv import load_dotenv

from utils.gpt import assess_relevance_and_tags, assess_paper_quality
from utils.baserow import (
    insert_to_baserow,
    paper_exists_in_baserow,
    ensure_baserow_fields_exist,
)

load_dotenv()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Search papers from OpenAlex')
parser.add_argument('--topic', type=str, default=os.getenv("SEARCH_QUERY", "LLM red teaming"),
                    help='Topic to search for')
parser.add_argument('--start-date', type=str, default=os.getenv("CUTOFF_DATE", "2022-01-01"),
                    help='Start date in YYYY-MM-DD format')
parser.add_argument('--max-pages', type=int, default=100,
                    help='Maximum number of pages to fetch')
args = parser.parse_args()

# Load environment variables
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASEROW_API_TOKEN = os.getenv("BASEROW_API_TOKEN")
BASEROW_TABLE_ID = os.getenv("BASEROW_TABLE_ID")

# Query parameters
SEARCH_QUERY = args.topic
CUTOFF_DATE = args.start_date
MAX_PAGES = args.max_pages

# OpenAlex base URL
BASE_URL = "https://api.openalex.org/works"

# Ensure Baserow columns exist
ensure_baserow_fields_exist(
    BASEROW_API_TOKEN,
    BASEROW_TABLE_ID,
    ["Clarity", "Novelty", "Significance", "Try-worthiness",
        "Justification", "Code repository"]
)


def reconstruct_abstract(inverted_index):
    position_word = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            position_word[pos] = word
    return " ".join(word for pos, word in sorted(position_word.items()))


# Time formatting
print(
    f"🔎 Searching OpenAlex for '{SEARCH_QUERY}' papers since {CUTOFF_DATE}...")

cursor = "*"
count = 0
page_count = 0

while cursor and page_count < MAX_PAGES:
    url = f"{BASE_URL}?search={SEARCH_QUERY}&filter=from_publication_date:{CUTOFF_DATE}&per-page=100&cursor={cursor}&mailto={OPENALEX_EMAIL}"
    response = requests.get(url)
    data = response.json()

    works = data.get("results", [])
    if not works:
        print("✅ No more results.")
        break

    for work in works:
        title = work.get("title", "")
        url = work.get("id", "")
        abstract = work.get("abstract_inverted_index", {})
        authors = ", ".join([a.get("author", {}).get("display_name", "")
                            for a in work.get("authorships", [])])
        date = work.get("publication_date",
                        datetime.now().strftime("%Y-%m-%d"))

        if not url or not title:
            continue

        abstract_text = reconstruct_abstract(abstract) if abstract else ""
        full_text = f"Title: {title}\nAbstract: {abstract_text}\nURL: {url}"

        result = assess_relevance_and_tags(full_text, OPENAI_API_KEY)
        if not result["relevant"]:
            print(f"🚫 Not relevant: {title}")
            continue

        print(f"✅ Relevant: {title}")

        row = {
            "Title": title,
            "URL": url,
            "Summary": "\n".join(result["summary"]),
            "Tags": ", ".join(result["tags"]),
            "Authors": authors,
            "Date": date,
            "Relevance": result.get("relevance_score", 3),
        }

        if "arxiv.org" in url:
            arxiv_id = url.split("/")[-1]
            html_url = f"https://arxiv.org/html/{arxiv_id}"
            try:
                html_response = requests.get(html_url)
                if html_response.status_code == 200:
                    html_text = html_response.text
                    quality = assess_paper_quality(
                        title, html_text, OPENAI_API_KEY)
                    row.update(quality)
                else:
                    print(f"⚠️ Failed to retrieve HTML for: {title}")
            except Exception as e:
                print(f"❌ Error fetching full text: {e}")

        if not paper_exists_in_baserow(url, BASEROW_API_TOKEN, BASEROW_TABLE_ID):
            insert_to_baserow(row, BASEROW_API_TOKEN, BASEROW_TABLE_ID)
        else:
            print(f"⏭️ Skipped (already exists in Baserow): {title}")

        count += 1
        time.sleep(1.5)  # Rate limiting

    cursor = data.get("meta", {}).get("next_cursor")
    page_count += 1

print(f"🎉 Finished. Total relevant papers: {count}")
