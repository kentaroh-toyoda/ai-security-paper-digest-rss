import os
import time
import json
import requests
import argparse
from datetime import datetime
from dotenv import load_dotenv

from utils.gpt import (
    assess_relevance_and_tags,
    assess_paper_quality,
    generate_search_keywords
)
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
parser.add_argument('--max-pages', type=int, default=None,
                    help='Maximum number of pages to fetch (default: fetch all available results)')
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


def get_total_pages(url: str) -> int:
    """Get total number of pages from OpenAlex API."""
    try:
        response = requests.get(url)
        data = response.json()
        total_results = data.get("meta", {}).get("count", 0)
        results_per_page = data.get("meta", {}).get("per_page", 100)
        total_pages = (total_results + results_per_page -
                       1) // results_per_page
        print(f"📊 Found {total_results} total results ({total_pages} pages)")
        return total_pages
    except Exception as e:
        print(f"❌ Error getting total pages: {e}")
        return 0


# Time formatting
print(f"🔍 Generating search keywords for '{SEARCH_QUERY}'...")
optimized_query = generate_search_keywords(SEARCH_QUERY, OPENAI_API_KEY)
print(f"🔎 Searching OpenAlex with query: '{optimized_query}'")

# Construct initial URL to get total count
initial_url = f"{BASE_URL}?search={optimized_query}&filter=from_publication_date:{CUTOFF_DATE}&per-page=100&mailto={OPENALEX_EMAIL}"

# Get total number of pages
total_pages = get_total_pages(initial_url)
if total_pages == 0:
    print("❌ No results found or error occurred")
    exit(1)

# If max_pages is not specified, use total_pages
if MAX_PAGES is None:
    MAX_PAGES = total_pages
else:
    MAX_PAGES = min(MAX_PAGES, total_pages)

print(f"📚 Will process {MAX_PAGES} pages of results")

cursor = "*"
count = 0
page_count = 0

while cursor and page_count < MAX_PAGES:
    url = f"{BASE_URL}?search={optimized_query}&filter=from_publication_date:{CUTOFF_DATE}&per-page=100&cursor={cursor}&mailto={OPENALEX_EMAIL}"
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

        # Get additional metadata from OpenAlex
        cited_by_count = work.get("cited_by_count", 0)
        publication_type = work.get("type", "")
        source = work.get("primary_location", {}).get(
            "source", {}).get("display_name", "")
        code_url = work.get("open_access", {}).get("oa_url", "")

        if not url or not title:
            continue

        # Check if paper already exists in Baserow before processing
        if paper_exists_in_baserow(url, BASEROW_API_TOKEN, BASEROW_TABLE_ID):
            print(f"⏭️ Skipped (already exists in Baserow): {title}")
            continue

        abstract_text = reconstruct_abstract(abstract) if abstract else ""
        full_text = f"Title: {title}\nAbstract: {abstract_text}\nURL: {url}"

        result = assess_relevance_and_tags(full_text, OPENAI_API_KEY)
        if not result["relevant"]:
            print(f"🚫 Not relevant: {title}")
            continue

        print(f"✅ Relevant: {title}")

        # Prepare metadata for quality assessment
        metadata = {
            "title": title,
            "abstract": abstract_text,
            "cited_by_count": cited_by_count,
            "publication_type": publication_type,
            "source": source,
            "code_url": code_url,
            "date": date
        }

        # Assess paper quality using metadata
        quality = assess_paper_quality(metadata, OPENAI_API_KEY)

        row = {
            "Title": title,
            "URL": url,
            "Summary": "\n".join(result["summary"]),
            "Tags": ", ".join(result["tags"]),
            "Authors": authors,
            "Date": date,
            "Relevance": result.get("relevance_score", 3),
            "Paper Type": result.get("paper_type", "Other"),
            **quality  # Include quality assessment results
        }

        insert_to_baserow(row, BASEROW_API_TOKEN, BASEROW_TABLE_ID)
        count += 1
        time.sleep(1.5)  # Rate limiting

    cursor = data.get("meta", {}).get("next_cursor")
    page_count += 1
    print(f"📄 Processed page {page_count}/{MAX_PAGES}")

print(f"🎉 Finished. Total relevant papers: {count}")
