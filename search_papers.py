import os
import json
import time
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from utils.baserow import get_all_papers, insert_to_baserow, paper_exists_in_baserow
from utils.gpt import assess_relevance_and_tags, assess_paper_quality
from collections import defaultdict
import requests
from urllib.parse import quote

# Load environment variables
load_dotenv()

# Get API keys and configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASEROW_API_TOKEN = os.getenv("BASEROW_API_TOKEN")
BASEROW_TABLE_ID = os.getenv("BASEROW_TABLE_ID")
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL")
OPENALEX_URL = "https://api.openalex.org/works"

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
if not BASEROW_API_TOKEN:
    raise ValueError("BASEROW_API_TOKEN environment variable is not set")
if not BASEROW_TABLE_ID:
    raise ValueError("BASEROW_TABLE_ID environment variable is not set")
if not OPENALEX_EMAIL:
    raise ValueError("OPENALEX_EMAIL environment variable is not set")

# Map field IDs to logical names
FIELD_TITLE = "field_4496823"
FIELD_URL = "field_4496824"
FIELD_ABSTRACT = "field_4496825"
FIELD_TAGS = "field_4496826"
FIELD_AUTHORS = "field_4496827"
FIELD_DATE = "field_4496828"
FIELD_RELEVANCE = "field_4496829"
FIELD_SUMMARY = "field_4496830"
FIELD_PAPER_TYPE = "field_4496831"
FIELD_CLARITY = "field_4496832"
FIELD_NOVELTY = "field_4496833"
FIELD_SIGNIFICANCE = "field_4496834"
FIELD_TRY_WORTHINESS = "field_4496835"
FIELD_JUSTIFICATION = "field_4496836"
FIELD_CODE_REPO = "field_4496837"


def fetch_from_openalex(query: str, start_date: str = None, max_pages: int = 10) -> list:
    """Fetch papers from OpenAlex based on search query."""
    all_results = []
    page = 1
    per_page = 100  # Maximum page size

    # URL encode the query
    encoded_query = quote(query)

    # Construct the search query with better filtering
    # Use title_and_abstract.search for better matching and add type filter for journal articles
    search_query = f"filter=title_and_abstract.search:{encoded_query}&filter=type:journal-article"
    if start_date:
        search_query += f"&filter=from_publication_date:{start_date}"

    print(f"\n🔍 Searching OpenAlex for: {query}")
    if start_date:
        print(f"From date: {start_date}")

    # First, get the total number of results
    try:
        url = f"{OPENALEX_URL}?{search_query}&per-page=1&page=1&mailto={OPENALEX_EMAIL}"
        resp = requests.get(url)
        if resp.status_code == 200:
            data = resp.json()
            meta = data.get("meta", {})
            total_results = meta.get("count", 0)
            total_pages = (total_results + per_page -
                           1) // per_page  # Ceiling division
            print(
                f"📊 Found {total_results} total results ({total_pages} pages)")
        else:
            print(f"❌ Error getting total results: {resp.status_code}")
            print(f"Response: {resp.text}")
            return []
    except Exception as e:
        print(f"❌ Error getting total results: {e}")
        return []

    # Now fetch all pages
    while page <= total_pages:
        url = f"{OPENALEX_URL}?{search_query}&per-page={per_page}&page={page}&mailto={OPENALEX_EMAIL}"
        print(f"📄 Fetching page {page} of {total_pages}...")
        print(f"URL: {url}")  # Debug: print the URL

        try:
            resp = requests.get(url)
            if resp.status_code != 200:
                print(f"❌ Error from OpenAlex API: {resp.status_code}")
                print(f"Response: {resp.text}")
                break

            data = resp.json()
            results = data.get("results", [])

            if not results:  # No more results
                break

            # Filter out results without required fields
            valid_results = []
            for paper in results:
                if (paper.get("title") and
                    paper.get("id") and
                    paper.get("authorships") and
                        paper.get("publication_date")):
                    valid_results.append(paper)
                else:
                    print(
                        f"Skipping paper due to missing required fields: {paper.get('title', 'Unknown')}")

            all_results.extend(valid_results)
            print(f"📚 Found {len(valid_results)} valid papers on page {page}")

            page += 1
            time.sleep(1)  # Be nice to the API
        except Exception as e:
            print(f"❌ Error fetching from OpenAlex: {e}")
            break

    print(f"📊 Found {len(all_results)} total valid papers")
    return all_results


def process_paper(paper: dict) -> dict:
    """Process a paper and prepare it for Baserow."""
    title = paper.get("title", "Unknown")
    print(f"\n🔍 Processing paper: {title}")

    # Extract basic paper information
    abstract = paper.get("abstract_inverted_index", {})
    if abstract:
        # Convert inverted index to text
        abstract_text = " ".join(abstract.keys())
    else:
        abstract_text = paper.get("abstract", "")

    # Get authors
    authors = []
    for authorship in paper.get("authorships", []):
        author = authorship.get("author", {})
        if author:
            authors.append(author.get("display_name", ""))

    # Get publication date with better fallback handling
    pub_date = paper.get("publication_date", "")
    if not pub_date:
        pub_date = paper.get("publication_year", "")
        if pub_date:
            # If we only have year, use January 1st of that year
            pub_date = f"{pub_date}-01-01"

    # Get OpenAlex ID and URL
    openalex_id = paper.get("id", "").split("/")[-1]
    url = paper.get("id", "")  # OpenAlex ID is also the URL

    print(f"📄 Basic info:")
    print(f"  Title: {title}")
    print(f"  URL: {url}")
    print(f"  Date: {pub_date}")
    print(f"  Authors: {', '.join(authors)}")

    # Initialize paper data with all required fields using Baserow field IDs
    # Set default values for rating fields to 3 (neutral)
    paper_data = {
        FIELD_TITLE: title,
        FIELD_ABSTRACT: abstract_text,
        FIELD_URL: url,
        FIELD_DATE: pub_date,
        FIELD_AUTHORS: ", ".join(authors),
        FIELD_TAGS: [],
        FIELD_SUMMARY: [],
        FIELD_RELEVANCE: 0,
        FIELD_PAPER_TYPE: "Other",
        FIELD_CLARITY: 3,  # Default neutral rating
        FIELD_NOVELTY: 3,  # Default neutral rating
        FIELD_SIGNIFICANCE: 3,  # Default neutral rating
        FIELD_TRY_WORTHINESS: 3,  # Default neutral rating
        FIELD_JUSTIFICATION: "Quality assessment not available",
        FIELD_CODE_REPO: ""
    }

    # Process paper even if we don't have an abstract
    try:
        print(f"🤖 Assessing relevance and tags...")
        # Use title for assessment if no abstract is available
        assessment_text = f"Title: {title}"
        if abstract_text:
            assessment_text += f"\n\nAbstract: {abstract_text}"

        # Assess relevance and get tags
        result, _ = assess_relevance_and_tags(
            text=assessment_text,
            api_key=OPENAI_API_KEY
        )

        if result.get("relevant", False):
            print(f"✅ Paper is relevant")
            paper_data[FIELD_TAGS] = result.get("tags", [])
            paper_data[FIELD_RELEVANCE] = result.get("relevance_score", 0)
            paper_data[FIELD_PAPER_TYPE] = result.get(
                "paper_type", "Research Paper")
            paper_data[FIELD_SUMMARY] = result.get("summary", [])

            print(f"📊 Relevance assessment:")
            print(f"  Score: {paper_data[FIELD_RELEVANCE]}")
            print(f"  Tags: {', '.join(paper_data[FIELD_TAGS])}")
            print(f"  Summary: {paper_data[FIELD_SUMMARY]}")

            print(f"🔍 Assessing paper quality...")
            try:
                # Assess paper quality
                quality = assess_paper_quality(paper_data, OPENAI_API_KEY)
                if quality:
                    print(f"📈 Quality assessment results:")
                    print(json.dumps(quality, indent=2))
                    # Map quality assessment fields to Baserow field IDs
                    if "clarity" in quality:
                        paper_data[FIELD_CLARITY] = quality["clarity"]
                    if "novelty" in quality:
                        paper_data[FIELD_NOVELTY] = quality["novelty"]
                    if "significance" in quality:
                        paper_data[FIELD_SIGNIFICANCE] = quality["significance"]
                    if "try_worthiness" in quality:
                        paper_data[FIELD_TRY_WORTHINESS] = quality["try_worthiness"]
                    if "justification" in quality:
                        paper_data[FIELD_JUSTIFICATION] = quality["justification"]
                    if "code_url" in quality:
                        paper_data[FIELD_CODE_REPO] = quality["code_url"]
            except Exception as e:
                print(f"⚠️ Warning: Error during quality assessment: {e}")
                # Keep default values for rating fields
        else:
            print(f"❌ Paper is not relevant")
    except Exception as e:
        print(f"❌ Error processing paper '{title}': {e}")
        return None  # Return None to indicate processing failed

    # Validate required fields
    required_fields = [FIELD_TITLE, FIELD_URL,
                       FIELD_SUMMARY, FIELD_TAGS, FIELD_AUTHORS, FIELD_DATE]
    print(f"\n🔍 Validating required fields:")
    for field in required_fields:
        value = paper_data.get(field)
        print(f"  {field}: {value}")

    # Check for missing or empty fields
    missing_fields = []
    for field in required_fields:
        value = paper_data.get(field)
        if not value or (isinstance(value, list) and len(value) == 0):
            missing_fields.append(field)

    if missing_fields:
        print(f"❌ Validation errors for '{title}':")
        for field in missing_fields:
            print(f"  - Missing or empty required field: {field}")
        return None  # Return None to indicate validation failed

    print(f"✅ Paper processed successfully")
    return paper_data


def main():
    # Get search query from user with default value
    default_query = "LLM red teaming"
    query = input(
        f"Enter your search topic (default: '{default_query}'): ").strip()
    if not query:
        query = default_query
        print(f"Using default topic: {query}")

    # Get date range for search
    default_start_date = "2022-01-01"
    start_date = input(
        f"\nEnter start date (YYYY-MM-DD) or press Enter for {default_start_date}: ").strip()
    if not start_date:
        start_date = default_start_date
        print(f"Using default start date: {start_date}")

    # Fetch papers from OpenAlex
    papers = fetch_from_openalex(query, start_date)
    if not papers:
        print("No papers found in OpenAlex")
        return

    print(f"\nProcessing {len(papers)} papers...")
    new_papers = []
    existing_papers = []

    for paper in papers:
        url = paper.get("id", "")
        if not url:
            continue

        if paper_exists_in_baserow(url, BASEROW_API_TOKEN, BASEROW_TABLE_ID):
            existing_papers.append(paper)
            continue

        # Process and store new paper
        paper_data = process_paper(paper)
        if paper_data and insert_to_baserow(paper_data, BASEROW_API_TOKEN, BASEROW_TABLE_ID):
            new_papers.append(paper_data)
            print(f"✅ Added new paper: {paper_data[FIELD_TITLE]}")
        else:
            print(f"❌ Failed to add paper: {paper.get('title', 'Unknown')}")

    print(f"\n📊 Summary:")
    print(f"Total papers found: {len(papers)}")
    print(f"New papers added: {len(new_papers)}")
    print(f"Existing papers: {len(existing_papers)}")

    # Show new papers
    if new_papers:
        print("\nNew papers added:")
        for paper in new_papers:
            print(f"\nTitle: {paper[FIELD_TITLE]}")
            print(f"URL: {paper[FIELD_URL]}")
            print(f"Date: {paper[FIELD_DATE]}")
            print(f"Relevance score: {paper[FIELD_RELEVANCE]}")
            print(f"Tags: {', '.join(paper[FIELD_TAGS])}")
            print("-" * 80)


if __name__ == "__main__":
    main()
