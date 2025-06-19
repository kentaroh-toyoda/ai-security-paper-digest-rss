import os
import json
import time
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from utils.qdrant import init_qdrant_client, ensure_collection_exists, paper_exists, insert_paper, get_all_papers
from utils.gpt import assess_relevance_and_tags, assess_paper_quality
from collections import defaultdict
import requests
from urllib.parse import quote
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

# Load environment variables
load_dotenv()

# Get API keys and configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
QDRANT_API_URL = os.getenv("QDRANT_API_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL")
OPENALEX_URL = "https://api.openalex.org/works"
AI_MODEL = os.getenv("AI_MODEL", "openai/gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set")
if not QDRANT_API_URL:
    raise ValueError("QDRANT_API_URL environment variable is not set")
if not QDRANT_API_KEY:
    raise ValueError("QDRANT_API_KEY environment variable is not set")
if not OPENALEX_EMAIL:
    raise ValueError("OPENALEX_EMAIL environment variable is not set")

# Initialize Qdrant client
qdrant_client = init_qdrant_client()
ensure_collection_exists(qdrant_client)


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
    """Process a paper and prepare it for Qdrant."""
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

    # Initialize paper data
    paper_data = {
        "title": title,
        "abstract": abstract_text,
        "url": url,
        "date": pub_date,
        "authors": ", ".join(authors),
        "tags": [],
        "summary": [],
        "relevance_score": 0,
        "paper_type": "Other",
        "modalities": [],  # Add modalities field
        "code_repository": ""  # Add code repository field
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
            api_key=OPENROUTER_API_KEY,
            temperature=TEMPERATURE,
            model=AI_MODEL
        )

        if result.get("relevant", False):
            print(f"✅ Paper is relevant")
            paper_data["tags"] = result.get("tags", [])
            paper_data["relevance_score"] = result.get("relevance_score", 0)
            paper_data["paper_type"] = result.get(
                "paper_type", "Research Paper")
            paper_data["summary"] = result.get("summary", [])
            paper_data["modalities"] = result.get(
                "modalities", [])  # Store modalities
            paper_data["code_repository"] = result.get(
                "code_repository", "")  # Store code repository URL

            print(f"📊 Relevance assessment:")
            print(f"  Score: {paper_data['relevance_score']}")
            print(f"  Tags: {', '.join(paper_data['tags'])}")
            print(f"  Summary: {paper_data['summary']}")
            print(f"  Modalities: {', '.join(paper_data['modalities'])}")
            if paper_data["code_repository"]:
                print(f"  Code Repository: {paper_data['code_repository']}")
        else:
            print(f"❌ Paper is not relevant")
    except Exception as e:
        print(f"❌ Error processing paper '{title}': {e}")
        return None  # Return None to indicate processing failed

    # Validate required fields
    required_fields = ["title", "url", "authors", "date"]
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

    # Ensure summary, tags, and modalities are at least empty lists if not present
    if not paper_data.get("summary"):
        paper_data["summary"] = []
    if not paper_data.get("tags"):
        paper_data["tags"] = []
    if not paper_data.get("modalities"):
        paper_data["modalities"] = []
    if not paper_data.get("code_repository"):
        paper_data["code_repository"] = ""

    print(f"✅ Paper processed successfully")
    return paper_data


def generate_related_keywords(query: str, api_key: str) -> list:
    """Generate related keywords for a search query."""
    from utils.gpt import generate_search_keywords

    # Generate optimized search keywords
    optimized_query = generate_search_keywords(query, api_key)

    # Split the query into individual terms
    # Remove common operators and clean up
    terms = optimized_query.replace(
        "AND", "").replace("OR", "").replace("NOT", "")
    terms = terms.replace("(", "").replace(")", "").replace('"', "")

    # Split by spaces and filter out empty strings
    keywords = [term.strip() for term in terms.split() if term.strip()]

    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for keyword in keywords:
        if keyword.lower() not in seen:
            seen.add(keyword.lower())
            unique_keywords.append(keyword)

    return unique_keywords[:10]  # Limit to 10 keywords


def main():
    # Get search query from user with default value
    default_query = "LLM red teaming"
    query = input(
        f"Enter your search topic (default: '{default_query}'): ").strip()
    if not query:
        query = default_query
        print(f"Using default topic: {query}")

    # Generate related keywords
    print("\n🔍 Generating related keywords...")
    related_keywords = generate_related_keywords(query, OPENROUTER_API_KEY)
    if related_keywords:
        print(f"Related keywords: {', '.join(related_keywords)}")
    else:
        print("Failed to generate related keywords, using only the main query")
        related_keywords = []

    # Get date range for search
    default_start_date = "2022-01-01"
    start_date = input(
        f"\nEnter start date (YYYY-MM-DD) or press Enter for {default_start_date}: ").strip()
    if not start_date:
        start_date = default_start_date
        print(f"Using default start date: {start_date}")

    # Fetch papers from OpenAlex for main query and related keywords
    all_papers = []
    search_queries = [query] + related_keywords

    for search_query in search_queries:
        print(f"\n🔍 Searching for: {search_query}")
        papers = fetch_from_openalex(search_query, start_date)
        if papers:
            all_papers.extend(papers)
            print(f"Found {len(papers)} papers for '{search_query}'")

    # Remove duplicates based on OpenAlex ID
    unique_papers = {paper.get("id"): paper for paper in all_papers}.values()
    papers = list(unique_papers)

    if not papers:
        print("No papers found in OpenAlex")
        return

    print(f"\nProcessing {len(papers)} unique papers...")
    new_papers = []
    existing_papers = []
    irrelevant_papers = []

    for paper in papers:
        url = paper.get("id", "")
        if not url:
            continue

        if paper_exists(qdrant_client, url):
            existing_papers.append(paper)
            continue

        # Process paper
        paper_data = process_paper(paper)

        # Only store relevant papers
        if paper_data and paper_data.get("relevance_score", 0) > 0:
            if insert_paper(qdrant_client, paper_data):
                new_papers.append(paper_data)
                print(f"✅ Added relevant paper: {paper_data['title']}")
            else:
                print(
                    f"❌ Failed to add paper: {paper.get('title', 'Unknown')}")
        else:
            irrelevant_papers.append(paper)
            print(
                f"⏭️ Skipping irrelevant paper: {paper.get('title', 'Unknown')}")

    print(f"\n📊 Summary:")
    print(f"Total unique papers found: {len(papers)}")
    print(f"New relevant papers added: {len(new_papers)}")
    print(f"Existing papers: {len(existing_papers)}")
    print(f"Irrelevant papers skipped: {len(irrelevant_papers)}")

    # Show new papers
    if new_papers:
        print("\nNew papers added:")
        for paper in new_papers:
            print(f"\nTitle: {paper['title']}")
            print(f"URL: {paper['url']}")
            print(f"Date: {paper['date']}")
            print(f"Relevance score: {paper['relevance_score']}")
            print(f"Tags: {', '.join(paper['tags'])}")
            if paper.get("code_repository"):
                print(f"Code Repository: {paper['code_repository']}")
            print("-" * 80)


if __name__ == "__main__":
    main()
