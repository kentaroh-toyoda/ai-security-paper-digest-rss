import os
import json
import time
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from utils.baserow import get_all_papers, insert_to_baserow, paper_exists_in_baserow, FIELD_TITLE, FIELD_URL, FIELD_ABSTRACT, FIELD_TAGS, FIELD_AUTHORS, FIELD_DATE, FIELD_RELEVANCE, FIELD_SUMMARY, FIELD_PAPER_TYPE, FIELD_CLARITY, FIELD_NOVELTY, FIELD_SIGNIFICANCE, FIELD_TRY_WORTHINESS, FIELD_JUSTIFICATION, FIELD_CODE_REPO
from utils.gpt import assess_relevance_and_tags, assess_paper_quality
from collections import defaultdict
import requests
from urllib.parse import quote
import openai

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
    required_fields = [FIELD_TITLE, FIELD_URL, FIELD_AUTHORS, FIELD_DATE]
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

    # Ensure summary and tags are at least empty lists if not present
    if not paper_data.get(FIELD_SUMMARY):
        paper_data[FIELD_SUMMARY] = []
    if not paper_data.get(FIELD_TAGS):
        paper_data[FIELD_TAGS] = []

    print(f"✅ Paper processed successfully")
    return paper_data


def generate_related_keywords(query: str, api_key: str) -> list:
    """Generate related keywords for the search query using GPT."""
    prompt = f"""Given the search topic "{query}", generate 4 closely related keywords or phrases that would help find relevant papers.
    The keywords should be specific and focused on the same domain.
    Return only the keywords as a comma-separated list, without any additional text.
    Example format: keyword1, keyword2, keyword3, keyword4"""

    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates relevant search keywords for academic papers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=100
        )

        # Extract and clean the keywords
        keywords_text = response.choices[0].message.content.strip()
        keywords = [k.strip() for k in keywords_text.split(',')]

        # Ensure we have exactly 4 keywords
        if len(keywords) > 4:
            keywords = keywords[:4]
        elif len(keywords) < 4:
            # If we got fewer than 4 keywords, duplicate the last one
            while len(keywords) < 4:
                keywords.append(keywords[-1])

        return keywords
    except Exception as e:
        print(f"Error generating related keywords: {e}")
        return []


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
    related_keywords = generate_related_keywords(query, OPENAI_API_KEY)
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
    print(f"Total unique papers found: {len(papers)}")
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
