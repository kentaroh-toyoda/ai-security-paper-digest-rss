#!/usr/bin/env python3

import os
import sys
import json
import requests
import argparse
import feedparser
from urllib.parse import urlparse
from dotenv import load_dotenv
from utils.qdrant import init_qdrant_client, ensure_collection_exists, paper_exists, insert_paper
from utils.llm import assess_relevance_and_tags, check_rate_limit_status, quick_assess_relevance

# Load environment variables
load_dotenv()

# Get API keys and configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
QDRANT_API_URL = os.getenv("QDRANT_API_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL")
OPENALEX_URL = "https://api.openalex.org/works"
AI_MODEL = os.getenv("AI_MODEL", "openai/gpt-4.1-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))

# Validate environment variables
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


def parse_url(url):
    """Parse the URL to determine the source and ID."""
    parsed_url = urlparse(url)
    
    # Check if it's an arXiv URL
    if "arxiv.org" in parsed_url.netloc:
        # Extract the arXiv ID from the URL
        # URL format: https://arxiv.org/abs/2506.07948
        paper_id = parsed_url.path.split("/")[-1]
        return "arxiv", paper_id
    
    # Check if it's an OpenAlex URL
    elif "openalex.org" in parsed_url.netloc:
        # Extract the OpenAlex ID from the URL
        # URL format: https://openalex.org/works/w4392222514
        paper_id = parsed_url.path.split("/")[-1]
        return "openalex", paper_id
    
    else:
        return None, None


def fetch_from_arxiv(arxiv_id):
    """Fetch paper information from arXiv."""
    print(f"🔍 Fetching paper from arXiv with ID: {arxiv_id}")
    
    # Use the arXiv API to fetch the paper
    arxiv_api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    response = requests.get(arxiv_api_url)
    
    if response.status_code != 200:
        print(f"❌ Error fetching from arXiv API: {response.status_code}")
        return None
    
    # Parse the response using feedparser
    feed = feedparser.parse(response.text)
    
    if not feed.entries:
        print(f"❌ No entries found for arXiv ID: {arxiv_id}")
        return None
    
    entry = feed.entries[0]
    
    # Extract paper information
    title = entry.title
    abstract = entry.summary
    authors = ", ".join([author.name for author in entry.authors])
    date = entry.published.split("T")[0]  # Get just the date part
    url = f"https://arxiv.org/abs/{arxiv_id}"
    
    print(f"📄 Found paper: {title}")
    print(f"👥 Authors: {authors}")
    print(f"📅 Date: {date}")
    
    # Create paper data dictionary
    paper_data = {
        "title": title,
        "abstract": abstract,
        "url": url,
        "date": date,
        "authors": authors,
        "source": "arxiv",
        "arxiv_id": arxiv_id,
        "tags": [],
        "summary": [],
        "relevance_score": 0,
        "paper_type": "Other",
        "modalities": [],
        "code_repository": ""
    }
    
    return paper_data


def fetch_from_openalex(openalex_id):
    """Fetch paper information from OpenAlex."""
    print(f"🔍 Fetching paper from OpenAlex with ID: {openalex_id}")
    
    # Use the OpenAlex API to fetch the paper
    openalex_api_url = f"{OPENALEX_URL}/{openalex_id}?mailto={OPENALEX_EMAIL}"
    response = requests.get(openalex_api_url)
    
    if response.status_code != 200:
        print(f"❌ Error fetching from OpenAlex API: {response.status_code}")
        return None
    
    paper = response.json()
    
    # Extract paper information
    title = paper.get("title", "Unknown")
    
    # Handle abstract which might be in inverted index format
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
    
    # Get publication date
    pub_date = paper.get("publication_date", "")
    if not pub_date:
        pub_date = paper.get("publication_year", "")
        if pub_date:
            # If we only have year, use January 1st of that year
            pub_date = f"{pub_date}-01-01"
    
    # Get OpenAlex ID and URL
    url = paper.get("id", "")  # OpenAlex ID is also the URL
    
    print(f"📄 Found paper: {title}")
    print(f"👥 Authors: {', '.join(authors)}")
    print(f"📅 Date: {pub_date}")
    
    # Create paper data dictionary
    paper_data = {
        "title": title,
        "abstract": abstract_text,
        "url": url,
        "date": pub_date,
        "authors": ", ".join(authors),
        "source": "openalex",
        "tags": [],
        "summary": [],
        "relevance_score": 0,
        "paper_type": "Other",
        "modalities": [],
        "code_repository": ""
    }
    
    return paper_data


def process_paper(paper_data):
    """Process a paper and prepare it for Qdrant."""
    title = paper_data.get("title", "Unknown")
    print(f"\n🔍 Processing paper: {title}")
    
    # Define models for different stages
    QUICK_ASSESSMENT_MODEL = "openai/gpt-4.1-nano"  # Cheaper model for initial filtering
    DETAILED_ASSESSMENT_MODEL = AI_MODEL  # More expensive model for detailed analysis
    
    try:
        # Use title and abstract for assessment
        assessment_text = f"Title: {title}"
        abstract = paper_data.get("abstract", "")
        if abstract:
            assessment_text += f"\n\nAbstract: {abstract}"
        
        # STAGE 1: Quick assessment with cheaper model
        print(f"🔍 Quick relevance assessment...")
        potentially_relevant, quick_tokens = quick_assess_relevance(
            assessment_text, 
            OPENROUTER_API_KEY, 
            temperature=TEMPERATURE, 
            model=QUICK_ASSESSMENT_MODEL
        )
        
        if not potentially_relevant:
            print(f"🚫 Not relevant (quick assessment): {title}")
            return paper_data  # Return with default values (not relevant)
            
        print(f"✓ Potentially relevant (quick assessment): {title}")
        
        # STAGE 2: Detailed assessment with more expensive model
        print(f"🔍 Detailed relevance assessment...")
        result, detailed_tokens = assess_relevance_and_tags(
            assessment_text,
            api_key=OPENROUTER_API_KEY,
            temperature=TEMPERATURE,
            model=DETAILED_ASSESSMENT_MODEL
        )
        
        # Print token usage
        print(f"📊 Token usage:")
        print(f"  Quick assessment: {quick_tokens} tokens")
        print(f"  Detailed assessment: {detailed_tokens} tokens")
        print(f"  Total: {quick_tokens + detailed_tokens} tokens")
        
        # Calculate cost savings
        quick_cost = estimate_cost(quick_tokens, QUICK_ASSESSMENT_MODEL)
        detailed_cost = estimate_cost(detailed_tokens, DETAILED_ASSESSMENT_MODEL)
        old_approach_cost = estimate_cost(quick_tokens + detailed_tokens, DETAILED_ASSESSMENT_MODEL)
        savings = old_approach_cost - (quick_cost + detailed_cost)
        
        print(f"💰 Estimated cost savings: ${savings:.6f}")
        
        if result.get("relevant", False):
            print(f"✅ Paper is relevant")
            paper_data["tags"] = result.get("tags", [])
            paper_data["relevance_score"] = result.get("relevance_score", 0)
            paper_data["paper_type"] = result.get("paper_type", "Research Paper")
            paper_data["summary"] = result.get("summary", [])
            paper_data["modalities"] = result.get("modalities", [])
            paper_data["code_repository"] = result.get("code_repository", "")
            
            print(f"📊 Relevance assessment:")
            print(f"  Score: {paper_data['relevance_score']}")
            print(f"  Tags: {', '.join(paper_data['tags'])}")
            print(f"  Summary: {paper_data['summary']}")
            print(f"  Modalities: {', '.join(paper_data['modalities'])}")
            if paper_data["code_repository"]:
                print(f"  Code Repository: {paper_data['code_repository']}")
        else:
            print(f"❌ Paper is not relevant (detailed assessment)")
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


def estimate_cost(tokens, model=None):
    """Estimate cost based on token usage and model."""
    if model is None:
        model = AI_MODEL
    
    # Pricing per 1K tokens (input/output combined for simplicity)
    # Based on OpenRouter pricing as of 2024
    pricing = {
        # OpenAI models
        "openai/gpt-4o": 0.005,  # $5.00 per 1M tokens
        "openai/gpt-4o-mini": 0.00015,  # $0.15 per 1M tokens
        "openai/gpt-4-turbo": 0.01,  # $10.00 per 1M tokens
        "openai/gpt-3.5-turbo": 0.0005,  # $0.50 per 1M tokens
        "openai/gpt-4.1": 0.01,  # $10.00 per 1M tokens
        "openai/gpt-4.1-mini": 0.00015,  # $0.15 per 1M tokens
        "openai/gpt-4.1-nano": 0.000075,  # $0.075 per 1M tokens
        
        # Anthropic models
        "anthropic/claude-3-5-sonnet": 0.003,  # $3.00 per 1M tokens
        "anthropic/claude-3-haiku": 0.00025,  # $0.25 per 1M tokens
        "anthropic/claude-3-sonnet": 0.015,  # $15.00 per 1M tokens
        "anthropic/claude-3-opus": 0.075,  # $75.00 per 1M tokens
        
        # Google models
        "google/gemini-pro": 0.0005,  # $0.50 per 1M tokens
        "google/gemini-flash": 0.000075,  # $0.075 per 1M tokens
        
        # Meta models
        "meta-llama/llama-3.1-8b-instruct": 0.0002,  # $0.20 per 1M tokens
        "meta-llama/llama-3.1-70b-instruct": 0.0008,  # $0.80 per 1M tokens
        
        # Moonshot models
        "moonshotai/kimi-dev-72b:free": 0.0,  # Free tier
        "moonshotai/kimi-dev-72b": 0.0006,  # $0.60 per 1M tokens
        
        # Mistral models
        "mistralai/mistral-7b-instruct": 0.00014,  # $0.14 per 1M tokens
        "mistralai/mixtral-8x7b-instruct": 0.00024,  # $0.24 per 1M tokens
        
        # Default fallback
        "default": 0.001  # $1.00 per 1M tokens
    }
    
    # Get cost per token (convert from per 1M to per token)
    cost_per_1k_tokens = pricing.get(model, pricing["default"])
    cost_per_token = cost_per_1k_tokens / 1000
    
    return round(tokens * cost_per_token, 6)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Add a paper to the Qdrant database from a URL'
    )
    parser.add_argument('url', type=str, help='URL of the paper to add (arXiv or OpenAlex)')
    
    args = parser.parse_args()
    
    # Check rate limit status before starting
    print("\n📊 Checking rate limit status...")
    check_rate_limit_status()
    
    # Parse the URL to determine the source and ID
    source, paper_id = parse_url(args.url)
    
    if not source or not paper_id:
        print(f"❌ Invalid URL: {args.url}")
        print("Please provide a valid arXiv or OpenAlex URL")
        sys.exit(1)
    
    # Check if the paper already exists in Qdrant
    if paper_exists(qdrant_client, args.url):
        print(f"⏭️ Paper already exists in the database: {args.url}")
        sys.exit(0)
    
    # Fetch paper information based on the source
    if source == "arxiv":
        paper_data = fetch_from_arxiv(paper_id)
    elif source == "openalex":
        paper_data = fetch_from_openalex(paper_id)
    else:
        print(f"❌ Unsupported source: {source}")
        sys.exit(1)
    
    if not paper_data:
        print(f"❌ Failed to fetch paper information")
        sys.exit(1)
    
    # Process the paper
    processed_paper = process_paper(paper_data)
    
    if not processed_paper:
        print(f"❌ Failed to process paper")
        sys.exit(1)
    
    # Only store relevant papers
    if processed_paper.get("relevance_score", 0) > 0:
        if insert_paper(qdrant_client, processed_paper):
            print(f"✅ Added relevant paper: {processed_paper['title']}")
        else:
            print(f"❌ Failed to add paper: {processed_paper['title']}")
    else:
        print(f"⏭️ Skipping irrelevant paper: {processed_paper['title']}")


if __name__ == "__main__":
    main()
