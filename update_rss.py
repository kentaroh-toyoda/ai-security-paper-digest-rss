# update_rss.py

import os
import time
import feedparser
import requests
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from feedgen.feed import FeedGenerator
from utils.llm import assess_relevance_and_tags, check_rate_limit_status, get_rate_limiter, update_daily_limit_for_paid_user, quick_assess_relevance
from utils.qdrant import init_qdrant_client, ensure_collection_exists, paper_exists, insert_paper

load_dotenv()

# Environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
QDRANT_API_URL = os.getenv("QDRANT_API_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
RSS_FEED_URL = os.getenv("RSS_FEED_URL")
AI_MODEL = os.getenv("AI_MODEL", "openai/gpt-4.1-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))

# Constants
ARXIV_FEEDS = [
    "https://export.arxiv.org/rss/cs.AI",
    "https://export.arxiv.org/rss/cs.LG",
    "https://export.arxiv.org/rss/cs.CL",
    "https://export.arxiv.org/rss/cs.CV",
]

# Token usage tracking
total_tokens = 0

# Initialize Qdrant client
qdrant_client = init_qdrant_client()
ensure_collection_exists(qdrant_client)


def fetch_arxiv():
    entries = []
    cutoff_time = datetime.now(timezone.utc) - timedelta(days=1)
    print(f"\n🔍 Fetching ArXiv papers since: {cutoff_time.isoformat()}")

    for feed_url in ARXIV_FEEDS:
        parsed = feedparser.parse(feed_url)
        for entry in parsed.entries:
            # Parse the published date
            published = datetime(
                *entry.published_parsed[:6], tzinfo=timezone.utc)
            if published >= cutoff_time:
                entries.append(entry)
            else:
                # Since entries are sorted by date, we can stop once we find an older entry
                break

    return entries


def build_rss_feed(relevant_papers):
    fg = FeedGenerator()
    fg.title("AI Security Paper Digest")
    fg.link(href=RSS_FEED_URL)
    fg.description("Curated papers on AI security from OpenAlex and ArXiv")

    for paper in relevant_papers:
        fe = fg.add_entry()
        fe.title(paper["title"])
        fe.link(href=paper["url"])

        # Build a more readable description
        description = []

        # Summary
        if "summary" in paper:
            description.append("<h3>Summary</h3>")
            description.append("<ul>")
            for point in paper["summary"]:
                description.append(f"<li>{point}</li>")
            description.append("</ul>")

        # Paper Type
        if "paper_type" in paper:
            description.append("<h3>Paper Type</h3>")
            description.append("<ul>")
            description.append(f"<li>{paper['paper_type']}</li>")
            description.append("</ul>")

        # Additional Information
        description.append("<h3>Additional Information</h3>")
        description.append("<ul>")
        if "authors" in paper:
            description.append(f"<li>Authors: {paper['authors']}</li>")
        if "tags" in paper:
            description.append(f"<li>Tags: {paper['tags']}</li>")
        if "relevance" in paper:
            description.append(
                f"<li>Relevance Score: {paper['relevance']}/5</li>")
        if "code_repository" in paper and paper["code_repository"]:
            description.append(
                f"<li>Code Repository: <a href='{paper['code_repository']}'>{paper['code_repository']}</a></li>")
        description.append("</ul>")

        fe.description("".join(description))
        fe.author({"name": paper["authors"]})

        # Handle date conversion more robustly
        try:
            if isinstance(paper["date"], str):
                # Ensure the date string is in ISO format
                if "T" not in paper["date"]:  # If it's just a date without time
                    paper["date"] = f"{paper['date']}T00:00:00+00:00"
                pub_date = datetime.fromisoformat(
                    paper["date"].replace("Z", "+00:00"))
            else:
                # If date is already a datetime object, use it directly
                pub_date = paper["date"]
            fe.pubDate(pub_date.astimezone(timezone.utc))
        except (ValueError, TypeError) as e:
            print(
                f"Warning: Could not parse date for paper {paper['title']}: {e}")
            # Use current time as fallback
            fe.pubDate(datetime.now(timezone.utc))

        fe.guid(paper["url"])

    fg.rss_file("rss.xml")


def process_paper(paper: dict) -> dict:
    """Process a paper and prepare it for storage."""
    paper_data = {
        "title": paper.get("title", ""),
        "abstract": paper.get("abstract", ""),
        "url": paper.get("url", ""),
        "date": paper.get("date", ""),
        "authors": paper.get("authors", ""),
        "source": paper.get("source", ""),
        "arxiv_id": paper.get("arxiv_id", ""),
        "cited_by_count": paper.get("cited_by_count", 0),
        "publication_type": paper.get("publication_type", ""),
        "code_repository": paper.get("code_repository", ""),
        "is_relevant": False,
        "tags": [],
        "relevance_score": 0,
        "relevance_reason": "",
        "paper_type": "Other",
        "modalities": []
    }

    # Two-stage assessment process is now handled in process_papers
    # This function now assumes the paper has already passed the quick assessment
    
    # Get the text to assess
    text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
    
    # Use the result from the detailed assessment that was already done in process_papers
    # This avoids duplicate API calls
    result = paper.get("_assessment_result", None)
    
    # If we don't have a result (e.g., when called directly), perform the assessment
    if result is None:
        result, _ = assess_relevance_and_tags(
            text=text,
            api_key=OPENROUTER_API_KEY,
            temperature=TEMPERATURE,
            model=AI_MODEL
        )

    if result.get("relevant", False):
        paper_data["is_relevant"] = True
        paper_data["tags"] = result.get("tags", [])
        paper_data["relevance_score"] = result.get("relevance_score", 0)
        paper_data["relevance_reason"] = result.get("reason", "")
        paper_data["paper_type"] = result.get("paper_type", "Research Paper")
        paper_data["modalities"] = result.get("modalities", [])
        paper_data["summary"] = result.get("summary", [])

    return paper_data


def process_papers(raw_papers):
    global total_tokens
    relevant = []
    current_date = datetime.now(timezone.utc).date()
    
    # Track token usage for different models
    quick_assessment_tokens = 0
    detailed_assessment_tokens = 0
    
    # Define models for different stages
    QUICK_ASSESSMENT_MODEL = "mistralai/mistral-7b-instruct"  # Cheaper model for initial filtering
    DETAILED_ASSESSMENT_MODEL = AI_MODEL  # More expensive model for detailed analysis

    # Check rate limit status before starting
    print("\n📊 Checking rate limit status...")
    check_rate_limit_status()

    for paper in raw_papers:
        title = paper.title if hasattr(paper, 'title') else ""
        url = paper.link if hasattr(paper, 'link') else ""
        abstract = paper.summary if hasattr(paper, 'summary') else ""
        authors = [author.strip() for author in paper.author.split(
            ",")] if hasattr(paper, 'author') else ["Unknown"]
        date = datetime.now(timezone.utc).date().isoformat()

        if not title or not url:
            continue

        print(f"\n📅 Processing paper published on: {date}")
        print(f"📄 Title: {title}")

        if paper_exists(qdrant_client, url):
            print(f"⏭️ Already exists: {title}")
            continue

        fulltext = f"Title: {title}\nAbstract: {abstract}\nURL: {url}"
        
        # STAGE 1: Quick assessment with cheaper model
        print(f"🔍 Quick relevance assessment...")
        potentially_relevant, quick_tokens = quick_assess_relevance(
            fulltext, OPENROUTER_API_KEY, temperature=TEMPERATURE, model=QUICK_ASSESSMENT_MODEL)
        quick_assessment_tokens += quick_tokens
        
        if not potentially_relevant:
            print(f"🚫 Not relevant (quick assessment): {title}")
            continue
            
        print(f"✓ Potentially relevant (quick assessment): {title}")
        
        # STAGE 2: Detailed assessment with more expensive model
        print(f"🔍 Detailed relevance assessment...")
        result, detailed_tokens = assess_relevance_and_tags(
            fulltext, OPENROUTER_API_KEY, temperature=TEMPERATURE, model=DETAILED_ASSESSMENT_MODEL)
        detailed_assessment_tokens += detailed_tokens

        if not result["relevant"]:
            print(f"🚫 Not relevant (detailed assessment): {title}")
            continue

        print(f"✅ Relevant: {title}")

        # Create a paper dict that matches what process_paper expects
        paper_dict = {
            "title": title,
            "abstract": abstract,
            "url": url,
            "authors": authors,
            "date": date,
            "source": "arxiv",
            "arxiv_id": url.split("/")[-1] if "arxiv.org" in url else "",
            "cited_by_count": 0,
            "publication_type": "preprint",
            "code_repository": ""
        }

        row = process_paper(paper_dict)

        # Ensure code repository is empty string if not present
        if "code_repository" not in row or row["code_repository"] == "None":
            row["code_repository"] = ""

        insert_paper(qdrant_client, row)
        relevant.append(row)
        # Rate limiting is now handled automatically by the new system
    
    # Add both token counts to the total
    total_tokens = quick_assessment_tokens + detailed_assessment_tokens
    
    # Print token usage breakdown
    print(f"\n📊 Token usage breakdown:")
    print(f"  Quick assessment: {quick_assessment_tokens} tokens")
    print(f"  Detailed assessment: {detailed_assessment_tokens} tokens")
    print(f"  Total: {total_tokens} tokens")
    
    # Calculate cost savings
    quick_cost = estimate_cost(quick_assessment_tokens, QUICK_ASSESSMENT_MODEL)
    detailed_cost = estimate_cost(detailed_assessment_tokens, DETAILED_ASSESSMENT_MODEL)
    total_cost = quick_cost + detailed_cost
    
    # Estimate what it would have cost without the two-stage approach
    old_approach_cost = estimate_cost(quick_assessment_tokens + detailed_assessment_tokens, DETAILED_ASSESSMENT_MODEL)
    savings = old_approach_cost - total_cost
    
    print(f"💰 Cost breakdown:")
    print(f"  Quick assessment: ${quick_cost:.4f} (using {QUICK_ASSESSMENT_MODEL})")
    print(f"  Detailed assessment: ${detailed_cost:.4f} (using {DETAILED_ASSESSMENT_MODEL})")
    print(f"  Total: ${total_cost:.4f}")
    print(f"  Estimated savings: ${savings:.4f} (compared to single-stage approach)")

    return relevant


def estimate_cost(tokens, model=None):
    """Estimate cost based on token usage and model.

    Args:
        tokens: Number of tokens used
        model: Model name (e.g., 'openai/gpt-4o', 'moonshotai/kimi-dev-72b:free')

    Returns:
        float: Estimated cost in USD
    """
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

    return round(tokens * cost_per_token, 4)


def main():
    # Update daily limit for paid users
    update_daily_limit_for_paid_user()
    
    print("🔄 Fetching ArXiv RSS feeds...")
    arxiv_papers = fetch_arxiv()
    print(f"📚 Found {len(arxiv_papers)} papers from ArXiv")

    print("🧠 Filtering ArXiv papers...")
    arxiv_results = process_papers(arxiv_papers)

    print(f"📝 Generating RSS feed with {len(arxiv_results)} papers...")
    build_rss_feed(arxiv_results)

    cost = estimate_cost(total_tokens, AI_MODEL)
    print(
        f"✅ Done. Total tokens used: {total_tokens}, Estimated cost: ${cost:.4f} (using {AI_MODEL})")


if __name__ == "__main__":
    main()
