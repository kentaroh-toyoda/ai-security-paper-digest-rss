# update_rss.py

import os
import sys
import time
import argparse
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
DETAILED_ASSESSMENT_MODEL = os.getenv(
    "DETAILED_ASSESSMENT_MODEL", "openai/gpt-4.1-mini")
QUICK_ASSESSMENT_MODEL = os.getenv(
    "QUICK_ASSESSMENT_MODEL", "openai/gpt-4.1-nano")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))

# Constants
FEEDS = [
    "https://export.arxiv.org/rss/cs.AI",
    "https://export.arxiv.org/rss/cs.LG",
    "https://export.arxiv.org/rss/cs.CL",
    "https://export.arxiv.org/rss/cs.CV",
    "https://aclanthology.org/papers/index.xml",
]

# Configuration for different feed types
FEED_CONFIGS = {
    "ai-security": {
        "title": "AI Security Paper Digest",
        "description": "Curated papers on AI security from ArXiv and ACL",
        "output_file": "rss.xml",
        "collection_name": "ai_security_papers",
        "feed_type": "ai-security"
    },
    "web3-security": {
        "title": "Web3 Security Paper Digest",
        "description": "Curated papers on Web3, blockchain, and smart contract security from ArXiv and ACL",
        "output_file": "web3_security_rss.xml",
        "collection_name": "web3_security_papers",
        "feed_type": "web3-security"
    }
}


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate RSS feed for security papers")
    parser.add_argument(
        "--feed-type",
        type=str,
        choices=["ai-security", "web3-security"],
        default="ai-security",
        help="Type of security feed to generate (default: ai-security)"
    )
    return parser.parse_args()


# Token usage tracking
total_tokens = 0


def fetch_papers():
    entries = []
    cutoff_time = datetime.now(timezone.utc) - timedelta(days=1)
    print(f"\n🔍 Fetching papers since: {cutoff_time.isoformat()}")

    for feed_url in FEEDS:
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


def build_rss_feed(relevant_papers, config):
    fg = FeedGenerator()
    fg.title(config["title"])
    fg.link(href=RSS_FEED_URL)
    fg.description(config["description"])

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
        if "topics" in paper:
            description.append(f"<li>Tags: {paper['topics']}</li>")
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
            if isinstance(paper["published_date"], str):
                # Ensure the date string is in ISO format
                if "T" not in paper["published_date"]:  # If it's just a date without time
                    paper["published_date"] = f"{paper['published_date']}T00:00:00+00:00"
                pub_date = datetime.fromisoformat(
                    paper["published_date"].replace("Z", "+00:00"))
            else:
                # If date is already a datetime object, use it directly
                pub_date = paper["published_date"]
            fe.pubDate(pub_date.astimezone(timezone.utc))
        except (ValueError, TypeError) as e:
            print(
                f"Warning: Could not parse date for paper {paper['title']}: {e}")
            # Use current time as fallback
            fe.pubDate(datetime.now(timezone.utc))

        fe.guid(paper["url"])

    fg.rss_file(config["output_file"])


def process_paper(paper: dict, feed_type: str = "ai-security") -> dict:
    """Process a paper and prepare it for storage."""
    paper_data = {
        "title": paper.get("title", ""),
        "abstract": paper.get("abstract", ""),
        "url": paper.get("url", ""),
        "published_date": paper.get("date", ""),  # Renamed from 'date'
        "authors": paper.get("authors", ""),
        "source": paper.get("source", ""),
        "paper_id": paper.get("arxiv_id", ""),  # Renamed from 'arxiv_id'
        "cited_by_count": paper.get("cited_by_count", 0),
        "publication_type": paper.get("publication_type", ""),
        "code_repository": paper.get("code_repository", ""),
        "is_relevant": False,
        "topics": [],  # Renamed from 'tags'
        "relevance_score": 0,
        "relevance_reason": "",
        "paper_type": "Other",
        "modalities": [],
        "star": False  # New field, default to False
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
            model=DETAILED_ASSESSMENT_MODEL,
            feed_type=feed_type
        )

    if result.get("relevant", False):
        paper_data["is_relevant"] = True
        paper_data["topics"] = result.get("tags", [])
        paper_data["relevance_score"] = result.get("relevance_score", 0)
        paper_data["relevance_reason"] = result.get("reason", "")
        paper_data["paper_type"] = result.get("paper_type", "Research Paper")
        paper_data["modalities"] = result.get("modalities", [])
        paper_data["summary"] = result.get("summary", [])

    return paper_data


def process_papers(raw_papers, feed_type: str, collection_name: str, qdrant_client):
    global total_tokens
    relevant = []
    current_date = datetime.now(timezone.utc).date()

    # Track token usage for different models
    quick_assessment_tokens = 0
    detailed_assessment_tokens = 0

    # Define models for different stages - using environment variables
    # These variables are defined in .env and loaded at the top of the file

    # Define delay between detailed assessments to avoid rate limiting
    # Only apply delay for models subject to free tier limit (like kimi)
    # With 20 requests per minute allowed, we can safely use a 1.5 second delay (60/20 = 3, but we can be a bit more aggressive)
    DETAILED_ASSESSMENT_DELAY = 1.5  # seconds between requests

    # Check if the detailed assessment model is subject to rate limiting
    is_free_tier_model = DETAILED_ASSESSMENT_MODEL.endswith(':free')

    for paper in raw_papers:
        title = paper.title if hasattr(paper, 'title') else ""
        url = paper.link if hasattr(paper, 'link') else ""
        abstract = paper.summary if hasattr(paper, 'summary') else ""
        date = datetime.now(timezone.utc).date().isoformat()

        # Determine source and parse metadata accordingly
        if "aclanthology.org" in url:
            source = "acl"
            # For ACL, authors are in the description
            description = paper.summary if hasattr(paper, 'summary') else ""
            # Extract authors from description (format: "Author1 and Author2 in Proceedings...")
            if " in " in description:
                authors_part = description.split(" in ")[0]
                authors = [author.strip() for author in authors_part.split(" and ")]
            else:
                authors = ["Unknown"]
            paper_id = url.split("/")[-1] if url else ""
            publication_type = "conference"
            # For ACL, use title only for assessment
            assessment_text = f"Title: {title}"
        else:
            source = "arxiv"
            authors = [author.strip() for author in paper.author.split(
                ",")] if hasattr(paper, 'author') else ["Unknown"]
            paper_id = url.split("/")[-1] if "arxiv.org" in url else ""
            publication_type = "preprint"
            # For ArXiv, use title + abstract
            assessment_text = f"Title: {title}\n\nAbstract: {abstract}"

        if not title or not url:
            continue

        print(f"\n📅 Processing paper published on: {date}")
        print(f"📄 Title: {title}")

        if paper_exists(qdrant_client, url, collection_name):
            print(f"⏭️ Already exists: {title}")
            continue

        fulltext = f"Title: {title}\nAbstract: {abstract}\nURL: {url}"

        # STAGE 1: Quick assessment with cheaper model
        print(f"🔍 Quick relevance assessment...")
        potentially_relevant, quick_tokens = quick_assess_relevance(
            fulltext, OPENROUTER_API_KEY, temperature=TEMPERATURE, model=QUICK_ASSESSMENT_MODEL, feed_type=feed_type)
        quick_assessment_tokens += quick_tokens

        if not potentially_relevant:
            print(f"🚫 Not relevant (quick assessment): {title}")
            continue

        print(f"✓ Potentially relevant (quick assessment): {title}")

        # STAGE 2: Detailed assessment with more expensive model
        print(f"🔍 Detailed relevance assessment...")

        # Add delay between detailed assessments if using a free tier model
        # This helps avoid hitting the rate limit (20 requests/60 seconds)
        if is_free_tier_model and len(relevant) > 0:
            delay_time = DETAILED_ASSESSMENT_DELAY
            print(
                f"⏱️ Adding delay of {delay_time}s before detailed assessment to avoid rate limiting...")
            time.sleep(delay_time)

        result, detailed_tokens = assess_relevance_and_tags(
            assessment_text, OPENROUTER_API_KEY, temperature=TEMPERATURE, model=DETAILED_ASSESSMENT_MODEL, feed_type=feed_type)
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
            "source": source,
            "arxiv_id": paper_id,
            "cited_by_count": 0,
            "publication_type": publication_type,
            "code_repository": ""
        }

        row = process_paper(paper_dict, feed_type=feed_type)

        # Ensure code repository is empty string if not present
        if "code_repository" not in row or row["code_repository"] == "None":
            row["code_repository"] = ""

        insert_paper(qdrant_client, row, collection_name)
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
    detailed_cost = estimate_cost(
        detailed_assessment_tokens, DETAILED_ASSESSMENT_MODEL)
    total_cost = quick_cost + detailed_cost

    # Estimate what it would have cost without the two-stage approach
    old_approach_cost = estimate_cost(
        quick_assessment_tokens + detailed_assessment_tokens, DETAILED_ASSESSMENT_MODEL)
    savings = old_approach_cost - total_cost

    print(f"💰 Cost breakdown:")
    print(
        f"  Quick assessment: ${quick_cost:.4f} (using {QUICK_ASSESSMENT_MODEL})")
    print(
        f"  Detailed assessment: ${detailed_cost:.4f} (using {DETAILED_ASSESSMENT_MODEL})")
    print(f"  Total: ${total_cost:.4f}")
    print(
        f"  Estimated savings: ${savings:.4f} (compared to single-stage approach)")

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
        model = DETAILED_ASSESSMENT_MODEL

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
        "openai/gpt-5": 0.01,  # $10.00 per 1M tokens
        "openai/gpt-5-mini": 0.00015,  # $0.15 per 1M tokens
        "openai/gpt-5-nano": 0.000075,  # $0.075 per 1M tokens

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
    # Parse command-line arguments
    args = parse_arguments()
    feed_type = args.feed_type

    # Get configuration for the selected feed type
    config = FEED_CONFIGS[feed_type]
    collection_name = config["collection_name"]

    print(f"🔧 Generating {config['title']}...")
    print(f"📁 Using collection: {collection_name}")
    print(f"📄 Output file: {config['output_file']}")

    # Update daily limit for paid users
    update_daily_limit_for_paid_user()

    # Initialize Qdrant client and ensure collection exists
    qdrant_client = init_qdrant_client()
    ensure_collection_exists(qdrant_client, collection_name)

    print("🔄 Fetching RSS feeds...")
    papers = fetch_papers()
    print(f"📚 Found {len(papers)} papers")

    print("🧠 Filtering papers...")
    results = process_papers(papers, feed_type, collection_name, qdrant_client)

    print(f"📝 Generating RSS feed with {len(results)} papers...")
    build_rss_feed(results, config)

    cost = estimate_cost(total_tokens, DETAILED_ASSESSMENT_MODEL)
    print(
        f"✅ Done. Total tokens used: {total_tokens}, Estimated cost: ${cost:.4f} (using {DETAILED_ASSESSMENT_MODEL})")


if __name__ == "__main__":
    main()
