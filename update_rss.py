# update_rss.py

import os
import time
import feedparser
import requests
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from feedgen.feed import FeedGenerator
from utils.gpt import assess_relevance_and_tags, assess_paper_quality
from utils.qdrant import init_qdrant_client, ensure_collection_exists, paper_exists, insert_paper

load_dotenv()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_URL = os.getenv("QDRANT_API_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
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

# Initialize Qdrant client
qdrant_client = init_qdrant_client()
ensure_collection_exists(qdrant_client)


def fetch_openalex_24h():
    since = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
    print(f"\n🔍 Fetching OpenAlex papers since: {since}")

    all_results = []
    page = 1
    per_page = 100  # Maximum page size

    while True:
        url = f"{OPENALEX_URL}?filter=from_publication_date:{since}&per-page={per_page}&page={page}&mailto={OPENALEX_EMAIL}"
        print(f"📄 Fetching page {page}...")
        resp = requests.get(url)
        data = resp.json()
        results = data.get("results", [])

        if not results:  # No more results
            break

        all_results.extend(results)
        print(f"📚 Found {len(results)} papers on page {page}")

        # Check if we've reached the last page
        meta = data.get("meta", {})
        if page >= meta.get("page_count", 1):
            break

        page += 1
        time.sleep(1)  # Be nice to the API

    # Filter papers based on their availability date
    current_date = datetime.now(timezone.utc).date()
    valid_results = []
    for paper in all_results:
        pub_date = paper.get("publication_date")
        if pub_date:
            try:
                paper_date = datetime.strptime(pub_date, "%Y-%m-%d").date()
                # Include papers that are either:
                # 1. Published in the last 24 hours, or
                # 2. In early access (future publication date) but available now
                if paper_date <= current_date or paper.get("is_oa", False):
                    valid_results.append(paper)
                else:
                    print(
                        f"ℹ️ Skipping future paper (not yet available): {paper.get('title', 'Unknown')} (pub date: {pub_date})")
            except ValueError:
                print(
                    f"⚠️ Skipping paper with invalid date format {pub_date}: {paper.get('title', 'Unknown')}")

    print(
        f"📊 Found {len(all_results)} total papers, {len(valid_results)} available for processing")
    return valid_results


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

        # Quality Assessment
        if any(key in paper for key in ["clarity", "novelty", "significance", "try_worthiness"]):
            description.append("<h3>Quality Assessment</h3>")
            description.append("<ul>")
            if "clarity" in paper:
                description.append(f"<li>Clarity: {paper['clarity']}/5</li>")
            if "novelty" in paper:
                description.append(f"<li>Novelty: {paper['novelty']}/5</li>")
            if "significance" in paper:
                description.append(
                    f"<li>Significance: {paper['significance']}/5</li>")
            if "try_worthiness" in paper:
                description.append(
                    f"<li>Try-worthiness: {'Yes' if paper['try_worthiness'] else 'No'}</li>")
            if "justification" in paper:
                description.append(
                    f"<li>Justification: {paper['justification']}</li>")
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
        "openalex_id": paper.get("openalex_id", ""),
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

    # Assess relevance and get tags
    result, token_count = assess_relevance_and_tags(
        text=f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}",
        api_key=OPENAI_API_KEY
    )

    if result.get("relevant", False):
        paper_data["is_relevant"] = True
        paper_data["tags"] = result.get("tags", [])
        paper_data["relevance_score"] = result.get("relevance_score", 0)
        paper_data["relevance_reason"] = result.get("reason", "")
        paper_data["paper_type"] = result.get("paper_type", "Research Paper")
        paper_data["modalities"] = result.get("modalities", [])
        paper_data["summary"] = result.get("summary", [])

        # Assess paper quality
        quality = assess_paper_quality(paper_data, OPENAI_API_KEY)
        paper_data.update(quality)

    return paper_data


def process_papers(raw_papers, source):
    global total_tokens
    relevant = []
    current_date = datetime.now(timezone.utc).date()

    for paper in raw_papers:
        # Extract info based on source
        if source == "openalex":
            title = paper.get("title", "")
            url = paper.get("id", "")
            abstract = paper.get("abstract", "") or ""
            authors = ", ".join([a.get("author", {}).get(
                "display_name", "") for a in paper.get("authorships", [])])
            date = paper.get("publication_date", datetime.now(
                timezone.utc).date().isoformat())
            is_early_access = paper.get("is_oa", False)
        else:  # arxiv
            title = paper.title if hasattr(paper, 'title') else ""
            url = paper.link if hasattr(paper, 'link') else ""
            abstract = paper.summary if hasattr(paper, 'summary') else ""
            authors = paper.author if hasattr(paper, 'author') else "Unknown"
            date = datetime.now(timezone.utc).date().isoformat()
            is_early_access = False

        if not title or not url:
            continue

        # Validate date
        try:
            paper_date = datetime.strptime(date, "%Y-%m-%d").date()
            if paper_date > current_date and not is_early_access:
                print(
                    f"ℹ️ Skipping future paper (not yet available): {title} (pub date: {date})")
                continue
        except ValueError:
            print(
                f"⚠️ Skipping paper with invalid date format {date}: {title}")
            continue

        print(f"\n📅 Processing paper published on: {date}")
        if is_early_access:
            print("📢 Note: This is an early access paper")
        print(f"📄 Title: {title}")

        if paper_exists(qdrant_client, url):
            print(f"⏭️ Already exists: {title}")
            continue

        fulltext = f"Title: {title}\nAbstract: {abstract}\nURL: {url}"
        result, token_count = assess_relevance_and_tags(
            fulltext, OPENAI_API_KEY, temperature=0.1, model="gpt-4.1")
        total_tokens += token_count

        if not result["relevant"]:
            print(f"🚫 Not relevant: {title}")
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
            "arxiv_id": url.split("/")[-1] if "arxiv.org" in url else "",
            "openalex_id": paper.get("id", "") if source == "openalex" else "",
            "cited_by_count": paper.get("cited_by_count", 0) if source == "openalex" else 0,
            "publication_type": "preprint" if source == "arxiv" else paper.get("type", ""),
            "code_repository": ""
        }

        row = process_paper(paper_dict)

        # For arXiv papers, fetch full text and assess quality
        if "arxiv.org" in url:
            try:
                arxiv_id = url.split("/")[-1]
                html_url = f"https://arxiv.org/html/{arxiv_id}"
                html_response = requests.get(html_url)
                if html_response.status_code == 200:
                    print(f"📄 Assessing quality for: {title}")
                    # Create metadata dictionary for quality assessment
                    metadata = {
                        'title': title,
                        'abstract': abstract,
                        'date': date,
                        'cited_by_count': 0,  # arXiv doesn't provide citation count
                        'publication_type': 'preprint',
                        'source': 'arXiv',
                        'code_repository': ''  # Empty string instead of None
                    }
                    quality = assess_paper_quality(metadata, OPENAI_API_KEY)
                    if quality:  # Only update if we got valid quality assessment
                        # Convert "None" to empty string for code repository
                        if "code_repository" in quality and quality["code_repository"] == "None":
                            quality["code_repository"] = ""
                        row.update(quality)
                else:
                    print(f"⚠️ Failed to retrieve HTML for: {title}")
            except Exception as e:
                print(f"❌ Error fetching full text: {e}")
                print(f"Continuing without quality assessment for: {title}")

        # Ensure code repository is empty string if not present
        if "code_repository" not in row or row["code_repository"] == "None":
            row["code_repository"] = ""

        insert_paper(qdrant_client, row)
        relevant.append(row)
        time.sleep(1.5)  # prevent rate limiting

    return relevant


def estimate_cost(tokens):
    cost_per_token = 0.01 / 1000  # GPT-4.1 input pricing ($0.01/1K tokens)
    return round(tokens * cost_per_token, 4)


def main():
    print("🔄 Fetching OpenAlex papers (past 24h)...")
    openalex_papers = fetch_openalex_24h()
    print(f"📚 Found {len(openalex_papers)} papers from OpenAlex")

    print("🔄 Fetching ArXiv RSS feeds...")
    arxiv_papers = fetch_arxiv()
    print(f"📚 Found {len(arxiv_papers)} papers from ArXiv")

    print("🧠 Filtering OpenAlex papers...")
    openalex_results = process_papers(openalex_papers, source="openalex")

    print("🧠 Filtering ArXiv papers...")
    arxiv_results = process_papers(arxiv_papers, source="arxiv")

    all_results = openalex_results + arxiv_results

    print(f"📝 Generating RSS feed with {len(all_results)} papers...")
    build_rss_feed(all_results)

    cost = estimate_cost(total_tokens)
    print(
        f"✅ Done. Total tokens used: {total_tokens}, Estimated cost: ${cost:.4f}")


if __name__ == "__main__":
    main()
