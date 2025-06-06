# update_rss.py

import os
import time
import feedparser
import requests
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from feedgen.feed import FeedGenerator
from utils.gpt import assess_relevance_and_tags, assess_paper_quality
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
    cutoff_time = datetime.now(timezone.utc) - timedelta(days=1)

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
        fe.title(paper["Title"])
        fe.link(href=paper["URL"])

        # Build a more readable description
        description = []

        # Summary
        if "Summary" in paper:
            description.append("<h3>Summary</h3>")
            description.append("<ul>")
            for point in paper["Summary"]:
                description.append(f"<li>{point}</li>")
            description.append("</ul>")

        # Paper Type
        if "Paper Type" in paper:
            description.append("<h3>Paper Type</h3>")
            description.append("<ul>")
            description.append(f"<li>{paper['Paper Type']}</li>")
            description.append("</ul>")

        # Quality Assessment
        if any(key in paper for key in ["Clarity", "Novelty", "Significance", "Try-worthiness"]):
            description.append("<h3>Quality Assessment</h3>")
            description.append("<ul>")
            if "Clarity" in paper:
                description.append(f"<li>Clarity: {paper['Clarity']}/5</li>")
            if "Novelty" in paper:
                description.append(f"<li>Novelty: {paper['Novelty']}/5</li>")
            if "Significance" in paper:
                description.append(
                    f"<li>Significance: {paper['Significance']}/5</li>")
            if "Try-worthiness" in paper:
                description.append(
                    f"<li>Try-worthiness: {'Yes' if paper['Try-worthiness'] else 'No'}</li>")
            if "Justification" in paper:
                description.append(
                    f"<li>Justification: {paper['Justification']}</li>")
            description.append("</ul>")

        # Additional Information
        description.append("<h3>Additional Information</h3>")
        description.append("<ul>")
        if "Authors" in paper:
            description.append(f"<li>Authors: {paper['Authors']}</li>")
        if "Tags" in paper:
            description.append(f"<li>Tags: {paper['Tags']}</li>")
        if "Relevance" in paper:
            description.append(
                f"<li>Relevance Score: {paper['Relevance']}/5</li>")
        if "Code repository" in paper and paper["Code repository"]:
            description.append(
                f"<li>Code Repository: <a href='{paper['Code repository']}'>{paper['Code repository']}</a></li>")
        description.append("</ul>")

        fe.description("".join(description))
        fe.author({"name": paper["Authors"]})
        fe.pubDate(datetime.fromisoformat(
            paper["Date"]).astimezone(timezone.utc))
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
            authors = ", ".join([a.get("author", {}).get(
                "display_name", "") for a in paper.get("authorships", [])])
            date = paper.get("publication_date", datetime.now(
                timezone.utc).date().isoformat())
        else:  # arxiv
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
        result, token_count = assess_relevance_and_tags(
            fulltext, OPENAI_API_KEY, temperature=0.1, model="gpt-4.1")
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
            "Relevance": result["relevance_score"],
            "Paper Type": result.get("paper_type", "Other")
        }

        # For arXiv papers, fetch full text and assess quality
        if "arxiv.org" in url:
            try:
                arxiv_id = url.split("/")[-1]
                html_url = f"https://arxiv.org/html/{arxiv_id}"
                html_response = requests.get(html_url)
                if html_response.status_code == 200:
                    print(f"📄 Assessing quality for: {title}")
                    quality = assess_paper_quality(
                        title, html_response.text, OPENAI_API_KEY)
                    if quality:  # Only update if we got valid quality assessment
                        row.update(quality)
                else:
                    print(f"⚠️ Failed to retrieve HTML for: {title}")
            except Exception as e:
                print(f"❌ Error fetching full text: {e}")
                print(f"Continuing without quality assessment for: {title}")

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
