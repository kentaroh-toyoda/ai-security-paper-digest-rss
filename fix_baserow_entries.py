#!/usr/bin/env python3

import os
import sys
import json
import requests
import feedparser
from datetime import datetime, timezone
from dotenv import load_dotenv
from utils.baserow import get_all_papers, update_paper_in_baserow
from utils.gpt import assess_relevance_and_tags, assess_paper_quality

load_dotenv()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASEROW_API_TOKEN = os.getenv("BASEROW_API_TOKEN")
BASEROW_TABLE_ID = os.getenv("BASEROW_TABLE_ID")

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

# Required fields and their default values
REQUIRED_FIELDS = {
    FIELD_TITLE: "",
    FIELD_URL: "",
    FIELD_ABSTRACT: "",
    FIELD_TAGS: "",
    FIELD_AUTHORS: "",
    FIELD_DATE: datetime.now(timezone.utc).date().isoformat(),
    FIELD_RELEVANCE: 0,
    FIELD_SUMMARY: [],
    FIELD_PAPER_TYPE: "Other",
    FIELD_CLARITY: 0,
    FIELD_NOVELTY: 0,
    FIELD_SIGNIFICANCE: 0,
    FIELD_TRY_WORTHINESS: False,
    FIELD_JUSTIFICATION: "",
    FIELD_CODE_REPO: ""
}


def get_field(paper, field):
    """Get a field value from a paper, with proper type conversion."""
    value = paper.get(field, "")
    if field == FIELD_SUMMARY and isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return value


def fetch_arxiv_data(url):
    """Fetch additional data from arXiv."""
    try:
        arxiv_id = url.split("/")[-1]
        html_url = f"https://arxiv.org/html/{arxiv_id}"
        html_response = requests.get(html_url)
        if html_response.status_code == 200:
            return html_response.text
    except Exception as e:
        print(f"❌ Error fetching arXiv data: {e}")
    return None


def fetch_openalex_data(url):
    """Fetch additional data from OpenAlex."""
    try:
        # Extract the OpenAlex ID from the URL
        openalex_id = url.split("/")[-1]
        api_url = f"https://api.openalex.org/works/{openalex_id}"
        response = requests.get(api_url)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"❌ Error fetching OpenAlex data: {e}")
    return None


def fix_paper_data(paper):
    """Fix missing data in a paper entry."""
    url = get_field(paper, FIELD_URL)
    if not url:
        print("⚠️ Paper has no URL, cannot fetch additional data")
        return paper

    print(f"\n🔍 Fixing data for: {get_field(paper, FIELD_TITLE)}")
    needs_update = False
    update_data = {}

    # Check for missing required fields
    for field, default_value in REQUIRED_FIELDS.items():
        current_value = get_field(paper, field)
        if not current_value or current_value == default_value:
            print(f"⚠️ Missing or empty field: {field}")
            needs_update = True

    # If we need to fetch additional data
    if needs_update:
        if "arxiv.org" in url:
            # Fetch from arXiv
            html_content = fetch_arxiv_data(url)
            if html_content:
                # Create metadata for quality assessment
                metadata = {
                    'title': get_field(paper, FIELD_TITLE),
                    'abstract': get_field(paper, FIELD_ABSTRACT),
                    'date': get_field(paper, FIELD_DATE),
                    'cited_by_count': 0,
                    'publication_type': 'preprint',
                    'source': 'arXiv',
                    'code_url': ''
                }
                quality = assess_paper_quality(metadata, OPENAI_API_KEY)
                if quality:
                    update_data.update(quality)

        elif "openalex.org" in url:
            # Fetch from OpenAlex
            openalex_data = fetch_openalex_data(url)
            if openalex_data:
                # Extract relevant fields from OpenAlex data
                if not get_field(paper, FIELD_ABSTRACT):
                    update_data[FIELD_ABSTRACT] = openalex_data.get(
                        "abstract", "")
                if not get_field(paper, FIELD_AUTHORS):
                    authors = [a.get("author", {}).get("display_name", "")
                               for a in openalex_data.get("authorships", [])]
                    update_data[FIELD_AUTHORS] = ", ".join(authors)

        # If we still have missing fields, try to generate them using GPT
        if not get_field(paper, FIELD_SUMMARY) or not get_field(paper, FIELD_TAGS):
            fulltext = f"Title: {get_field(paper, FIELD_TITLE)}\nAbstract: {get_field(paper, FIELD_ABSTRACT)}\nURL: {url}"
            result, _ = assess_relevance_and_tags(fulltext, OPENAI_API_KEY)
            if result["relevant"]:
                update_data[FIELD_SUMMARY] = result["summary"]
                update_data[FIELD_TAGS] = ", ".join(result["tags"])
                update_data[FIELD_RELEVANCE] = result["relevance_score"]
                update_data[FIELD_PAPER_TYPE] = result.get(
                    "paper_type", "Other")

    # Update the paper with any new data
    if update_data:
        paper.update(update_data)
        print("✅ Updated paper data")
        return paper
    else:
        print("ℹ️ No updates needed")
        return None


def main():
    if not all([OPENAI_API_KEY, BASEROW_API_TOKEN, BASEROW_TABLE_ID]):
        print("❌ Missing required environment variables")
        sys.exit(1)

    print("📚 Fetching all papers from Baserow...")
    papers = get_all_papers(BASEROW_API_TOKEN, BASEROW_TABLE_ID)
    print(f"Found {len(papers)} papers to check")

    fixed_count = 0
    skipped_count = 0

    for paper in papers:
        updated_paper = fix_paper_data(paper)
        if updated_paper:
            if update_paper_in_baserow(updated_paper, BASEROW_API_TOKEN, BASEROW_TABLE_ID):
                print("📝 Updated paper in Baserow")
                fixed_count += 1
            else:
                print("❌ Failed to update paper in Baserow")
        else:
            skipped_count += 1

    print(f"\n📊 Summary:")
    print(f"Total papers checked: {len(papers)}")
    print(f"Papers fixed: {fixed_count}")
    print(f"Papers skipped: {skipped_count}")


if __name__ == "__main__":
    main()
