#!/usr/bin/env python3

import os
import sys
import json
from dotenv import load_dotenv
from utils.baserow import get_all_papers, update_paper_in_baserow
from utils.gpt import assess_relevance_and_tags

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


def get_field(paper, field):
    return paper.get(field, "")


def check_paper_relevance(paper):
    """Check if a paper is relevant to AI security."""
    title = get_field(paper, FIELD_TITLE)
    url = get_field(paper, FIELD_URL)
    abstract = get_field(paper, FIELD_ABSTRACT)

    if not title or not url:
        print(f"⚠️ Skipping paper with missing title or URL")
        print(f"Title: {title}")
        print(f"URL: {url}")
        return False

    fulltext = f"Title: {title}\nAbstract: {abstract}\nURL: {url}"
    result, _ = assess_relevance_and_tags(
        fulltext,
        OPENAI_API_KEY,
        temperature=0.1,
        model="gpt-4.1"
    )

    return result["relevant"]


def main():
    if not all([OPENAI_API_KEY, BASEROW_API_TOKEN, BASEROW_TABLE_ID]):
        print("❌ Missing required environment variables")
        sys.exit(1)

    print("📚 Fetching all papers from Baserow...")
    papers = get_all_papers(BASEROW_API_TOKEN, BASEROW_TABLE_ID)
    print(f"Found {len(papers)} papers to check")

    if papers:
        print("\n🔎 First record for debugging:")
        print(json.dumps(papers[0], indent=2))

    relevant_count = 0
    irrelevant_count = 0

    for paper in papers:
        title = get_field(paper, FIELD_TITLE) or "Unknown Title"
        print(f"\n🔍 Checking: {title}")

        is_relevant = check_paper_relevance(paper)

        if is_relevant:
            print("✅ Paper is relevant to AI security")
            relevant_count += 1
        else:
            print("❌ Paper is not relevant to AI security")
            irrelevant_count += 1

            # Update the paper in Baserow to mark it as irrelevant
            update_data = {
                FIELD_RELEVANCE: 0,
                FIELD_TAGS: "irrelevant"
            }
            paper.update(update_data)

            if update_paper_in_baserow(paper, BASEROW_API_TOKEN, BASEROW_TABLE_ID):
                print("📝 Updated paper in Baserow")
            else:
                print("❌ Failed to update paper in Baserow")

    print(f"\n📊 Summary:")
    print(f"Total papers checked: {len(papers)}")
    print(f"Relevant papers: {relevant_count}")
    print(f"Irrelevant papers: {irrelevant_count}")


if __name__ == "__main__":
    main()
