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


def check_paper_relevance(paper):
    """Check if a paper is relevant to AI security."""
    # Baserow returns data in a different format, we need to access it correctly
    title = paper.get("Title", {}).get("value", "")
    url = paper.get("URL", {}).get("value", "")
    abstract = paper.get("Abstract", {}).get("value", "")

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

    relevant_count = 0
    irrelevant_count = 0

    for paper in papers:
        title = paper.get("Title", {}).get("value", "Unknown Title")
        print(f"\n🔍 Checking: {title}")

        is_relevant = check_paper_relevance(paper)

        if is_relevant:
            print("✅ Paper is relevant to AI security")
            relevant_count += 1
        else:
            print("❌ Paper is not relevant to AI security")
            irrelevant_count += 1

            # Update the paper in Baserow to mark it as irrelevant
            # Baserow expects the data in a specific format
            update_data = {
                "Relevance": {"value": 0},
                "Tags": {"value": "irrelevant"}
            }

            # Keep the original paper data structure
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
