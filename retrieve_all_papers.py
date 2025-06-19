#!/usr/bin/env python3
"""
Script to retrieve all papers from the Qdrant knowledge base.
"""

import json
import sys
from typing import List, Dict, Any
from utils.qdrant import init_qdrant_client, get_all_papers


def format_paper_output(paper: Dict[str, Any], index: int) -> str:
    """Format a single paper for display."""
    output = f"\n{'-' * 80}\n"
    output += f"Paper #{index + 1}\n"
    output += f"{'-' * 80}\n"

    # Title
    if paper.get('title'):
        output += f"Title: {paper['title']}\n"

    # Authors
    if paper.get('authors'):
        authors = paper['authors']
        if isinstance(authors, list):
            authors_str = ', '.join(authors)
        else:
            authors_str = str(authors)
        output += f"Authors: {authors_str}\n"

    # URL
    if paper.get('url'):
        output += f"URL: {paper['url']}\n"

    # Date
    if paper.get('date'):
        output += f"Date: {paper['date']}\n"

    # Modalities
    if paper.get('modalities'):
        modalities = paper['modalities']
        if isinstance(modalities, list):
            modalities_str = ', '.join(modalities)
        else:
            modalities_str = str(modalities)
        output += f"Modalities: {modalities_str}\n"

    # Code Repository
    if paper.get('code_repository'):
        output += f"Code Repository: {paper['code_repository']}\n"

    # Relevance Score
    if paper.get('relevance_score'):
        output += f"Relevance Score: {paper['relevance_score']}\n"

    # Abstract (if available)
    if paper.get('abstract'):
        output += f"Abstract: {paper['abstract'][:200]}...\n" if len(
            paper['abstract']) > 200 else f"Abstract: {paper['abstract']}\n"

    return output


def retrieve_all_papers(output_format: str = "display", output_file: str = None) -> List[Dict[str, Any]]:
    """
    Retrieve all papers from the Qdrant knowledge base.

    Args:
        output_format: Format for output ('display', 'json', 'csv')
        output_file: Optional file path to save output

    Returns:
        List of paper dictionaries
    """
    try:
        # Initialize Qdrant client
        print("Connecting to Qdrant...")
        client = init_qdrant_client()

        # Retrieve all papers
        print("Retrieving all papers...")
        papers = get_all_papers(client)

        print(f"Found {len(papers)} papers in the knowledge base.")

        if not papers:
            print("No papers found in the knowledge base.")
            return []

        # Process output based on format
        if output_format == "json":
            output_data = papers
            output_str = json.dumps(papers, indent=2, default=str)
        elif output_format == "csv":
            # Create CSV format
            if papers:
                headers = list(papers[0].keys())
                output_str = ",".join(headers) + "\n"
                for paper in papers:
                    row = []
                    for header in headers:
                        value = paper.get(header, "")
                        if isinstance(value, list):
                            value = ";".join(str(v) for v in value)
                        elif isinstance(value, (dict, list)):
                            value = str(value)
                        row.append(f'"{str(value).replace('"', '""')}"')
                    output_str += ",".join(row) + "\n"
            else:
                output_str = ""
        else:  # display format
            output_str = f"\n{'=' * 80}\n"
            output_str += f"ALL PAPERS IN QDRANT KNOWLEDGE BASE ({len(papers)} papers)\n"
            output_str += f"{'=' * 80}\n"

            for i, paper in enumerate(papers):
                output_str += format_paper_output(paper, i)

        # Output to console
        print(output_str)

        # Save to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output_str)
            print(f"\nOutput saved to: {output_file}")

        return papers

    except Exception as e:
        print(f"Error retrieving papers: {str(e)}")
        return []


def main():
    """Main function to handle command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Retrieve all papers from Qdrant knowledge base")
    parser.add_argument(
        "--format",
        choices=["display", "json", "csv"],
        default="display",
        help="Output format (default: display)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (optional)"
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only show the count of papers"
    )

    args = parser.parse_args()

    if args.count_only:
        # Just get the count
        try:
            client = init_qdrant_client()
            papers = get_all_papers(client)
            print(f"Total papers in knowledge base: {len(papers)}")
        except Exception as e:
            print(f"Error: {str(e)}")
            sys.exit(1)
    else:
        # Get all papers with specified format
        papers = retrieve_all_papers(args.format, args.output)

        if not papers:
            sys.exit(1)


if __name__ == "__main__":
    main()
