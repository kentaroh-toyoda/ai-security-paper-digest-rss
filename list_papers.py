#!/usr/bin/env python3
"""
Script to list papers from the Qdrant KB in a tabular format.
Includes search/filtering functionality based on Authors, Tags, and Modalities.
"""

import os
import argparse
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from utils.qdrant import init_qdrant_client, ensure_collection_exists, get_all_papers
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

try:
    from tabulate import tabulate
except ImportError:
    print("The 'tabulate' package is required. Please install it using:")
    print("pip install tabulate")
    exit(1)

# Load environment variables
load_dotenv()


def filter_papers(papers: List[Dict[str, Any]],
                  authors: Optional[str] = None,
                  tags: Optional[str] = None,
                  modalities: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Filter papers based on search criteria.

    Args:
        papers: List of paper dictionaries
        authors: Comma-separated list of authors to filter by
        tags: Comma-separated list of tags to filter by
        modalities: Comma-separated list of modalities to filter by

    Returns:
        Filtered list of papers
    """
    filtered_papers = papers

    # Filter by authors
    if authors:
        author_list = [a.strip().lower() for a in authors.split(',')]
        filtered_papers = [
            paper for paper in filtered_papers
            if any(author.lower() in paper.get('authors', '').lower() for author in author_list)
        ]

    # Filter by tags
    if tags:
        tag_list = [t.strip().lower() for t in tags.split(',')]
        filtered_papers = [
            paper for paper in filtered_papers
            if any(tag.lower() in [t.lower() for t in paper.get('tags', [])] for tag in tag_list)
        ]

    # Filter by modalities
    if modalities:
        modality_list = [m.strip().lower() for m in modalities.split(',')]
        filtered_papers = [
            paper for paper in filtered_papers
            if any(modality.lower() in [m.lower() for m in paper.get('modalities', [])] for modality in modality_list)
        ]

    return filtered_papers


def display_papers(papers: List[Dict[str, Any]],
                   show_abstract: bool = False,
                   show_summary: bool = False,
                   max_width: int = 80) -> None:
    """
    Display papers in a tabular format.

    Args:
        papers: List of paper dictionaries
        show_abstract: Whether to show the abstract
        show_summary: Whether to show the summary
        max_width: Maximum width for text fields
    """
    if not papers:
        print("No papers found matching the search criteria.")
        return

    # Prepare table data
    table_data = []

    # Define headers based on options
    headers = ["Date", "Title", "URL", "Modalities", "Tags"]
    if show_abstract:
        headers.append("Abstract")
    if show_summary:
        headers.append("Summary")

    # Truncate text to max_width
    def truncate(text, max_len=max_width):
        if isinstance(text, str) and len(text) > max_len:
            return text[:max_len-3] + "..."
        return text

    # Format lists as comma-separated strings
    def format_list(lst):
        if not lst:
            return ""
        if isinstance(lst, list):
            return ", ".join(lst)
        return str(lst)

    # Add data for each paper
    for paper in papers:
        # Get basic paper info
        title = truncate(paper.get('title', 'N/A'))
        date = paper.get('date', 'N/A')
        # Truncate the time part from the date (e.g., "2025-06-17T00:00:00" to "2025-06-17")
        if isinstance(date, str) and 'T' in date:
            date = date.split('T')[0]
        url = truncate(paper.get('url', 'N/A'))

        # Get modalities and tags as lists
        modalities_list = paper.get('modalities', [])
        if not modalities_list:
            modalities_list = ['N/A']

        tags_list = paper.get('tags', [])
        if not tags_list:
            tags_list = ['N/A']

        # Get optional columns
        abstract = truncate(paper.get('abstract', 'N/A')
                            ) if show_abstract else None
        summary = truncate(format_list(
            paper.get('summary', []))) if show_summary else None

        # Determine the maximum number of rows needed for this paper
        max_rows = max(len(modalities_list), len(tags_list))

        # Create rows for this paper
        for i in range(max_rows):
            # For the first row, include all main fields
            if i == 0:
                row = [
                    date,
                    title,
                    url,
                    modalities_list[i] if i < len(modalities_list) else '',
                    tags_list[i] if i < len(tags_list) else ''
                ]

                # Add optional columns
                if show_abstract:
                    row.append(abstract)
                if show_summary:
                    row.append(summary)

            # For subsequent rows, only include modalities and tags
            else:
                row = [
                    '',  # Empty date
                    '',  # Empty title
                    '',  # Empty URL
                    modalities_list[i] if i < len(modalities_list) else '',
                    tags_list[i] if i < len(tags_list) else ''
                ]

                # Add empty cells for optional columns
                if show_abstract:
                    row.append('')
                if show_summary:
                    row.append('')

            table_data.append(row)

        # Add a separator row between papers (empty row)
        separator_row = [''] * len(headers)
        table_data.append(separator_row)

    # Display table
    print(tabulate(table_data, headers=headers, tablefmt="grid", showindex=False))
    print(f"\nTotal papers: {len(papers)}")


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Namespace object with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='List papers from the Qdrant KB in a tabular format.'
    )
    parser.add_argument('--authors', type=str,
                        help='Filter by authors (comma-separated)')
    parser.add_argument('--tags', type=str,
                        help='Filter by tags (comma-separated)')
    parser.add_argument('--modalities', type=str,
                        help='Filter by modalities (comma-separated)')
    parser.add_argument('--abstract', action='store_true',
                        help='Show abstract')
    parser.add_argument('--summary', action='store_true', help='Show summary')
    parser.add_argument('--max-width', type=int, default=80,
                        help='Maximum width for text fields')
    
    return parser.parse_args()



def main():
    # Parse command line arguments
    args = parse_arguments()

    # Initialize Qdrant client
    client = init_qdrant_client()
    ensure_collection_exists(client, verbose=False)

    # Get all papers
    print("Fetching papers from Qdrant KB...")
    papers = get_all_papers(client)

    if not papers:
        print("No papers found in the Qdrant KB.")
        return

    # Filter papers based on search criteria
    filtered_papers = filter_papers(
        papers,
        authors=args.authors,
        tags=args.tags,
        modalities=args.modalities
    )

    # Sort papers by date (oldest to newest)
    filtered_papers = sorted(filtered_papers, key=lambda x: x.get('date', ''))

    # Display papers
    display_papers(
        filtered_papers,
        show_abstract=args.abstract,
        show_summary=args.summary,
        max_width=args.max_width
    )


if __name__ == "__main__":
    main()
