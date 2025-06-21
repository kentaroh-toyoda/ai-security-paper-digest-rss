#!/usr/bin/env python3
"""
Script to remove a paper from the Qdrant KB identified by its URL.
Usage: python remove_paper.py --url "https://openalex.org/W2068405379"
"""

import argparse
import sys
from typing import Optional
from qdrant_client.http import models
from utils.qdrant import init_qdrant_client, ensure_collection_exists, COLLECTION_NAME, generate_point_id


def find_paper_by_url(client, url: str) -> Optional[dict]:
    """Find a paper in Qdrant by its URL.
    
    Args:
        client: Qdrant client instance
        url: URL of the paper to find
        
    Returns:
        A tuple of (point_id, paper_payload) if found, None otherwise
    """
    response = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="url",
                    match=models.MatchValue(value=url)
                )
            ]
        ),
        limit=1
    )
    
    if not response[0]:
        return None
    
    point = response[0][0]
    return (point.id, point.payload)


def delete_paper(client, point_id: str) -> bool:
    """Delete a paper from Qdrant by its point ID.
    
    Args:
        client: Qdrant client instance
        point_id: ID of the point to delete
        
    Returns:
        True if deletion was successful, False otherwise
    """
    try:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.PointIdsList(
                points=[point_id]
            )
        )
        return True
    except Exception as e:
        print(f"Error deleting paper: {str(e)}")
        return False


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Remove a paper from the Qdrant KB identified by its URL.'
    )
    parser.add_argument('--url', required=True, help='URL of the paper to remove')
    
    args = parser.parse_args()
    
    # Initialize Qdrant client
    try:
        client = init_qdrant_client()
        ensure_collection_exists(client, verbose=False)
    except Exception as e:
        print(f"Error connecting to Qdrant: {str(e)}")
        sys.exit(1)
    
    # Find the paper by URL
    paper_info = find_paper_by_url(client, args.url)
    
    if not paper_info:
        print(f"Error: No paper found with URL: {args.url}")
        sys.exit(1)
    
    point_id, paper = paper_info
    
    # Display paper information
    print("Found paper:")
    print(f"Title: {paper.get('title', 'Unknown title')}")
    print(f"URL: {args.url}")
    print(f"Authors: {paper.get('authors', 'Unknown authors')}")
    if 'tags' in paper and paper['tags']:
        print(f"Tags: {', '.join(paper['tags'])}")
    if 'modalities' in paper and paper['modalities']:
        print(f"Modalities: {', '.join(paper['modalities'])}")
    
    # Ask for confirmation
    confirmation = input("\nAre you sure you want to delete this paper? (yes/no): ")
    
    if confirmation.lower() not in ["yes", "y"]:
        print("Operation cancelled.")
        sys.exit(0)
    
    # Delete the paper
    success = delete_paper(client, point_id)
    
    if success:
        print(f"Successfully deleted paper: {paper.get('title', 'Unknown paper')}")
    else:
        print("Failed to delete paper.")
        sys.exit(1)


if __name__ == "__main__":
    main()
