#!/usr/bin/env python3
"""
Script to add tags to a Qdrant point identified by a URL.
Usage: python add_tags.py "tag1" "tag2" ... --url "https://example.com"
"""

import argparse
import sys
from typing import List
from qdrant_client.http import models
from utils.qdrant import init_qdrant_client, ensure_collection_exists, COLLECTION_NAME


def find_point_by_url(client, url: str):
    """Find a point in Qdrant by its URL."""
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
    
    return response[0][0]


def add_tags_to_point(client, point_id: str, new_tags: List[str]):
    """Add tags to a point in Qdrant."""
    # First, get the current point to retrieve existing tags
    point = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[point_id]
    )[0]
    
    # Get existing tags or initialize empty list
    existing_tags = point.payload.get("tags", [])
    
    # Combine existing and new tags, removing duplicates
    all_tags = list(set(existing_tags + new_tags))
    
    # Update the point with the new tags
    client.set_payload(
        collection_name=COLLECTION_NAME,
        payload={"tags": all_tags},
        points=[point_id]
    )
    
    return all_tags


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Add tags to a Qdrant point identified by a URL.'
    )
    parser.add_argument('tags', nargs='+', help='Tags to add to the point')
    parser.add_argument('--url', required=True, help='URL of the paper to tag')
    
    args = parser.parse_args()
    
    # Initialize Qdrant client
    try:
        client = init_qdrant_client()
        ensure_collection_exists(client, verbose=False)
    except Exception as e:
        print(f"Error connecting to Qdrant: {str(e)}")
        sys.exit(1)
    
    # Find the point by URL
    point = find_point_by_url(client, args.url)
    
    if not point:
        print(f"Error: No paper found with URL: {args.url}")
        sys.exit(1)
    
    # Add tags to the point
    try:
        all_tags = add_tags_to_point(client, point.id, args.tags)
        print(f"Successfully added tags to paper: {point.payload.get('title', 'Unknown paper')}")
        print(f"URL: {args.url}")
        print(f"All tags: {', '.join(all_tags)}")
    except Exception as e:
        print(f"Error adding tags: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
