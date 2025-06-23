#!/usr/bin/env python3
"""
Script to delete a tag from a Qdrant point identified by a URL.
Usage: python delete_tags.py --url <URL> <TAG>
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


def delete_tag_from_point(client, point_id: str, tag_to_delete: str):
    """Delete a tag from a point in Qdrant."""
    # First, get the current point to retrieve existing tags
    point = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[point_id]
    )[0]
    
    # Get existing tags or initialize empty list
    existing_tags = point.payload.get("tags", [])
    
    # Check if the tag exists
    if tag_to_delete not in existing_tags:
        return False, existing_tags
    
    # Remove the tag
    updated_tags = [tag for tag in existing_tags if tag != tag_to_delete]
    
    # Update the point with the new tags
    client.set_payload(
        collection_name=COLLECTION_NAME,
        payload={"tags": updated_tags},
        points=[point_id]
    )
    
    return True, updated_tags


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Delete a tag from a Qdrant point identified by a URL.'
    )
    parser.add_argument('tag', help='Tag to delete from the point')
    parser.add_argument('--url', required=True, help='URL of the paper')
    
    args = parser.parse_args()
    
    # Initialize Qdrant client
    try:
        client = init_qdrant_client()
        ensure_collection_exists(client)
    except Exception as e:
        print(f"Error connecting to Qdrant: {str(e)}")
        sys.exit(1)
    
    # Find the point by URL
    point = find_point_by_url(client, args.url)
    
    if not point:
        print(f"Error: No paper found with URL: {args.url}")
        sys.exit(1)
    
    # Delete tag from the point
    try:
        success, remaining_tags = delete_tag_from_point(client, point.id, args.tag)
        
        if success:
            print(f"Successfully deleted tag '{args.tag}' from paper: {point.payload.get('title', 'Unknown paper')}")
            print(f"URL: {args.url}")
            if remaining_tags:
                print(f"Remaining tags: {', '.join(remaining_tags)}")
            else:
                print("No tags remaining.")
        else:
            print(f"Tag '{args.tag}' not found in paper: {point.payload.get('title', 'Unknown paper')}")
            if remaining_tags:
                print(f"Existing tags: {', '.join(remaining_tags)}")
            else:
                print("Paper has no tags.")
    except Exception as e:
        print(f"Error deleting tag: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
