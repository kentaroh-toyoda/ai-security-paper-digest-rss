import os
from dotenv import load_dotenv
from utils.qdrant import init_qdrant_client, ensure_collection_exists

# Load environment variables
load_dotenv()


def create_relevance_score_index():
    """Create the relevance_score index in Qdrant."""
    # Initialize Qdrant client
    client = init_qdrant_client()

    # This will ensure the collection exists and create all necessary indexes
    ensure_collection_exists(client)

    print("✅ Relevance score index has been created or already exists")


if __name__ == "__main__":
    print("Creating relevance score index...")
    create_relevance_score_index()
    print("Done!")
