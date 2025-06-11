import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

COLLECTION_NAME = "ai_security_papers"


def init_qdrant_client() -> QdrantClient:
    """Initialize Qdrant client with environment variables."""
    url = os.getenv("QDRANT_API_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url or not api_key:
        raise ValueError(
            "QDRANT_API_URL and QDRANT_API_KEY must be set in environment")

    return QdrantClient(url=url, api_key=api_key)


def ensure_collection_exists(client: QdrantClient) -> None:
    """Ensure the collection exists in Qdrant."""
    try:
        # Create collection if it doesn't exist
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if COLLECTION_NAME not in collection_names:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=1536,  # OpenAI embedding dimension
                    distance=models.Distance.COSINE
                )
            )
            print(f"Created collection: {COLLECTION_NAME}")

            # Create index for title field
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="title",
                field_schema="keyword"
            )
            print("Created index for title field")
        else:
            print(f"Collection {COLLECTION_NAME} already exists")

            # Create index for title field if it doesn't exist
            try:
                client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name="title",
                    field_schema="keyword"
                )
                print("Created index for title field")
            except Exception as e:
                if "already exists" in str(e).lower():
                    print("Title index already exists")
                else:
                    raise

    except Exception as e:
        print(f"Error ensuring collection exists: {str(e)}")
        raise


def generate_point_id(url: str) -> str:
    """Generate a unique ID for a paper based on its URL.
    Returns a UUID string that's compatible with Qdrant's ID requirements."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, url))


def paper_exists(client: QdrantClient, paper_url: str) -> bool:
    """Check if a paper with the given URL already exists."""
    response = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="url",
                    match=models.MatchValue(value=paper_url)
                )
            ]
        ),
        limit=1
    )
    return len(response[0]) > 0


def insert_paper(client: QdrantClient, paper_data: Dict[str, Any]) -> bool:
    """Insert a paper into Qdrant."""
    try:
        # Convert date string to datetime if needed
        if isinstance(paper_data.get("date"), str):
            paper_data["date"] = datetime.fromisoformat(
                paper_data["date"].replace("Z", "+00:00"))

        # Create point with zero vector of correct dimension (1536)
        point = PointStruct(
            id=generate_point_id(paper_data["url"]),  # Generate UUID from URL
            vector=[0.0] * 1536,  # Zero vector of correct dimension
            payload=paper_data
        )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )
        print(f"✅ Added to Qdrant: {paper_data.get('title', 'Unknown paper')}")
        return True

    except Exception as e:
        print(f"❌ Error pushing to Qdrant: {str(e)}")
        if hasattr(e, 'response'):
            print("Raw response content:")
            print(e.response.content)
        return False


def get_recent_papers(client: QdrantClient, hours: int = 24) -> List[Dict[str, Any]]:
    """Get papers from the last N hours."""
    cutoff = datetime.now() - timedelta(hours=hours)

    response = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="date",
                    range=models.Range(
                        gte=cutoff
                    )
                )
            ]
        ),
        limit=100  # Adjust as needed
    )

    return [point.payload for point in response[0]]


def get_all_papers(client: QdrantClient) -> List[Dict[str, Any]]:
    """Get all papers from the collection."""
    response = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10000  # Adjust as needed
    )
    return [point.payload for point in response[0]]
