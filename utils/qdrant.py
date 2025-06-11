import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import uuid

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
    """Ensure the papers collection exists with proper schema."""
    collections = client.get_collections().collections
    exists = any(col.name == COLLECTION_NAME for col in collections)

    if not exists:
        # Create collection with a dummy vector - we're using Qdrant as a document store
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1, distance=Distance.COSINE),
        )

        # Create payload indexes for efficient filtering
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="url",
            field_schema="keyword"
        )
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="date",
            field_schema="datetime"
        )


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

        # Create point with dummy vector since we're using Qdrant as a document store
        point = PointStruct(
            id=generate_point_id(paper_data["url"]),  # Generate UUID from URL
            vector=[0.0],  # Dummy vector
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
