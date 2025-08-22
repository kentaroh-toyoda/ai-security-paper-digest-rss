import os
from typing import Dict, Any
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct
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
    """Ensure the collection exists in Qdrant with updated schema indexes."""
    try:
        # Create collection if it doesn't exist
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if COLLECTION_NAME not in collection_names:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=384,  # sentence-transformers/all-MiniLM-L6-v2 dimension
                    distance=models.Distance.COSINE
                )
            )
            print(f"Created collection: {COLLECTION_NAME}")

            # Create indexes for metadata fields with new schema
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="metadata.title",
                field_schema="keyword"
            )
            print("Created index for metadata.title field")

            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="metadata.url",
                field_schema="keyword"
            )
            print("Created index for metadata.url field")

            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="metadata.authors",
                field_schema="keyword"
            )
            print("Created index for metadata.authors field")

            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="metadata.topics",
                field_schema="keyword"
            )
            print("Created index for metadata.topics field")

            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="metadata.modalities",
                field_schema="keyword"
            )
            print("Created index for metadata.modalities field")

            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="metadata.star",
                field_schema="bool"
            )
            print("Created index for metadata.star field")

            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="metadata.paper_type",
                field_schema="keyword"
            )
            print("Created index for metadata.paper_type field")

            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="metadata.source",
                field_schema="keyword"
            )
            print("Created index for metadata.source field")

            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="metadata.published_date",
                field_schema="datetime"
            )
            print("Created index for metadata.published_date field")

            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="metadata.relevance_score",
                field_schema="integer"
            )
            print("Created index for metadata.relevance_score field")

        else:
            print(f"Collection {COLLECTION_NAME} already exists")

            # Create indexes for metadata fields if they don't exist
            index_configs = [
                ("metadata.title", "keyword"),
                ("metadata.url", "keyword"),
                ("metadata.authors", "keyword"),
                ("metadata.topics", "keyword"),
                ("metadata.modalities", "keyword"),
                ("metadata.star", "bool"),
                ("metadata.paper_type", "keyword"),
                ("metadata.source", "keyword"),
                ("metadata.published_date", "datetime"),
                ("metadata.relevance_score", "integer")
            ]

            for field_name, field_schema in index_configs:
                try:
                    client.create_payload_index(
                        collection_name=COLLECTION_NAME,
                        field_name=field_name,
                        field_schema=field_schema
                    )
                    print(f"Created index for {field_name} field")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        print(f"{field_name} index already exists")
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
                    key="metadata.url",
                    match=models.MatchValue(value=paper_url)
                )
            ]
        ),
        limit=1
    )
    return len(response[0]) > 0





def insert_paper(client: QdrantClient, paper_data: Dict[str, Any]) -> bool:
    """Insert a paper into Qdrant with the new schema."""
    try:
        # Prepare metadata payload according to the new schema
        metadata = {
            "paper_id": paper_data.get("paper_id", paper_data.get("arxiv_id", "")),  # Use paper_id if available, fallback to arxiv_id
            "title": paper_data.get("title", ""),
            "authors": paper_data.get("authors", []),
            "published_date": paper_data.get("published_date", paper_data.get("date", "")),
            "topics": paper_data.get("topics", paper_data.get("tags", [])),
            "summary": paper_data.get("summary", []),
            "paper_type": paper_data.get("paper_type", ""),
            "modalities": paper_data.get("modalities", []),
            "embedding_source": ["title", "abstract"],
            "embedding_size": 384,
            "embedding_model_version": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_distance": "cosine",
            "source": paper_data.get("source", ""),
            "url": paper_data.get("url", ""),
            "code_repository": paper_data.get("code_repository", ""),
            "star": paper_data.get("star", False)
        }

        # Ensure authors is a list
        if isinstance(metadata["authors"], str):
            metadata["authors"] = [
                author.strip() for author in metadata["authors"].split(",") if author.strip()]

        # Generate text for embedding
        embedding_text = f"Title: {paper_data.get('title', '')}\n\nAbstract: {paper_data.get('abstract', '')}"

        # Generate embedding
        from utils.llm import generate_embeddings
        vector = generate_embeddings(embedding_text)

        # Create point with the new schema
        point = PointStruct(
            id=generate_point_id(paper_data["url"]),  # Generate UUID from URL
            vector=vector,  # Embedding vector
            payload={
                "embedding": vector,  # Include the embedding in the payload as per schema
                "metadata": metadata
            }
        )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )
        print(f"✅ Added to Qdrant with new schema: {paper_data.get('title', 'Unknown paper')}")
        return True

    except Exception as e:
        print(f"❌ Error pushing to Qdrant: {str(e)}")
        if hasattr(e, 'response'):
            print("Raw response content:")
            print(e.response.content)
        return False
