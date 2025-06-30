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

            # Create index for url field
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="url",
                field_schema="keyword"
            )
            print("Created index for url field")

            # Create index for authors field
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="authors",
                field_schema="keyword"
            )
            print("Created index for authors field")

            # Create index for modalities field
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="modalities",
                field_schema="keyword"
            )
            print("Created index for modalities field")

            # Create index for code_repository field
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="code_repository",
                field_schema="keyword"
            )
            print("Created index for code_repository field")

            # Create index for relevance_score field
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="relevance_score",
                field_schema="integer"
            )
            print("Created index for relevance_score field")
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

            # Create index for url field if it doesn't exist
            try:
                client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name="url",
                    field_schema="keyword"
                )
                print("Created index for url field")
            except Exception as e:
                if "already exists" in str(e).lower():
                    print("URL index already exists")
                else:
                    raise

            # Create index for authors field if it doesn't exist
            try:
                client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name="authors",
                    field_schema="keyword"
                )
                print("Created index for authors field")
            except Exception as e:
                if "already exists" in str(e).lower():
                    print("Authors index already exists")
                else:
                    raise

            # Create index for modalities field if it doesn't exist
            try:
                client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name="modalities",
                    field_schema="keyword"
                )
                print("Created index for modalities field")
            except Exception as e:
                if "already exists" in str(e).lower():
                    print("Modalities index already exists")
                else:
                    raise

            # Create index for code_repository field if it doesn't exist
            try:
                client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name="code_repository",
                    field_schema="keyword"
                )
                print("Created index for code_repository field")
            except Exception as e:
                if "already exists" in str(e).lower():
                    print("Code repository index already exists")
                else:
                    raise

            # Create index for relevance_score field if it doesn't exist
            try:
                client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name="relevance_score",
                    field_schema="integer"
                )
                print("Created index for relevance_score field")
            except Exception as e:
                if "already exists" in str(e).lower():
                    print("Relevance score index already exists")
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

        # Convert authors string to array if it's not already
        if isinstance(paper_data.get("authors"), str):
            paper_data["authors"] = [
                author.strip() for author in paper_data["authors"].split(",") if author.strip()]

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


def search_papers_by_modalities(client: QdrantClient, modalities: List[str]) -> List[Dict[str, Any]]:
    """Search papers by their modalities.

    Args:
        client: Qdrant client instance
        modalities: List of modalities to search for (e.g., ["text", "image", "video", "audio", "multimodal"])

    Returns:
        List of paper payloads that match the specified modalities
    """
    try:
        # Create a filter that matches any of the specified modalities
        filter_conditions = []
        for modality in modalities:
            filter_conditions.append(
                models.FieldCondition(
                    key="modalities",
                    match=models.MatchValue(value=modality)
                )
            )

        # Use OR condition to match any of the modalities
        response = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                should=filter_conditions  # Use 'should' for OR condition
            ),
            limit=1000  # Adjust as needed
        )

        return [point.payload for point in response[0]]
    except Exception as e:
        print(f"Error searching papers by modalities: {str(e)}")
        return []


def search_papers_by_relevance_score(client: QdrantClient, min_score: int = 1, max_score: int = 5) -> List[Dict[str, Any]]:
    """Search papers by their relevance score.

    Args:
        client: Qdrant client instance
        min_score: Minimum relevance score (inclusive)
        max_score: Maximum relevance score (inclusive)

    Returns:
        List of paper payloads that match the relevance score range
    """
    try:
        # Create a filter for the relevance score range
        response = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="relevance_score",
                        range=models.Range(
                            gte=min_score,
                            lte=max_score
                        )
                    )
                ]
            ),
            limit=1000  # Adjust as needed
        )

        return [point.payload for point in response[0]]
    except Exception as e:
        print(f"Error searching papers by relevance score: {str(e)}")
        return []


def get_paper_by_url(client: QdrantClient, url: str) -> Optional[Dict[str, Any]]:
    """Get a paper by its URL.

    Args:
        client: Qdrant client instance
        url: The paper URL (e.g., "https://arxiv.org/abs/2411.14133")

    Returns:
        Paper payload if found, None otherwise
    """
    try:
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
        
        if response[0]:
            return response[0][0].payload
        return None
    except Exception as e:
        print(f"Error getting paper by URL: {str(e)}")
        return None


def get_paper_by_arxiv_id(client: QdrantClient, arxiv_id: str) -> Optional[Dict[str, Any]]:
    """Get a paper by its arXiv ID (deprecated - use get_paper_by_url instead).

    Args:
        client: Qdrant client instance
        arxiv_id: The arXiv ID (e.g., "2411.14133")

    Returns:
        Paper payload if found, None otherwise
    """
    try:
        response = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="arxiv_id",
                        match=models.MatchValue(value=arxiv_id)
                    )
                ]
            ),
            limit=1
        )
        
        if response[0]:
            return response[0][0].payload
        return None
    except Exception as e:
        print(f"Error getting paper by arXiv ID: {str(e)}")
        return None


def insert_structured_paper(client: QdrantClient, paper_data: Dict[str, Any]) -> bool:
    """Insert a structured paper with sections into Qdrant.
    
    Args:
        client: Qdrant client instance
        paper_data: Dictionary containing structured paper data with sections
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure required fields exist
        if "url" not in paper_data:
            print("❌ Paper data must contain 'url' field")
            return False
            
        # Add processed timestamp
        paper_data["processed_date"] = datetime.now().isoformat()
        
        # Initialize past_conversations if not present
        if "past_conversations" not in paper_data:
            paper_data["past_conversations"] = []
        
        # Convert date string to datetime if needed
        if isinstance(paper_data.get("date"), str):
            try:
                paper_data["date"] = datetime.fromisoformat(
                    paper_data["date"].replace("Z", "+00:00"))
            except ValueError:
                # If date parsing fails, use current date
                paper_data["date"] = datetime.now()

        # Convert authors string to array if it's not already
        if isinstance(paper_data.get("authors"), str):
            paper_data["authors"] = [
                author.strip() for author in paper_data["authors"].split(",") if author.strip()]

        # Create point with zero vector of correct dimension (1536)
        point = PointStruct(
            id=generate_point_id(paper_data["url"]),
            vector=[0.0] * 1536,  # Zero vector of correct dimension
            payload=paper_data
        )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )
        print(f"✅ Added structured paper to Qdrant: {paper_data.get('title', 'Unknown paper')}")
        return True

    except Exception as e:
        print(f"❌ Error inserting structured paper to Qdrant: {str(e)}")
        if hasattr(e, 'response'):
            print("Raw response content:")
            print(e.response.content)
        return False


def update_paper_conversations(client: QdrantClient, url: str, user_prompt: str, ai_response: str) -> bool:
    """Update a paper's conversation history.
    
    Args:
        client: Qdrant client instance
        url: The URL of the paper
        user_prompt: The user's question/prompt
        ai_response: The AI's response
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get the existing paper
        paper = get_paper_by_url(client, url)
        if not paper:
            print(f"❌ Paper with URL {url} not found")
            return False
        
        # Add new conversation entry
        new_conversation = {
            "timestamp": datetime.now().isoformat(),
            "user_prompt": user_prompt,
            "ai_response": ai_response
        }
        
        # Initialize past_conversations if it doesn't exist
        if "past_conversations" not in paper:
            paper["past_conversations"] = []
        
        paper["past_conversations"].append(new_conversation)
        
        # Update the paper in Qdrant
        point = PointStruct(
            id=generate_point_id(paper["url"]),
            vector=[0.0] * 1536,  # Zero vector of correct dimension
            payload=paper
        )
        
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )
        
        print(f"✅ Updated conversation history for paper: {paper.get('title', 'Unknown paper')}")
        return True
        
    except Exception as e:
        print(f"❌ Error updating paper conversations: {str(e)}")
        return False


def update_paper_conversations_by_arxiv_id(client: QdrantClient, arxiv_id: str, user_prompt: str, ai_response: str) -> bool:
    """Update a paper's conversation history by arXiv ID (deprecated - use update_paper_conversations instead).
    
    Args:
        client: Qdrant client instance
        arxiv_id: The arXiv ID of the paper
        user_prompt: The user's question/prompt
        ai_response: The AI's response
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get the existing paper
        paper = get_paper_by_arxiv_id(client, arxiv_id)
        if not paper:
            print(f"❌ Paper with arXiv ID {arxiv_id} not found")
            return False
        
        # Use the URL-based function
        return update_paper_conversations(client, paper["url"], user_prompt, ai_response)
        
    except Exception as e:
        print(f"❌ Error updating paper conversations: {str(e)}")
        return False


def update_paper_attacks(client: QdrantClient, url: str, attack_data: Dict[str, Any]) -> bool:
    """Update a paper with attack analysis data.
    
    Args:
        client: Qdrant client instance
        url: The URL of the paper
        attack_data: Dictionary containing attack analysis results
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get the existing paper
        paper = get_paper_by_url(client, url)
        if not paper:
            print(f"❌ Paper with URL {url} not found")
            return False
        
        # Add attack analysis data
        paper["attack_analysis"] = attack_data
        paper["attack_analysis_date"] = datetime.now().isoformat()
        
        # Update the paper in Qdrant
        point = PointStruct(
            id=generate_point_id(paper["url"]),
            vector=[0.0] * 1536,  # Zero vector of correct dimension
            payload=paper
        )
        
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )
        
        print(f"✅ Updated attack analysis for paper: {paper.get('title', 'Unknown paper')}")
        return True
        
    except Exception as e:
        print(f"❌ Error updating paper attack analysis: {str(e)}")
        return False


def update_paper_attacks_by_arxiv_id(client: QdrantClient, arxiv_id: str, attack_data: Dict[str, Any]) -> bool:
    """Update a paper with attack analysis data by arXiv ID (deprecated - use update_paper_attacks instead).
    
    Args:
        client: Qdrant client instance
        arxiv_id: The arXiv ID of the paper
        attack_data: Dictionary containing attack analysis results
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get the existing paper
        paper = get_paper_by_arxiv_id(client, arxiv_id)
        if not paper:
            print(f"❌ Paper with arXiv ID {arxiv_id} not found")
            return False
        
        # Use the URL-based function
        return update_paper_attacks(client, paper["url"], attack_data)
        
    except Exception as e:
        print(f"❌ Error updating paper attack analysis: {str(e)}")
        return False


def get_papers_with_attacks(client: QdrantClient) -> List[Dict[str, Any]]:
    """Get all papers that have attack analysis data.
    
    Args:
        client: Qdrant client instance
        
    Returns:
        List of papers with attack analysis
    """
    try:
        response = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="attack_analysis",
                        match=models.MatchExcept(value=None)
                    )
                ]
            ),
            limit=1000
        )
        
        return [point.payload for point in response[0]]
    except Exception as e:
        print(f"Error getting papers with attacks: {str(e)}")
        return []


def get_papers_with_implementable_attacks(client: QdrantClient) -> List[Dict[str, Any]]:
    """Get papers that have implementable attacks (attacks_found = true).
    
    Args:
        client: Qdrant client instance
        
    Returns:
        List of papers with implementable attacks
    """
    try:
        response = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="attack_analysis.analysis_results.attacks_found",
                        match=models.MatchValue(value=True)
                    )
                ]
            ),
            limit=1000
        )
        
        return [point.payload for point in response[0]]
    except Exception as e:
        print(f"Error getting papers with implementable attacks: {str(e)}")
        return []
