import os
import requests
from typing import List, Dict
from dotenv import load_dotenv
from utils.qdrant import init_qdrant_client
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

# Load environment variables
load_dotenv()

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
AI_MODEL = os.getenv("AI_MODEL", "openai/gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))

# Initialize Qdrant client
qdrant_client = init_qdrant_client()

# RAG-specific collection name
RAG_COLLECTION_NAME = "rag_knowledge_base"


def create_openrouter_client(api_key: str):
    """Create an OpenRouter client using requests."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://paper-digest.com",  # Replace with your domain
        "X-Title": "Paper Digest"  # Replace with your app name
    }
    return headers


def ensure_rag_collection_exists():
    """Ensure the RAG collection exists with proper schema."""
    collections = qdrant_client.get_collections().collections
    exists = any(col.name == RAG_COLLECTION_NAME for col in collections)

    if not exists:
        # Create collection with proper vector dimensions for embeddings
        qdrant_client.create_collection(
            collection_name=RAG_COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )


def get_embedding(text: str) -> List[float]:
    """Get embedding for a text using OpenRouter's API."""
    headers = create_openrouter_client(OPENROUTER_API_KEY)
    payload = {
        "input": text,
        "model": "openai/text-embedding-ada-002"
    }

    try:
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/embeddings",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        result = response.json()
        return result["data"][0]["embedding"]
    except Exception as e:
        print(f"❌ Error getting embedding: {str(e)}")
        return []


def add_to_knowledge_base(text: str, metadata: Dict = None):
    """Add a text to the knowledge base."""
    embedding = get_embedding(text)

    # Generate a unique ID for the document
    import uuid
    doc_id = str(uuid.uuid4())

    # Add the document to Qdrant
    qdrant_client.upsert(
        collection_name=RAG_COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=doc_id,
                vector=embedding,
                payload={"text": text, **(metadata or {})}
            )
        ]
    )


def search_knowledge_base(query: str, limit: int = 3) -> List[Dict]:
    """Search the knowledge base for relevant documents."""
    query_embedding = get_embedding(query)

    search_result = qdrant_client.search(
        collection_name=RAG_COLLECTION_NAME,
        query_vector=query_embedding,
        limit=limit
    )

    return [hit.payload for hit in search_result]


def generate_response(query: str) -> str:
    """Generate a response using RAG."""
    # Search for relevant documents
    relevant_docs = search_knowledge_base(query)

    # Prepare the context from relevant documents
    context = "\n".join([doc["text"] for doc in relevant_docs])

    # Create the prompt for the chat model
    prompt = f"""Based on the following context, please answer the question. If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {query}

Answer:"""

    # Generate response using OpenRouter
    headers = create_openrouter_client(OPENROUTER_API_KEY)
    payload = {
        "model": AI_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": prompt}
        ],
        "temperature": TEMPERATURE
    }

    try:
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        result_data = response.json()
        return result_data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Error generating response: {str(e)}"


def main():
    # Ensure RAG collection exists
    ensure_rag_collection_exists()

    print("Welcome to the Knowledge Base Chatbot!")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("\nYour question: ")

        if user_input.lower() == 'exit':
            break

        try:
            response = generate_response(user_input)
            print("\nAnswer:", response)
        except Exception as e:
            print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
