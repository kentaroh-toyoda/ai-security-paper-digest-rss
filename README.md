# Knowledge Base Chatbot

This is a chatbot that uses the existing Qdrant database as a vector store for RAG (Retrieval Augmented Generation). It can answer questions based on the knowledge stored in the vector database.

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure your `.env` file has the following variables:

```
OPENROUTER_API_KEY=your_openrouter_api_key
QDRANT_API_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_cloud_api_key
AI_MODEL=openai/gpt-4o-mini  # Optional: specify the model to use
TEMPERATURE=0.1              # Optional: specify the temperature (0.0 to 1.0)
```

3. Add documents to the knowledge base:

```python
from db_chat import add_to_knowledge_base

# Add a document to the knowledge base
add_to_knowledge_base("Your text here", metadata={"source": "document_name"})
```

## Usage

Run the chatbot:

```bash
python db_chat.py
```

The chatbot will:

1. Use the existing Qdrant collection
2. Start an interactive session where you can ask questions
3. Use RAG to find relevant information and generate answers
4. Type 'exit' to quit the chatbot

## Features

- Uses OpenRouter's text-embedding-ada-002 model for generating embeddings
- Uses configurable AI models (default: GPT-4o-mini) for generating responses
- Implements RAG using the existing Qdrant database
- Supports metadata for documents
- Interactive command-line interface
- Configurable model and temperature via environment variables

## Integration with Existing System

This chatbot integrates with the existing Qdrant setup in `utils/qdrant.py`, which is already configured for storing and retrieving documents. The chatbot adds RAG capabilities on top of this existing infrastructure.
