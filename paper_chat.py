import os
import re
import requests
from typing import Optional, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

from utils.baserow import (
    store_conversation,
    get_conversation_history
)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Get Baserow configuration
BASEROW_API_TOKEN = os.getenv("BASEROW_API_TOKEN")
BASEROW_CONVERSATION_TABLE_ID = os.getenv("BASEROW_CONVERSATION_TABLE_ID")

# Debug print
# Only show first 5 chars for security
print(f"Debug - BASEROW_API_TOKEN: {BASEROW_API_TOKEN[:5]}...")
print(
    f"Debug - BASEROW_CONVERSATION_TABLE_ID: {BASEROW_CONVERSATION_TABLE_ID}")

if not BASEROW_CONVERSATION_TABLE_ID:
    print("❌ BASEROW_CONVERSATION_TABLE_ID is not set. Please update your .env file.")
    exit(1)


def extract_arxiv_id(text: str) -> Optional[str]:
    """Extract arXiv ID from various formats."""
    # Match patterns like:
    # - 2301.12345
    # - arxiv:2301.12345
    # - https://arxiv.org/abs/2301.12345
    patterns = [
        r'(\d{4}\.\d{4,5})',  # Just the ID
        r'arxiv:(\d{4}\.\d{4,5})',  # arxiv: prefix
        r'arxiv\.org/abs/(\d{4}\.\d{4,5})',  # URL format
    ]

    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1)
    return None


def fetch_paper_content(arxiv_id: str) -> Dict[str, Any]:
    """Fetch paper content from arXiv."""
    # First get the paper metadata
    abs_url = f"https://arxiv.org/abs/{arxiv_id}"
    html_url = f"https://arxiv.org/html/{arxiv_id}"

    try:
        # Get HTML content
        html_response = requests.get(html_url)
        if html_response.status_code != 200:
            return {"error": f"Failed to fetch paper HTML: {html_response.status_code}"}

        # Get abstract page for metadata
        abs_response = requests.get(abs_url)
        if abs_response.status_code != 200:
            return {"error": f"Failed to fetch paper metadata: {abs_response.status_code}"}

        # Extract title from abstract page
        title_match = re.search(
            r'<h1 class="title mathjax">(.*?)</h1>', abs_response.text)
        if not title_match:
            title_match = re.search(
                r'<h1 class="title">(.*?)</h1>', abs_response.text)

        title_html = title_match.group(1) if title_match else "Unknown Title"
        # Remove HTML tags from the title
        title = re.sub(
            r'<[^>]+>', '', title_html).replace("Title:", "").strip()

        # Extract abstract
        abstract_match = re.search(
            r'<blockquote class="abstract mathjax">(.*?)</blockquote>', abs_response.text, re.DOTALL)
        abstract = abstract_match.group(1).strip() if abstract_match else ""

        return {
            "title": title,
            "abstract": abstract,
            "html_content": html_response.text,
            "url": abs_url
        }
    except Exception as e:
        return {"error": f"Error fetching paper: {str(e)}"}


def create_chat_context(paper_content: Dict[str, Any], conversation_history: list = None) -> str:
    """Create a context string for the chat model."""
    if "error" in paper_content:
        return f"Error: {paper_content['error']}"

    # Extract main content from HTML
    html_content = paper_content["html_content"]

    # Remove HTML tags and clean up the text
    text = re.sub(r'<[^>]+>', ' ', html_content)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    # Truncate if too long (GPT-4 has a context limit)
    max_length = 12000
    if len(text) > max_length:
        text = text[:max_length] + "..."

    context = f"""Paper Title: {paper_content['title']}
Paper URL: {paper_content['url']}
Abstract: {paper_content.get('abstract', 'No abstract available')}

Paper Content:
{text}

You are a helpful AI assistant that answers questions about this academic paper. 
Please provide clear, accurate, and concise answers based on the paper's content.
If the answer cannot be found in the paper, say so clearly.
When answering:
1. Be specific and cite relevant parts of the paper
2. Explain technical terms if they might be unfamiliar
3. If the answer is not in the paper, say so clearly
4. If you're unsure about something, acknowledge the uncertainty"""

    # Add conversation history if available
    if conversation_history:
        context += "\n\nPrevious conversation:\n"
        for entry in conversation_history:
            context += f"\nQ: {entry['Question']}\nA: {entry['Answer']}\n"

    return context


def chat_with_paper(arxiv_id: str, question: str, paper_url: str) -> str:
    """Chat with the paper using GPT."""
    # Fetch paper content
    paper_content = fetch_paper_content(arxiv_id)
    if "error" in paper_content:
        return f"❌ {paper_content['error']}"

    # Get conversation history
    conversation_history = []
    if BASEROW_CONVERSATION_TABLE_ID:
        conversation_history = get_conversation_history(
            BASEROW_API_TOKEN, BASEROW_CONVERSATION_TABLE_ID, paper_url)

    # Create chat context
    context = create_chat_context(paper_content, conversation_history)

    try:
        # Get response from GPT
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": question}
            ],
            temperature=0.1
        )
        answer = response.choices[0].message.content.strip()

        # Store conversation if table exists
        if BASEROW_CONVERSATION_TABLE_ID:
            store_conversation(
                BASEROW_API_TOKEN,
                BASEROW_CONVERSATION_TABLE_ID,
                paper_url,
                question,
                answer
            )

        return answer
    except Exception as e:
        return f"❌ Error getting response from GPT: {str(e)}"


def show_conversation_history(paper_url: str):
    """Show conversation history for a paper."""
    if not BASEROW_CONVERSATION_TABLE_ID:
        return

    history = get_conversation_history(
        BASEROW_API_TOKEN, BASEROW_CONVERSATION_TABLE_ID, paper_url)
    if not history:
        return

    print("\n📜 Previous conversations:")
    for entry in history:
        print(f"\nQ: {entry['Question']}")
        print(f"A: {entry['Answer']}")
        print(f"Date: {entry['Timestamp']}")


def main():
    print("🤖 Welcome to the Paper Chat Bot!")
    print("Enter an arXiv paper ID (e.g., 2301.12345) or 'quit' to exit.")

    while True:
        # Get paper ID
        paper_input = input("\n📄 Enter arXiv paper ID: ").strip()
        if paper_input.lower() == 'quit':
            break

        arxiv_id = extract_arxiv_id(paper_input)
        if not arxiv_id:
            print("❌ Invalid arXiv ID format. Please try again.")
            continue

        print(f"\n📚 Fetching paper {arxiv_id}...")
        paper_content = fetch_paper_content(arxiv_id)
        if "error" in paper_content:
            print(f"❌ {paper_content['error']}")
            continue

        print(f"\n📖 Paper: {paper_content['title']}")

        # Show conversation history if available
        show_conversation_history(paper_content['url'])

        print(
            "💬 Ask questions about the paper (type 'new' for a new paper or 'quit' to exit)")

        # Chat loop for this paper
        while True:
            question = input("\n❓ Your question: ").strip()
            if question.lower() == 'quit':
                return
            if question.lower() == 'new':
                break

            if not question:
                continue

            print("\n🤔 Thinking...")
            answer = chat_with_paper(arxiv_id, question, paper_content['url'])
            print(f"\n📝 Answer: {answer}")


if __name__ == "__main__":
    main()
