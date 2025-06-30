#!/usr/bin/env python3
"""
Paper Reader Script for ArXiv Papers

This script downloads and processes ArXiv papers, extracting their content,
structuring it into sections, and storing it in Qdrant for future analysis.

Usage:
    python paper_read.py --url https://arxiv.org/abs/2411.14133
"""

import os
import re
import argparse
import requests
import PyPDF2
from io import BytesIO
from typing import Optional, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

from utils.llm import (
    extract_paper_structure_with_cost,
    analyze_paper_content,
    check_rate_limit_status,
    format_cost
)
from utils.qdrant import (
    init_qdrant_client,
    ensure_collection_exists,
    get_paper_by_url,
    get_paper_by_arxiv_id,
    insert_structured_paper
)

# Load environment variables
load_dotenv()

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def extract_arxiv_id(url: str) -> Optional[str]:
    """Extract arXiv ID from various URL formats.
    
    Args:
        url: ArXiv URL in various formats
        
    Returns:
        ArXiv ID if found, None otherwise
    """
    # Match patterns like:
    # - https://arxiv.org/abs/2411.14133
    # - https://arxiv.org/pdf/2411.14133
    # - 2411.14133
    patterns = [
        r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})',  # URL format
        r'^(\d{4}\.\d{4,5})$',  # Just the ID
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    print(f"❌ Could not extract arXiv ID from: {url}")
    return None


def download_pdf(arxiv_id: str) -> Optional[bytes]:
    """Download PDF from arXiv.
    
    Args:
        arxiv_id: The arXiv ID (e.g., "2411.14133")
        
    Returns:
        PDF content as bytes if successful, None otherwise
    """
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    try:
        print(f"📥 Downloading PDF from: {pdf_url}")
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        
        print(f"✅ Downloaded PDF ({len(response.content)} bytes)")
        return response.content
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading PDF: {str(e)}")
        return None


def extract_text_from_pdf(pdf_content: bytes) -> Optional[str]:
    """Extract text from PDF content.
    
    Args:
        pdf_content: PDF content as bytes
        
    Returns:
        Extracted text if successful, None otherwise
    """
    try:
        print("📄 Extracting text from PDF...")
        
        # Create a PDF reader object
        pdf_file = BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Extract text from all pages
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
        
        # Clean up the text
        text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
        text = text.strip()
        
        print(f"✅ Extracted text ({len(text)} characters)")
        return text
        
    except Exception as e:
        print(f"❌ Error extracting text from PDF: {str(e)}")
        return None


def get_paper_metadata(arxiv_id: str) -> Dict[str, Any]:
    """Get paper metadata from arXiv abstract page.
    
    Args:
        arxiv_id: The arXiv ID
        
    Returns:
        Dictionary containing paper metadata
    """
    abs_url = f"https://arxiv.org/abs/{arxiv_id}"
    
    try:
        print(f"📋 Fetching metadata from: {abs_url}")
        response = requests.get(abs_url, timeout=30)
        response.raise_for_status()
        
        html_content = response.text
        
        # Extract title
        title_match = re.search(r'<h1 class="title mathjax">(.*?)</h1>', html_content)
        if not title_match:
            title_match = re.search(r'<h1 class="title">(.*?)</h1>', html_content)
        
        title_html = title_match.group(1) if title_match else "Unknown Title"
        title = re.sub(r'<[^>]+>', '', title_html).replace("Title:", "").strip()
        
        # Extract authors
        authors_match = re.search(r'<div class="authors">(.*?)</div>', html_content, re.DOTALL)
        authors = []
        if authors_match:
            authors_html = authors_match.group(1)
            # Extract author names from links
            author_matches = re.findall(r'<a[^>]*>(.*?)</a>', authors_html)
            authors = [re.sub(r'<[^>]+>', '', author).strip() for author in author_matches]
        
        # Extract abstract
        abstract_match = re.search(r'<blockquote class="abstract mathjax">(.*?)</blockquote>', html_content, re.DOTALL)
        abstract = ""
        if abstract_match:
            abstract_html = abstract_match.group(1)
            abstract = re.sub(r'<[^>]+>', '', abstract_html).replace("Abstract:", "").strip()
        
        # Extract submission date
        date_match = re.search(r'<div class="dateline">(.*?)</div>', html_content)
        date_str = datetime.now().strftime("%Y-%m-%d")  # Default to today
        if date_match:
            date_text = date_match.group(1)
            # Try to extract date from text like "Submitted on 14 Nov 2024"
            date_extract = re.search(r'(\d{1,2}\s+\w+\s+\d{4})', date_text)
            if date_extract:
                try:
                    date_obj = datetime.strptime(date_extract.group(1), "%d %b %Y")
                    date_str = date_obj.strftime("%Y-%m-%d")
                except ValueError:
                    pass
        
        metadata = {
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "date": date_str,
            "url": abs_url,
            "arxiv_id": arxiv_id
        }
        
        print(f"✅ Extracted metadata for: {title}")
        return metadata
        
    except Exception as e:
        print(f"❌ Error fetching metadata: {str(e)}")
        return {
            "title": "Unknown Title",
            "authors": [],
            "abstract": "",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "url": abs_url,
            "arxiv_id": arxiv_id
        }


def process_paper_by_url(url: str) -> bool:
    """Process a paper by URL: download, extract, structure, and store.
    
    Args:
        url: The paper URL
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n🔄 Processing paper: {url}")
    
    # Initialize Qdrant client
    try:
        client = init_qdrant_client()
        ensure_collection_exists(client)
    except Exception as e:
        print(f"❌ Error initializing Qdrant: {str(e)}")
        return False
    
    # Check if paper already exists
    existing_paper = get_paper_by_url(client, url)
    if existing_paper:
        print(f"✅ Paper {url} already exists in database")
        print(f"📖 Title: {existing_paper.get('title', 'Unknown')}")
        print(f"📅 Processed: {existing_paper.get('processed_date', 'Unknown')}")
        
        # Show conversation history if any
        conversations = existing_paper.get('past_conversations', [])
        if conversations:
            print(f"💬 Has {len(conversations)} past conversation(s)")
            for i, conv in enumerate(conversations[-3:], 1):  # Show last 3
                print(f"  {i}. {conv.get('timestamp', 'Unknown time')}: {conv.get('user_prompt', 'No prompt')[:50]}...")
        else:
            print("💬 No past conversations")
        
        return True
    
    # Extract arXiv ID from URL for downloading
    arxiv_id = extract_arxiv_id(url)
    if not arxiv_id:
        print(f"❌ Could not extract arXiv ID from URL: {url}")
        return False


    # Get paper metadata
    metadata = get_paper_metadata(arxiv_id)
    
    # Download PDF
    pdf_content = download_pdf(arxiv_id)
    if not pdf_content:
        return False
    
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_content)
    if not text:
        return False
    
    # Check rate limit status before making API calls
    print("\n📊 Checking rate limit status...")
    check_rate_limit_status()
    
    # Structure the paper content using LLM with cost estimation
    print("\n🤖 Structuring paper content with AI...")
    structured_content, tokens_used, cost = extract_paper_structure_with_cost(text, OPENROUTER_API_KEY)
    
    if "error" in structured_content:
        print(f"❌ Error structuring content: {structured_content['error']}")
        return False
    
    print(f"✅ Structured content using {tokens_used} tokens")
    print(f"💰 Estimated cost: {format_cost(cost)}")
    
    # Merge metadata with structured content
    paper_data = {
        **metadata,
        "sections": structured_content.get("sections", {}),
        "past_conversations": []
    }
    
    # Override title and authors from LLM if available and better
    if structured_content.get("title") and len(structured_content["title"]) > len(metadata["title"]):
        paper_data["title"] = structured_content["title"]
    
    if structured_content.get("authors") and len(structured_content["authors"]) > len(metadata["authors"]):
        paper_data["authors"] = structured_content["authors"]
    
    # Store in Qdrant
    print("\n💾 Storing in Qdrant database...")
    success = insert_structured_paper(client, paper_data)
    
    if success:
        print(f"\n🎉 Successfully processed paper: {paper_data['title']}")
        print(f"📄 Sections extracted: {', '.join(paper_data['sections'].keys())}")
        print(f"🔗 URL: {paper_data['url']}")
        print(f"👥 Authors: {', '.join(paper_data['authors'])}")
        return True
    else:
        print(f"\n❌ Failed to store paper in database")
        return False


def process_paper(arxiv_id: str) -> bool:
    """Process a paper: download, extract, structure, and store (deprecated - use process_paper_by_url instead).
    
    Args:
        arxiv_id: The arXiv ID to process
        
    Returns:
        True if successful, False otherwise
    """
    # Convert to URL and use the URL-based function
    url = f"https://arxiv.org/abs/{arxiv_id}"
    return process_paper_by_url(url)


def main():
    """Main function to handle command line arguments and process papers."""
    parser = argparse.ArgumentParser(
        description="Download and process ArXiv papers for analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python paper_read.py --url https://arxiv.org/abs/2411.14133
  python paper_read.py --url https://arxiv.org/pdf/2411.14133.pdf
  python paper_read.py --url 2411.14133
        """
    )
    
    parser.add_argument(
        "--url",
        required=True,
        help="ArXiv paper URL or ID (e.g., https://arxiv.org/abs/2411.14133 or 2411.14133)"
    )
    
    args = parser.parse_args()
    
    # Check for required environment variables
    if not OPENROUTER_API_KEY:
        print("❌ OPENROUTER_API_KEY is not set. Please update your .env file.")
        exit(1)
    
    # Normalize URL format
    arxiv_id = extract_arxiv_id(args.url)
    if not arxiv_id:
        print("❌ Invalid arXiv URL or ID format")
        exit(1)
    
    # Convert to standard URL format
    paper_url = f"https://arxiv.org/abs/{arxiv_id}"
    
    print(f"🚀 Paper Reader - Processing {paper_url}")
    
    # Process the paper
    success = process_paper_by_url(paper_url)
    
    if success:
        print(f"\n✅ Paper processing completed successfully!")
        print(f"💡 You can now use this paper for analysis and conversations.")
    else:
        print(f"\n❌ Paper processing failed!")
        exit(1)


if __name__ == "__main__":
    main()
