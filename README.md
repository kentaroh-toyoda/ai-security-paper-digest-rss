# Paper Digest - AI Security Research Tool

This is a comprehensive tool for discovering, analyzing, and managing AI security research papers. It uses OpenRouter for AI processing and Qdrant for vector storage.

## RSS Feed

Subscribe to the AI Security Paper Digest RSS feed: [https://kentaroh-toyoda.github.io/ai-security-paper-digest-rss/rss.xml](https://kentaroh-toyoda.github.io/ai-security-paper-digest-rss/rss.xml)

## Features

- **Paper Discovery**: Fetch and process papers from ArXiv
- **Two-Stage Assessment**: Cost-efficient filtering with a quick assessment followed by detailed analysis
- **Intelligent Filtering**: AI-powered relevance assessment and tagging
- **Vector Storage**: Store and search papers using Qdrant
- **RSS Feed Generation**: Generate RSS feeds of relevant papers
- **Cost Optimization**: Efficient use of AI models to minimize costs

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
RSS_FEED_URL=your_rss_feed_url
QUICK_ASSESSMENT_MODEL=openai/gpt-4.1-nano  # Model for initial quick filtering
DETAILED_ASSESSMENT_MODEL=openai/gpt-4.1-mini  # Model for detailed analysis
TEMPERATURE=0.1              # Optional: specify the temperature (0.0 to 1.0)
```

## Usage

### Update RSS Feed from ArXiv

```bash
python update_rss.py
```

This will:

1. Fetch recent papers from ArXiv AI feeds
2. Perform a quick initial assessment using a cost-efficient model
3. Perform detailed assessment on promising papers using a more powerful model
4. Store relevant papers in Qdrant
5. Generate an RSS feed (`rss.xml`)

**Cost Optimization**: The two-stage assessment process significantly reduces costs:
- **Initial Filtering**: Uses a free or very low-cost model (e.g., gpt-4.1-nano)
- **Detailed Analysis**: Only papers that pass initial filtering are processed with a more expensive model
- **Cost Savings**: Detailed cost breakdown is provided in the output

## Integration with Existing System

This tool integrates with:

- **OpenRouter**: For AI processing and relevance assessment
- **ArXiv**: For preprint discovery
- **Qdrant**: For vector storage and retrieval

## Files Overview

- `update_rss.py`: Update RSS feed from ArXiv with two-stage assessment
- `utils/llm.py`: AI processing and rate limiting
- `utils/qdrant.py`: Vector database operations
