# Paper Digest - AI Security Research Tool

This is a comprehensive tool for discovering, analyzing, and managing AI security research papers. It uses OpenRouter for AI processing and Qdrant for vector storage.

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

### Process ArXiv Papers

```bash
python store_paper.py --url https://arxiv.org/abs/2411.14133
```

This will:

1. Download the PDF from ArXiv
2. Extract and structure the paper content into sections
3. **Display estimated cost** for AI processing operations
4. Store the structured content in Qdrant for future analysis

**Cost Estimation**: The script shows estimated costs for each AI operation:
- **Free models** (gpt-4.1-nano): $0.00
- **Cost-effective** (gpt-4.1-mini): ~$0.004 per paper
- **High-quality** (gpt-4o): ~$0.068 per paper

## Integration with Existing System

This tool integrates with:

- **OpenRouter**: For AI processing and relevance assessment
- **ArXiv**: For preprint discovery
- **Qdrant**: For vector storage and retrieval

## Files Overview

- `update_rss.py`: Update RSS feed from ArXiv with two-stage assessment
- `store_paper.py`: Download and process individual ArXiv papers
- `utils/llm.py`: AI processing and rate limiting
- `utils/qdrant.py`: Vector database operations
