# Paper Digest - AI Security Research Tool

This is a comprehensive tool for discovering, analyzing, and managing AI security research papers. It uses OpenRouter for AI processing and Qdrant for vector storage.

## Features

- **Paper Discovery**: Search OpenAlex and ArXiv for AI security papers
- **Intelligent Filtering**: AI-powered relevance assessment and tagging
- **Rate Limiting**: Built-in rate limiting for OpenRouter's free tier (10 requests/10 seconds)
- **Vector Storage**: Store and search papers using Qdrant
- **RSS Feed Generation**: Generate RSS feeds of relevant papers
- **Quality Assessment**: AI-powered paper quality evaluation

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
OPENALEX_EMAIL=your_email_for_openalex
RSS_FEED_URL=your_rss_feed_url
AI_MODEL=openai/gpt-4o-mini  # Optional: specify the model to use
TEMPERATURE=0.1              # Optional: specify the temperature (0.0 to 1.0)
```

## Rate Limiting

This tool includes comprehensive rate limiting for OpenRouter's free tier:

- **Automatic Throttling**: All API calls are automatically rate-limited
- **10 requests per 10 seconds**: Respects OpenRouter's free tier limits
- **Smart Waiting**: Automatically waits when rate limits are reached
- **Status Monitoring**: Check your current rate limit status

### Rate Limiting Utilities

```bash
# Check current rate limit status
python check_rate_limit.py

# Test rate limiting functionality
python test_rate_limiting.py
```

## Usage

### Search for Papers

```bash
python search_papers.py
```

This will:

1. Prompt for a search topic (default: "LLM red teaming")
2. Generate related keywords
3. Search OpenAlex for relevant papers
4. Assess relevance using AI
5. Store relevant papers in Qdrant

### Update RSS Feed from ArXiv

```bash
python update_rss.py
```

This will:

1. Fetch recent papers from ArXiv AI feeds
2. Assess relevance using AI
3. Store relevant papers in Qdrant
4. Generate an RSS feed (`rss.xml`)

### Chat with Papers

```bash
python db_chat.py
```

This will:

1. Start an interactive chat session
2. Use RAG to find relevant papers
3. Answer questions based on stored papers

### Check Rate Limit Status

```bash
python check_rate_limit.py
```

Shows your current rate limit usage without making API calls.

## Rate Limiting Details

The rate limiting system includes:

- **RateLimiter Class**: Tracks API calls in a sliding window
- **Automatic Waiting**: Pauses execution when limits are reached
- **Exponential Backoff**: Retries with increasing delays on failures
- **Status Monitoring**: Real-time rate limit status
- **Thread Safety**: Safe for concurrent operations

### Configuration

Rate limiting is configured for OpenRouter's free tier:

- 10 requests per 10-second window
- 10% safety margin
- Automatic retry with exponential backoff

## Integration with Existing System

This tool integrates with:

- **OpenRouter**: For AI processing and relevance assessment
- **OpenAlex**: For paper discovery
- **ArXiv**: For preprint discovery
- **Qdrant**: For vector storage and retrieval

## Files Overview

- `search_papers.py`: Search OpenAlex for papers
- `update_rss.py`: Update RSS feed from ArXiv
- `db_chat.py`: Chat with stored papers
- `check_rate_limit.py`: Check rate limit status
- `test_rate_limiting.py`: Test rate limiting functionality
- `utils/llm.py`: AI processing and rate limiting
- `utils/qdrant.py`: Vector database operations
