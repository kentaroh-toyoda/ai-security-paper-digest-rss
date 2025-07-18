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
- **20 requests per minute**: Respects OpenRouter's free tier limits
- **Daily Limits**: 50 requests/day (free users) or 1000 requests/day (paid users)
- **Free Model Detection**: Automatically detects `:free` model variants
- **Process Termination**: Exits with error when rate limits are reached
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

### Process ArXiv Papers

```bash
python store_paper.py --url https://arxiv.org/abs/2411.14133
```

This will:

1. Download the PDF from ArXiv
2. Extract and structure the paper content into sections
3. **Display estimated cost** for AI processing operations
4. Store the structured content in Qdrant for future analysis
5. Enable resumable conversations about the paper

**Cost Estimation**: The script now shows estimated costs for each AI operation:
- **Free models** (gpt-4.1-nano): $0.00
- **Cost-effective** (gpt-4o-mini): ~$0.004 per paper
- **High-quality** (gpt-4o): ~$0.068 per paper

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

- **RateLimiter Class**: Tracks API calls in a sliding window (minute-based)
- **DailyRateLimiter Class**: Tracks daily API call limits
- **Free Model Detection**: Automatically detects `:free` model variants
- **Process Termination**: Exits with error when limits are reached
- **Exponential Backoff**: Retries with increasing delays on other failures
- **Status Monitoring**: Real-time rate limit status for both minute and daily limits
- **Thread Safety**: Safe for concurrent operations

### Configuration

Rate limiting is configured for OpenRouter's free tier:

- 20 requests per 60-second window
- 50 requests per day (free users)
- 1000 requests per day (paid users with 10+ credits)
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
- `store_paper.py`: Download and process individual ArXiv papers
- `db_chat.py`: Chat with stored papers
- `check_rate_limit.py`: Check rate limit status
- `test_rate_limiting.py`: Test rate limiting functionality
- `utils/llm.py`: AI processing and rate limiting
- `utils/qdrant.py`: Vector database operations
