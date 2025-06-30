# utils/llm.py

import os
import json
import requests
import time
import threading
from collections import deque
from dotenv import load_dotenv
from typing import Tuple, Dict, Any
from datetime import datetime, timezone, timedelta
import re

# Load environment variables
load_dotenv()

# Default configuration
DEFAULT_MODEL = "openai/gpt-4.1"
DEFAULT_MINI_MODEL = "openai/gpt-4.1-mini"
DEFAULT_TEMPERATURE = 0.1

# OpenRouter configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Rate limiting configuration for free tier
FREE_TIER_REQUESTS_PER_WINDOW = 20
FREE_TIER_WINDOW_SECONDS = 60
FREE_TIER_DAILY_LIMIT = 50  # Default daily limit for free tier
FREE_TIER_DAILY_LIMIT_PAID = 1000  # Daily limit if you've purchased 10+ credits
SAFETY_MARGIN = 0.1  # 10% safety margin

# OpenRouter pricing (per 1M tokens) - Updated as of Dec 2024
# These are approximate rates and may change
OPENROUTER_PRICING = {
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai/gpt-4.1": {"input": 30.00, "output": 60.00},
    "openai/gpt-4.1-mini": {"input": 3.00, "output": 12.00},
    "openai/gpt-4.1-nano": {"input": 0.00, "output": 0.00},  # Free model
    "anthropic/claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
    "anthropic/claude-3-haiku": {"input": 0.25, "output": 1.25},
    "meta-llama/llama-3.1-8b-instruct": {"input": 0.18, "output": 0.18},
    "meta-llama/llama-3.1-70b-instruct": {"input": 0.88, "output": 0.88},
    "mistralai/mistral-7b-instruct": {"input": 0.20, "output": 0.20},
    "google/gemini-pro": {"input": 0.50, "output": 1.50},
}


class DailyRateLimiter:
    """Rate limiter for daily API call limits."""

    def __init__(self, daily_limit=FREE_TIER_DAILY_LIMIT):
        self.daily_limit = daily_limit
        self.request_dates = deque()
        self.lock = threading.Lock()

    def check_and_record(self):
        """Check daily limit and record the request."""
        with self.lock:
            now = datetime.now()
            today = now.date()

            # Remove requests from previous days
            while self.request_dates and self.request_dates[0].date() < today:
                self.request_dates.popleft()

            # Check if we're at the daily limit
            if len(self.request_dates) >= self.daily_limit:
                # Calculate time until midnight UTC (when the daily limit resets)
                midnight_utc = datetime.combine(today + timedelta(days=1), datetime.min.time()).replace(tzinfo=timezone.utc)
                now_utc = now.replace(tzinfo=timezone.utc)
                wait_time = (midnight_utc - now_utc).total_seconds()
                
                print(f"⏱️ Daily rate limit reached. Waiting until midnight UTC ({wait_time:.1f} seconds)...")
                print(f"💡 You've reached the daily limit ({self.daily_limit} requests/day).")
                
                # If wait time is too long (more than 1 hour), exit instead of waiting
                if wait_time > 3600:
                    print(f"❌ Wait time too long ({wait_time:.1f} seconds). Terminating process.")
                    exit(1)
                
                # Release the lock while waiting
                self.lock.release()
                time.sleep(wait_time)
                # Reacquire the lock
                self.lock.acquire()
                
                # After waiting, recalculate and clean up old requests
                now = datetime.now()
                today = now.date()
                while self.request_dates and self.request_dates[0].date() < today:
                    self.request_dates.popleft()

            # Add current request
            self.request_dates.append(now)

    def get_status(self):
        """Get current daily rate limit status."""
        with self.lock:
            now = datetime.now()
            today = now.date()

            # Remove old requests
            while self.request_dates and self.request_dates[0].date() < today:
                self.request_dates.popleft()

            return {
                "requests_today": len(self.request_dates),
                "daily_limit": self.daily_limit,
                "remaining_today": max(0, self.daily_limit - len(self.request_dates))
            }


class RateLimiter:
    """Rate limiter for OpenRouter API calls to respect free tier limits."""

    def __init__(self, requests_per_window=FREE_TIER_REQUESTS_PER_WINDOW,
                 window_seconds=FREE_TIER_WINDOW_SECONDS):
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.request_times = deque()
        self.lock = threading.Lock()

    def wait_if_needed(self):
        """Wait if necessary to respect rate limits."""
        with self.lock:
            now = time.time()

            # Remove old requests outside the window
            while self.request_times and now - self.request_times[0] >= self.window_seconds:
                self.request_times.popleft()

            # Check if we're at the limit
            if len(self.request_times) >= self.requests_per_window:
                # Calculate how long to wait
                oldest_request = self.request_times[0]
                wait_time = self.window_seconds - \
                    (now - oldest_request) + SAFETY_MARGIN

                if wait_time > 0:
                    print(f"⏱️ Rate limit reached. Waiting {wait_time:.1f} seconds...")
                    # Release the lock while waiting
                    self.lock.release()
                    time.sleep(wait_time)
                    # Reacquire the lock
                    self.lock.acquire()
                    
                    # After waiting, recalculate and clean up old requests
                    now = time.time()
                    while self.request_times and now - self.request_times[0] >= self.window_seconds:
                        self.request_times.popleft()

            # Add current request
            self.request_times.append(now)

    def get_status(self):
        """Get current rate limit status."""
        with self.lock:
            now = time.time()

            # Remove old requests
            while self.request_times and now - self.request_times[0] >= self.window_seconds:
                self.request_times.popleft()

            return {
                "requests_in_window": len(self.request_times),
                "max_requests": self.requests_per_window,
                "window_seconds": self.window_seconds,
                "time_until_reset": max(0, self.window_seconds - (now - self.request_times[0])) if self.request_times else 0
            }


# Global rate limiter instances
_rate_limiter = RateLimiter()
_daily_limiter = DailyRateLimiter()

# Simple in-memory cache for LLM responses
_llm_response_cache = {}


def get_cache_key(text: str, model: str, function_name: str) -> str:
    """Generate a cache key for LLM responses.
    
    Args:
        text: The input text
        model: The model name
        function_name: The function name (to avoid collisions between different functions)
        
    Returns:
        str: A cache key
    """
    # Use the first 100 chars of text + model + function name as the key
    # This is a simple approach - for production, consider using a hash function
    return f"{function_name}:{model}:{text[:100]}"


def cached_llm_call(func):
    """Decorator to cache LLM responses."""
    def wrapper(*args, **kwargs):
        global _llm_response_cache
        
        # Extract relevant parameters for cache key
        text = args[0] if len(args) > 0 else kwargs.get('text', '')
        model = kwargs.get('model', args[3] if len(args) > 3 else None)
        
        # Generate cache key
        cache_key = get_cache_key(text, model, func.__name__)
        
        # Check if we have a cached response
        if cache_key in _llm_response_cache:
            print(f"✅ Using cached LLM response for {func.__name__}")
            return _llm_response_cache[cache_key]
        
        # If not in cache, call the function
        result = func(*args, **kwargs)
        
        # Cache the result
        _llm_response_cache[cache_key] = result
        
        return result
    
    return wrapper


def get_rate_limiter():
    """Get the global rate limiter instance."""
    return _rate_limiter


def get_daily_limiter():
    """Get the global daily rate limiter instance."""
    return _daily_limiter


def is_free_model(model_name):
    """Check if a model is a free model variant."""
    return model_name and model_name.endswith(':free')

def is_exempt_from_rate_limit(model_name):
    """Check if a model should be exempt from the rate limit counter.
    
    Some models like gpt-4.1-nano are not counted toward the OpenRouter free tier limit.
    """
    exempt_models = [
        "openai/gpt-4.1-nano",
        # Add other models that don't count toward the rate limit
    ]
    return model_name in exempt_models


def update_daily_limit_for_paid_user():
    """Update daily limit for users who have purchased 10+ credits."""
    global _daily_limiter
    _daily_limiter = DailyRateLimiter(FREE_TIER_DAILY_LIMIT_PAID)
    print(
        f"✅ Updated daily limit to {FREE_TIER_DAILY_LIMIT_PAID} requests/day (paid user)")


def make_rate_limited_request(url, headers, payload, max_retries=3, retry_delay=1):
    """Make a rate-limited API request with automatic retries."""
    rate_limiter = get_rate_limiter()
    daily_limiter = get_daily_limiter()

    # Check if this is a free model request
    model_name = payload.get("model", "")
    is_free = is_free_model(model_name)
    is_exempt = is_exempt_from_rate_limit(model_name)

    for attempt in range(max_retries):
        try:
            # Check daily limit for free models
            if is_free:
                daily_limiter.check_and_record()

            # Wait if necessary to respect rate limits
            # Skip rate limiting for exempt models like gpt-4.1-nano
            if not is_exempt:
                rate_limiter.wait_if_needed()

            # Make the request
            response = requests.post(
                url, headers=headers, json=payload, timeout=30)

            # Handle rate limit errors - wait and retry
            if response.status_code == 429:
                # Get retry-after header if available, otherwise use default wait time
                retry_after = int(response.headers.get('retry-after', 60))
                print(f"⏱️ Rate limit hit on attempt {attempt + 1}. Waiting {retry_after} seconds...")
                
                if is_free:
                    print(f"💡 You've reached the OpenRouter free tier limit (20 requests/minute).")
                else:
                    print(f"💡 You've reached the OpenRouter rate limit.")
                
                time.sleep(retry_after)
                continue

            # Handle other errors
            if response.status_code != 200:
                print(
                    f"❌ API request failed with status {response.status_code}: {response.text}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    response.raise_for_status()

            return response

        except requests.exceptions.RequestException as e:
            print(f"❌ Request error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise

    raise Exception(f"Failed to make API request after {max_retries} attempts")


def check_rate_limit_status():
    """Check and display current rate limit status."""
    rate_limiter = get_rate_limiter()
    daily_limiter = get_daily_limiter()

    minute_status = rate_limiter.get_status()
    daily_status = daily_limiter.get_status()

    print(f"📊 Rate Limit Status:")
    print(
        f"  Minute Window: {minute_status['requests_in_window']}/{minute_status['max_requests']} requests")
    print(
        f"  Time until minute window resets: {minute_status['time_until_reset']:.2f} seconds")
    print(
        f"  Daily Usage: {daily_status['requests_today']}/{daily_status['daily_limit']} requests")
    print(f"  Remaining today: {daily_status['remaining_today']} requests")

    # Minute window analysis
    if minute_status['requests_in_window'] >= minute_status['max_requests']:
        print("  ⚠️ Minute rate limit window is full!")
    elif minute_status['requests_in_window'] >= minute_status['max_requests'] * 0.8:
        print("  ⚠️ Approaching minute rate limit!")
    else:
        print("  ✅ Minute rate limit window has capacity")

    # Daily limit analysis
    if daily_status['requests_today'] >= daily_status['daily_limit']:
        print("  ⚠️ Daily limit reached!")
    elif daily_status['requests_today'] >= daily_status['daily_limit'] * 0.8:
        print("  ⚠️ Approaching daily limit!")
    else:
        print("  ✅ Daily limit has capacity")

    return minute_status, daily_status


def get_llm_config():
    """Get LLM configuration from environment variables with defaults."""
    return {
        "model": os.getenv("AI_MODEL", DEFAULT_MODEL),
        "temperature": float(os.getenv("TEMPERATURE", DEFAULT_TEMPERATURE))
    }


def create_openrouter_client(api_key: str):
    """Create an OpenRouter client using requests."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://paper-digest.com",  # Replace with your domain
        "X-Title": "Paper Digest"  # Replace with your app name
    }
    return headers


@cached_llm_call
def generate_search_keywords(topic: str, api_key: str) -> str:
    """Generate optimized search keywords using OpenRouter.

    Args:
        topic: The topic to generate search keywords for
        api_key: OpenRouter API key

    Returns:
        str: Optimized search query string
    """
    headers = create_openrouter_client(api_key)

    # Optimized prompt to reduce token usage
    system_prompt = """Generate optimized search keywords for academic papers on OpenAlex.

Create a search query with:
- OR for related terms
- AND for required concepts
- Quotes for exact phrases
- NOT to exclude irrelevant areas

Example: "LLM red teaming" → "large language model" AND ("red teaming" OR "jailbreaking") AND (security OR safety) NOT (medical)

Respond with ONLY the search query."""

    payload = {
        "model": DEFAULT_MINI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": topic}
        ],
        "temperature": DEFAULT_TEMPERATURE
    }

    try:
        response = make_rate_limited_request(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            payload=payload
        )
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"❌ Error generating search keywords: {str(e)}")
        return topic


def clean_and_extract_json(response_text: str) -> dict:
    """Clean response text and extract JSON, handling thinking tokens and other formatting."""

    # Remove thinking tokens and other common formatting
    cleaned = response_text

    # Remove thinking tokens (◁think▷ and ◁/think▷)
    cleaned = re.sub(r'◁think▷.*?◁/think▷', '', cleaned, flags=re.DOTALL)

    # Remove other common thinking/reasoning tokens
    thinking_patterns = [
        r'<think>.*?</think>',
        r'<reasoning>.*?</reasoning>',
        r'<analysis>.*?</analysis>',
        r'<thought>.*?</thought>',
        r'<step>.*?</step>',
        r'<process>.*?</process>'
    ]

    for pattern in thinking_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL)

    # Remove leading/trailing whitespace and newlines
    cleaned = cleaned.strip()

    # Try to parse as JSON first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Look for JSON object in the cleaned text
    json_match = re.search(
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
    if json_match:
        try:
            json_str = json_match.group(0)
            result = json.loads(json_str)
            return result
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse extracted JSON: {json_str}")
            print(f"❌ JSON parsing error: {e}")
            print(f"❌ Original response: {response_text[:500]}...")
            exit(1)

    # Try a more aggressive JSON extraction
    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if json_match:
        try:
            json_str = json_match.group(0)
            result = json.loads(json_str)
            return result
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse extracted JSON: {json_str}")
            print(f"❌ JSON parsing error: {e}")
            print(f"❌ Original response: {response_text[:500]}...")
            exit(1)

    # Fallback: try to evaluate as Python dict (less safe but sometimes works)
    try:
        # Remove any remaining non-JSON text
        json_only = re.sub(r'^[^{]*', '', cleaned)
        json_only = re.sub(r'[^}]*$', '', json_only)
        result = eval(json_only)
        return result
    except Exception as e:
        print(f"❌ Failed to parse extracted JSON: {json_only}")
        print(f"❌ Evaluation error: {e}")
        print(f"❌ Original response: {response_text[:500]}...")
        exit(1)

    # If all else fails, exit with error
    print(
        f"❌ Could not extract valid JSON from response: {response_text[:500]}...")
    print(f"❌ Cleaned response: {cleaned[:500]}...")
    exit(1)


@cached_llm_call
def assess_relevance_and_tags(text: str, api_key: str, temperature: float = 0.1, model: str = "openai/gpt-4o") -> Tuple[Dict[str, Any], int]:
    """Assess if a paper is relevant and extract tags using OpenRouter."""
    headers = create_openrouter_client(api_key)

    # Optimized prompt to reduce token usage while maintaining essential instructions
    system_prompt = """Assess if this paper directly addresses AI security, safety, or red teaming.

Relevant topics: LLM red teaming, jailbreaking, prompt injection, adversarial prompting, model extraction, 
data poisoning, privacy attacks, alignment, robustness, safety evaluation, security standards.

NOT relevant: General AI/ML papers, AI applications without security focus, AI ethics without security aspects.

If relevant (score ≥3/5):
- Summary (2-4 bullet points)
- 3-5 tags
- Relevance score (1-5)
- Brief reason for score
- Paper type (Research/Survey/Benchmarking/Position/Other)
- Modalities (Text/Image/Video/Audio/Multimodal/Other)

If not relevant: {"relevant": false}

Output JSON: {"relevant": true/false, "summary": [...], "tags": [...], "relevance_score": 1-5, 
"reason": "...", "paper_type": "...", "modalities": [...]}"""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        "temperature": temperature
    }

    try:
        response = make_rate_limited_request(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            payload=payload
        )
        result_data = response.json()
        result = result_data["choices"][0]["message"]["content"]
        
        # Extract token usage
        usage = result_data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
        
        # Calculate cost
        cost = calculate_cost(input_tokens, output_tokens, model)
        
        # Use the improved JSON extraction function
        result_dict = clean_and_extract_json(result)
        
        # Add cost information to the result for display
        if cost > 0:
            print(f"💰 Relevance assessment cost: {format_cost(cost)}")
        
        return result_dict, total_tokens

    except Exception as e:
        print(f"❌ Error calling OpenRouter API: {str(e)}")
        return {"relevant": False}, 0


def assess_relevance_and_tags_with_cost(text: str, api_key: str, temperature: float = 0.1, model: str = "openai/gpt-4o") -> Tuple[Dict[str, Any], int, float]:
    """Assess if a paper is relevant and extract tags with cost estimation."""
    headers = create_openrouter_client(api_key)

    # Optimized prompt to reduce token usage while maintaining essential instructions
    system_prompt = """Assess if this paper directly addresses AI security, safety, or red teaming.

Relevant topics: LLM red teaming, jailbreaking, prompt injection, adversarial prompting, model extraction, 
data poisoning, privacy attacks, alignment, robustness, safety evaluation, security standards.

NOT relevant: General AI/ML papers, AI applications without security focus, AI ethics without security aspects.

If relevant (score ≥3/5):
- Summary (2-4 bullet points)
- 3-5 tags
- Relevance score (1-5)
- Brief reason for score
- Paper type (Research/Survey/Benchmarking/Position/Other)
- Modalities (Text/Image/Video/Audio/Multimodal/Other)

If not relevant: {"relevant": false}

Output JSON: {"relevant": true/false, "summary": [...], "tags": [...], "relevance_score": 1-5, 
"reason": "...", "paper_type": "...", "modalities": [...]}"""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        "temperature": temperature
    }

    try:
        response = make_rate_limited_request(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            payload=payload
        )
        result_data = response.json()
        result = result_data["choices"][0]["message"]["content"]
        
        # Extract token usage
        usage = result_data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
        
        # Calculate cost
        cost = calculate_cost(input_tokens, output_tokens, model)

        # Use the improved JSON extraction function
        result_dict = clean_and_extract_json(result)
        return result_dict, total_tokens, cost

    except Exception as e:
        print(f"❌ Error calling OpenRouter API: {str(e)}")
        return {"relevant": False}, 0, 0.0


@cached_llm_call
def assess_paper_quality(metadata: dict, api_key: str, return_usage=False):
    """Assess paper quality using available metadata.

    Args:
        metadata: Dictionary containing paper metadata:
            - title: Paper title
            - abstract: Paper abstract
            - cited_by_count: Number of citations
            - publication_type: Type of publication
            - source: Publication source/venue
            - code_repository: URL to code repository if available
            - date: Publication date (YYYY-MM-DD)
        api_key: OpenRouter API key
        return_usage: Whether to return token usage information

    Returns:
        dict: Quality assessment results
    """
    headers = create_openrouter_client(api_key)
    config = get_llm_config()

    # Calculate paper age in months
    pub_date = datetime.strptime(metadata['date'], "%Y-%m-%d")
    now = datetime.now()
    age_months = (now.year - pub_date.year) * 12 + (now.month - pub_date.month)

    # Adjust citation context based on age
    citation_context = ""
    if age_months < 3:
        citation_context = "This is a very recent paper (less than 3 months old), so citation count is not yet meaningful."
    elif age_months < 6:
        citation_context = "This is a recent paper (3-6 months old), so citation count should be considered with caution."
    elif age_months < 12:
        citation_context = "This paper is 6-12 months old, so citation count is starting to become meaningful."
    else:
        citation_context = f"This paper is {age_months} months old, so citation count is a good indicator of impact."

    # Optimized prompt to reduce token usage
    system_prompt = """Evaluate this research paper on:
1. Clarity (1-5): How clear is the writing?
2. Novelty (1-5): How original is the work?
3. Significance (1-5): How impactful is it?
4. Try-worthiness: Worth implementing? (true/false)

Also extract any code repository links."""

    user_prompt = f"""Paper Metadata:
Title: {metadata['title']}
Abstract: {metadata['abstract']}
Publication Date: {metadata['date']}
Citations: {metadata['cited_by_count']} ({citation_context})
Publication Type: {metadata['publication_type']}
Source/Venue: {metadata['source']}
Code Repository: {metadata['code_repository']}

Please assess the paper and output JSON:
{{
  "clarity": score from 1 to 5,
  "novelty": score from 1 to 5,
  "significance": score from 1 to 5,
  "try_worthiness": true or false,
  "justification": "brief explanation of the scores",
  "code_repository": "GitHub URL or similar, if found"
}}"""

    payload = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": config["temperature"]
    }

    try:
        response = make_rate_limited_request(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            payload=payload
        )
        result_data = response.json()
        content = result_data["choices"][0]["message"]["content"]

        # Print the raw response for debugging
        print(f"\nOpenRouter Quality Assessment for {metadata['title']}:")
        print(content)

        # Use the improved JSON extraction function
        parsed = clean_and_extract_json(content)

        if return_usage:
            return parsed, result_data["usage"]["total_tokens"]
        return parsed

    except Exception as e:
        print(f"❌ Error calling OpenRouter API: {str(e)}")
        if return_usage:
            return {"error": str(e)}, 0
        return {"error": str(e)}


def check_rate_limit(api_key: str) -> bool:
    """Check if we're currently rate limited by making a simple API call.

    Args:
        api_key: OpenRouter API key

    Returns:
        bool: True if rate limited (429 error), False otherwise
    """
    headers = create_openrouter_client(api_key)

    # Make a minimal API call to check rate limit
    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "test"}
        ],
        "max_tokens": 1
    }

    try:
        response = make_rate_limited_request(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            payload=payload
        )

        print("✅ Rate limit check passed - API is available")
        return False

    except Exception as e:
        if "429" in str(e) or "rate limit" in str(e).lower():
            print("⚠️ Rate limit detected - API is currently rate limited")
            return True
        else:
            print(f"❌ Error checking rate limit: {str(e)}")
            return False


def check_and_update_daily_limit():
    """Check current daily limit and provide option to update for paid users."""
    daily_limiter = get_daily_limiter()
    status = daily_limiter.get_status()

    print(f"📊 Current Daily Limit: {status['daily_limit']} requests/day")
    print(f"📊 Current Usage: {status['requests_today']} requests today")
    print(f"📊 Remaining: {status['remaining_today']} requests today")

    if status['daily_limit'] == FREE_TIER_DAILY_LIMIT:
        print(f"\n💡 If you've purchased 10+ credits, you can increase your daily limit.")
        print(
            f"💡 Call update_daily_limit_for_paid_user() to set limit to {FREE_TIER_DAILY_LIMIT_PAID} requests/day")

    return status


def reset_daily_limiter():
    """Reset the daily limiter (useful for testing)."""
    global _daily_limiter
    _daily_limiter = DailyRateLimiter()
    print("✅ Daily limiter reset to default limits")




@cached_llm_call
def quick_assess_relevance(text: str, api_key: str, temperature: float = 0.1, model: str = "openai/gpt-4.1-nano") -> Tuple[bool, int]:
    """Quick assessment of paper relevance using a smaller, cheaper model.
    
    This function performs a fast initial screening to determine if a paper is potentially
    relevant to AI security, safety, or red teaming. It uses a smaller, cheaper model
    to reduce costs for the initial filtering stage.
    
    Args:
        text: The paper title and abstract
        api_key: OpenRouter API key
        temperature: Temperature for the model (default: 0.1)
        model: Model to use (default: mistralai/mistral-7b-instruct)
        
    Returns:
        Tuple containing:
        - Boolean indicating if the paper is potentially relevant
        - Number of tokens used
    """
    headers = create_openrouter_client(api_key)
    
    system_prompt = """Determine if this paper is potentially relevant to AI security, safety, or red teaming.
Key topics: LLM security, red teaming, jailbreaking, prompt injection, adversarial attacks, model extraction, 
data poisoning, privacy attacks, alignment, robustness, safety evaluation, security standards.

Respond with ONLY "yes" or "no"."""
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        "temperature": temperature
    }
    
    try:
        response = make_rate_limited_request(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            payload=payload
        )
        result_data = response.json()
        result = result_data["choices"][0]["message"]["content"].lower().strip()
        token_count = result_data["usage"]["total_tokens"]
        
        return "yes" in result, token_count
    except Exception as e:
        print(f"❌ Error in quick relevance assessment: {str(e)}")
        return False, 0


@cached_llm_call
def extract_paper_structure(text: str, api_key: str, temperature: float = 0.1, model: str = "openai/gpt-4o") -> Tuple[Dict[str, Any], int]:
    """Extract and structure paper content into organized sections.
    
    Args:
        text: The full paper text
        api_key: OpenRouter API key
        temperature: Temperature for the model (default: 0.1)
        model: Model to use (default: openai/gpt-4o)
        
    Returns:
        Tuple containing:
        - Dictionary with structured sections
        - Number of tokens used
    """
    headers = create_openrouter_client(api_key)
    
    system_prompt = """Extract and organize this academic paper into structured sections in markdown format.

Identify and extract the following sections if they exist:
- abstract: Paper abstract
- introduction: Introduction/background section
- methodology: Methods, approach, or experimental setup
- results: Results, findings, or evaluation
- discussion: Discussion or analysis of results
- conclusion: Conclusion or future work
- references: References/bibliography
- appendix: Any appendix content

For each section, provide the content in clean markdown format. If a section doesn't exist, omit it from the output.

Output as JSON:
{
  "title": "Extracted paper title",
  "authors": ["Author 1", "Author 2"],
  "sections": {
    "abstract": "Abstract content in markdown...",
    "introduction": "Introduction content in markdown...",
    "methodology": "Methodology content in markdown...",
    "results": "Results content in markdown...",
    "discussion": "Discussion content in markdown...",
    "conclusion": "Conclusion content in markdown...",
    "references": "References in markdown...",
    "appendix": "Appendix content in markdown..."
  }
}"""
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please extract and structure this paper:\n\n{text}"}
        ],
        "temperature": temperature
    }
    
    try:
        response = make_rate_limited_request(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            payload=payload
        )
        result_data = response.json()
        result = result_data["choices"][0]["message"]["content"]
        token_count = result_data["usage"]["total_tokens"]
        
        # Use the improved JSON extraction function
        result_dict = clean_and_extract_json(result)
        return result_dict, token_count
        
    except Exception as e:
        print(f"❌ Error extracting paper structure: {str(e)}")
        return {"error": str(e)}, 0


@cached_llm_call
@cached_llm_call
def analyze_paper_content(structured_content: Dict[str, Any], api_key: str, temperature: float = 0.1, model: str = "openai/gpt-4o") -> Tuple[str, int]:
    """Analyze structured paper content and provide a comprehensive analysis.
    
    Args:
        structured_content: Dictionary containing structured paper sections
        api_key: OpenRouter API key
        temperature: Temperature for the model (default: 0.1)
        model: Model to use (default: openai/gpt-4o)
        
    Returns:
        Tuple containing:
        - Analysis text
        - Number of tokens used
    """
    headers = create_openrouter_client(api_key)
    
    # Create a summary of the paper content for analysis
    paper_summary = f"""Title: {structured_content.get('title', 'Unknown')}
Authors: {', '.join(structured_content.get('authors', []))}

Abstract: {structured_content.get('sections', {}).get('abstract', 'No abstract available')}

Introduction: {structured_content.get('sections', {}).get('introduction', 'No introduction available')[:1000]}...

Methodology: {structured_content.get('sections', {}).get('methodology', 'No methodology available')[:1000]}...

Results: {structured_content.get('sections', {}).get('results', 'No results available')[:1000]}...

Conclusion: {structured_content.get('sections', {}).get('conclusion', 'No conclusion available')[:1000]}..."""
    
    system_prompt = """Analyze this academic paper and provide a comprehensive analysis covering:

1. **Main Contributions**: What are the key contributions of this work?
2. **Methodology**: What approach or methods did the authors use?
3. **Key Findings**: What are the most important results or findings?
4. **Significance**: Why is this work important? What impact might it have?
5. **Limitations**: What are the limitations or potential weaknesses?
6. **Future Work**: What future research directions does this suggest?

Provide a clear, structured analysis that would be helpful for someone trying to understand the paper's value and implications."""
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": paper_summary}
        ],
        "temperature": temperature
    }
    
    try:
        response = make_rate_limited_request(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            payload=payload
        )
        result_data = response.json()
        analysis = result_data["choices"][0]["message"]["content"]
        token_count = result_data["usage"]["total_tokens"]
        
        return analysis, token_count
        
    except Exception as e:
        print(f"❌ Error analyzing paper content: {str(e)}")
        return f"Error analyzing paper: {str(e)}", 0


def calculate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Calculate the estimated cost for an API call.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model name
        
    Returns:
        Estimated cost in USD
    """
    if model not in OPENROUTER_PRICING:
        # Default to GPT-4o pricing if model not found
        pricing = OPENROUTER_PRICING.get("openai/gpt-4o", {"input": 2.50, "output": 10.00})
    else:
        pricing = OPENROUTER_PRICING[model]
    
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    
    return input_cost + output_cost


def format_cost(cost: float) -> str:
    """Format cost for display.
    
    Args:
        cost: Cost in USD
        
    Returns:
        Formatted cost string
    """
    if cost == 0:
        return "Free"
    elif cost < 0.001:
        return f"${cost:.6f}"
    elif cost < 0.01:
        return f"${cost:.4f}"
    else:
        return f"${cost:.3f}"


def extract_paper_structure_with_cost(text: str, api_key: str, temperature: float = 0.1, model: str = "openai/gpt-4o") -> Tuple[Dict[str, Any], int, float]:
    """Extract and structure paper content with cost estimation.
    
    Args:
        text: The full paper text
        api_key: OpenRouter API key
        temperature: Temperature for the model (default: 0.1)
        model: Model to use (default: openai/gpt-4o)
        
    Returns:
        Tuple containing:
        - Dictionary with structured sections
        - Number of tokens used
        - Estimated cost in USD
    """
    headers = create_openrouter_client(api_key)
    
    system_prompt = """Extract and organize this academic paper into structured sections in markdown format.

Identify and extract the following sections if they exist:
- abstract: Paper abstract
- introduction: Introduction/background section
- methodology: Methods, approach, or experimental setup
- results: Results, findings, or evaluation
- discussion: Discussion or analysis of results
- conclusion: Conclusion or future work
- references: References/bibliography
- appendix: Any appendix content

For each section, provide the content in clean markdown format. If a section doesn't exist, omit it from the output.

Output as JSON:
{
  "title": "Extracted paper title",
  "authors": ["Author 1", "Author 2"],
  "sections": {
    "abstract": "Abstract content in markdown...",
    "introduction": "Introduction content in markdown...",
    "methodology": "Methodology content in markdown...",
    "results": "Results content in markdown...",
    "discussion": "Discussion content in markdown...",
    "conclusion": "Conclusion content in markdown...",
    "references": "References in markdown...",
    "appendix": "Appendix content in markdown..."
  }
}"""
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please extract and structure this paper:\n\n{text}"}
        ],
        "temperature": temperature
    }
    
    try:
        response = make_rate_limited_request(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            payload=payload
        )
        result_data = response.json()
        result = result_data["choices"][0]["message"]["content"]
        
        # Extract token usage
        usage = result_data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
        
        # Calculate cost
        cost = calculate_cost(input_tokens, output_tokens, model)
        
        # Use the improved JSON extraction function
        result_dict = clean_and_extract_json(result)
        return result_dict, total_tokens, cost
        
    except Exception as e:
        print(f"❌ Error extracting paper structure: {str(e)}")
        return {"error": str(e)}, 0, 0.0
