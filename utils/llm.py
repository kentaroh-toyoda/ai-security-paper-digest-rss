# utils/llm.py

import os
import json
import requests
import time
import threading
from collections import deque
from dotenv import load_dotenv
from typing import Tuple, Dict, Any, List
from datetime import datetime, timezone, timedelta
import re
from sentence_transformers import SentenceTransformer

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


def create_openrouter_client(api_key: str):
    """Create an OpenRouter client using requests."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://paper-digest.com",  # Replace with your domain
        "X-Title": "Paper Digest"  # Replace with your app name
    }
    return headers


def clean_and_extract_json(response_text: str) -> dict:
    """Clean response text and extract JSON, handling thinking tokens and other formatting.

    Returns a default dict with 'relevant': False if JSON parsing fails.
    """

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
            return {"relevant": False}

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
            return {"relevant": False}

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
        return {"relevant": False}

    # If all else fails, return default
    print(
        f"❌ Could not extract valid JSON from response: {response_text[:500]}...")
    print(f"❌ Cleaned response: {cleaned[:500]}...")
    return {"relevant": False}


@cached_llm_call
def assess_relevance_and_tags(text: str, api_key: str, temperature: float = 0.1, model: str = "openai/gpt-4o", feed_type: str = "ai-security") -> Tuple[Dict[str, Any], int]:
    """Assess if a paper is relevant and extract tags using OpenRouter."""
    headers = create_openrouter_client(api_key)

    # Optimized prompt to reduce token usage while maintaining essential instructions
    if feed_type == "web3-security":
        system_prompt = """Assess if this paper directly addresses vulnerabilities in smart contracts, blockchains, or Web3 systems.

ONLY RELEVANT if the paper:
- Identifies, analyzes, or prevents vulnerabilities in smart contracts (e.g., reentrancy, integer overflow, access control flaws)
- Studies security flaws in DeFi protocols (e.g., flash loan exploits, MEV attacks, oracle manipulation)
- Analyzes blockchain consensus vulnerabilities or attacks (e.g., 51% attacks, selfish mining, double-spending)
- Develops tools for smart contract security (e.g., auditing tools, formal verification, vulnerability detection)
- Examines bridge vulnerabilities, Layer 2 security flaws, or cross-chain attack vectors
- Studies cryptocurrency wallet/exchange security vulnerabilities or blockchain attack techniques

NOT RELEVANT:
- Papers using blockchain as a data management layer or infrastructure for other purposes (e.g., "blockchain for IoT security", "blockchain-based access control")
- General privacy technologies that happen to use blockchain (e.g., decentralized identity without vulnerability focus)
- Cryptocurrency trading, economics, or market analysis without security vulnerability aspects
- Blockchain applications without vulnerability or security flaw analysis
- Papers about blockchain benefits, performance, or general system design without security vulnerability focus

If relevant (score ≥3/5):
- Summary (2-4 bullet points)
- 3-5 tags
- Relevance score (1-5)
- Brief reason for score
- Paper type (Research/Survey/Benchmarking/Position/Other)
- Modalities (Text/Image/Video/Audio/Multimodal/Other)

If not relevant: {"relevant": false}

IMPORTANT: Output ONLY valid JSON. No explanations, no thinking tokens, no markdown. Just the JSON object."""
    else:
        system_prompt = """Assess if this paper directly addresses AI security, safety, or red teaming.

Relevant topics: LLM red teaming, jailbreaking, prompt injection, adversarial prompting, model extraction,
data poisoning, privacy attacks, alignment, robustness, safety evaluation, security standards.

NOT relevant:
- General AI/ML papers without security focus
- AI applications without security focus
- AI ethics without security aspects
- Federated learning papers (unless specifically about security attacks on federated learning)
- Federated unlearning papers
- Distributed machine learning without security vulnerability focus

If relevant (score ≥3/5):
- Summary (2-4 bullet points)
- 3-5 tags
- Relevance score (1-5)
- Brief reason for score
- Paper type (Research/Survey/Benchmarking/Position/Other)
- Modalities (Text/Image/Video/Audio/Multimodal/Other)

If not relevant: {"relevant": false}

IMPORTANT: Output ONLY valid JSON. No explanations, no thinking tokens, no markdown. Just the JSON object."""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"}
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


@cached_llm_call
def quick_assess_relevance(text: str, api_key: str, temperature: float = 0.1, model: str = "openai/gpt-4.1-nano", feed_type: str = "ai-security") -> Tuple[bool, int]:
    """Quick assessment of paper relevance using a smaller, cheaper model.

    This function performs a fast initial screening to determine if a paper is potentially
    relevant to AI security, safety, or red teaming. It uses a smaller, cheaper model
    to reduce costs for the initial filtering stage.

    Args:
        text: The paper title and abstract
        api_key: OpenRouter API key
        temperature: Temperature for the model (default: 0.1)
        model: Model to use (default: openai/gpt-4.1-nano)
        feed_type: Type of feed to assess for (default: ai-security)

    Returns:
        Tuple containing:
        - Boolean indicating if the paper is potentially relevant
        - Number of tokens used
    """
    headers = create_openrouter_client(api_key)

    if feed_type == "web3-security":
        system_prompt = """Determine if this paper is about vulnerabilities in smart contracts, blockchains, or Web3 systems.

ONLY "yes" if about: smart contract vulnerabilities, DeFi security flaws, blockchain consensus attacks,
security auditing tools, formal verification, exploit analysis, vulnerability detection,
wallet/exchange security vulnerabilities, bridge attacks, Layer 2 security flaws.

"no" if: using blockchain as infrastructure for other purposes, general privacy tech,
trading/economics, blockchain applications without vulnerability focus.

Respond with ONLY "yes" or "no"."""
    else:
        system_prompt = """Determine if this paper is potentially relevant to AI security, safety, or red teaming.

ONLY "yes" if about: LLM security, red teaming, jailbreaking, prompt injection, adversarial attacks,
model extraction, data poisoning, privacy attacks, alignment, robustness, safety evaluation, security standards.

"no" if: general AI/ML without security, federated learning, federated unlearning, distributed ML without security focus.

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


# Initialize the embedding model once as a global variable for efficiency
_embedding_model = None

def get_embedding_model():
    """Get or initialize the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return _embedding_model

def generate_embeddings(text: str) -> List[float]:
    """Generate embeddings for the given text.
    
    Args:
        text: The text to generate embeddings for
        
    Returns:
        List of floats representing the embedding vector
    """
    model = get_embedding_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()  # Convert numpy array to list for JSON serialization
