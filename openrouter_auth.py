import requests
import json
import os
import time
from dotenv import load_dotenv
from collections import deque
from datetime import datetime, timedelta


class RateLimiter:
    def __init__(self, max_requests, time_window_seconds):
        """
        Initialize rate limiter

        Args:
            max_requests (int): Maximum number of requests allowed
            time_window_seconds (int): Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window_seconds
        self.requests = deque()

    def wait_if_needed(self):
        """Wait if necessary to respect rate limits"""
        now = datetime.now()

        # Remove requests older than the time window
        while self.requests and (now - self.requests[0]) > timedelta(seconds=self.time_window):
            self.requests.popleft()

        # If we've hit the limit, wait until the oldest request expires
        if len(self.requests) >= self.max_requests:
            sleep_time = (
                self.requests[0] + timedelta(seconds=self.time_window) - now).total_seconds()
            if sleep_time > 0:
                print(
                    f"Rate limit reached. Waiting {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                # Recursively check again after waiting
                return self.wait_if_needed()

        # Add current request timestamp
        self.requests.append(now)

    def make_request(self, url, headers=None, **kwargs):
        """Make a rate-limited request"""
        self.wait_if_needed()
        return requests.get(url, headers=headers, **kwargs)


# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError(
        "OPENROUTER_API_KEY not found in environment variables. Please check your .env file.")

# Initialize rate limiter based on your API response
# 10 requests per 10 seconds
rate_limiter = RateLimiter(max_requests=10, time_window_seconds=10)

# Make the rate-limited request
response = rate_limiter.make_request(
    url="https://openrouter.ai/api/v1/auth/key",
    headers={
        "Authorization": f"Bearer {api_key}"
    }
)

print(json.dumps(response.json(), indent=2))
