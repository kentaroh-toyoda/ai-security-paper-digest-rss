#!/usr/bin/env python3
"""
Simple utility to check OpenRouter rate limit status.
This script shows your current rate limit status without making API calls.
"""

import os
from dotenv import load_dotenv
from utils.llm import check_rate_limit_status, get_rate_limiter

# Load environment variables
load_dotenv()


def main():
    """Check and display rate limit status."""
    print("📊 OpenRouter Rate Limit Status Checker")
    print("=" * 40)

    # Check if API key is set
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY environment variable is not set")
        return

    print(f"✅ API key found: {api_key[:8]}...{api_key[-4:]}")

    # Get rate limiter status
    rate_limiter = get_rate_limiter()
    status = rate_limiter.get_status()

    print(f"\n📈 Current Status:")
    print(
        f"  Requests in current window: {status['requests_in_window']}/{status['max_requests']}")
    print(f"  Window duration: {status['window_seconds']} seconds")
    print(
        f"  Time until window resets: {status['time_until_reset']:.2f} seconds")

    # Calculate usage percentage
    usage_percentage = (status['requests_in_window'] /
                        status['max_requests']) * 100

    print(f"\n📊 Usage Analysis:")
    print(f"  Usage: {usage_percentage:.1f}%")

    if usage_percentage >= 100:
        print("  ⚠️ Rate limit window is full!")
        print("  💡 You'll need to wait before making more requests.")
    elif usage_percentage >= 80:
        print("  ⚠️ Approaching rate limit!")
        print("  💡 Consider spacing out your requests.")
    elif usage_percentage >= 50:
        print("  ⚡ Moderate usage")
        print("  💡 You have room for more requests.")
    else:
        print("  ✅ Low usage")
        print("  💡 You have plenty of capacity for requests.")

    # Show recommendations
    print(f"\n💡 Recommendations:")
    if status['requests_in_window'] > 0:
        print(
            f"  • Next request will be available in: {status['time_until_reset']:.2f} seconds")
    else:
        print("  • You can make requests immediately")

    print(
        f"  • Free tier limit: {status['max_requests']} requests per {status['window_seconds']} seconds")
    print(
        f"  • Average rate: {status['max_requests']/status['window_seconds']:.1f} requests/second")


if __name__ == "__main__":
    main()
