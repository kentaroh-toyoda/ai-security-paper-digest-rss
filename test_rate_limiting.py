#!/usr/bin/env python3
"""
Test script for the rate limiting functionality.
This script demonstrates how the rate limiter works with OpenRouter's free tier limits.
"""

import os
import time
from dotenv import load_dotenv
from utils.llm import check_rate_limit_status, get_rate_limiter, make_rate_limited_request, create_openrouter_client

# Load environment variables
load_dotenv()

# Get API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("❌ OPENROUTER_API_KEY environment variable is not set")
    exit(1)


def test_rate_limiting():
    """Test the rate limiting functionality."""
    print("🧪 Testing Rate Limiting Functionality")
    print("=" * 50)

    # Check initial status
    print("\n📊 Initial rate limit status:")
    check_rate_limit_status()

    # Test making multiple requests
    print("\n🚀 Making test requests to demonstrate rate limiting...")

    headers = create_openrouter_client(OPENROUTER_API_KEY)

    # Test payload (minimal to save tokens)
    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "Say 'test' and nothing else"}
        ],
        "max_tokens": 5
    }

    successful_requests = 0
    failed_requests = 0

    for i in range(15):  # Try 15 requests to see rate limiting in action
        try:
            print(f"\n📡 Request {i + 1}/15...")

            # Check status before each request
            status = check_rate_limit_status()

            # Make the request
            response = make_rate_limited_request(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                payload=payload
            )

            if response.status_code == 200:
                successful_requests += 1
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                print(f"✅ Success: {content}")
            else:
                failed_requests += 1
                print(f"❌ Failed with status {response.status_code}")

        except Exception as e:
            failed_requests += 1
            print(f"❌ Error: {e}")

        # Small delay between requests for better visibility
        time.sleep(0.5)

    print(f"\n📊 Test Results:")
    print(f"Successful requests: {successful_requests}")
    print(f"Failed requests: {failed_requests}")
    print(
        f"Success rate: {successful_requests/(successful_requests+failed_requests)*100:.1f}%")

    # Final status check
    print(f"\n📊 Final rate limit status:")
    check_rate_limit_status()


def test_rate_limiter_directly():
    """Test the rate limiter class directly."""
    print("\n🔧 Testing Rate Limiter Class Directly")
    print("=" * 50)

    rate_limiter = get_rate_limiter()

    print("Testing rate limiter with 12 requests (should trigger waiting)...")

    start_time = time.time()

    for i in range(12):
        print(f"Request {i + 1}/12...")
        rate_limiter.wait_if_needed()
        status = rate_limiter.get_status()
        print(
            f"  Status: {status['requests_in_window']}/{status['max_requests']} requests in window")

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n⏱️ Total time for 12 requests: {total_time:.2f} seconds")
    print(f"Average time per request: {total_time/12:.2f} seconds")

    # Check final status
    final_status = rate_limiter.get_status()
    print(f"Final status: {final_status}")


def main():
    """Main function to run tests."""
    print("🎯 OpenRouter Rate Limiting Test Suite")
    print("This will test the rate limiting functionality with your free tier account.")
    print("Note: This will use some of your API quota.")

    response = input("\nDo you want to proceed? (y/N): ").strip().lower()
    if response != 'y':
        print("Test cancelled.")
        return

    try:
        # Test the rate limiter class directly first
        test_rate_limiter_directly()

        # Test with actual API calls
        test_rate_limiting()

        print("\n✅ All tests completed!")

    except KeyboardInterrupt:
        print("\n\n⏹️ Test interrupted by user.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")


if __name__ == "__main__":
    main()
