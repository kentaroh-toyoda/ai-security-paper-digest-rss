#!/usr/bin/env python3
"""
Script to update daily rate limit for paid OpenRouter users.
Run this script if you've purchased 10+ credits to increase your daily limit to 1000 requests.
"""

import os
from dotenv import load_dotenv
from utils.llm import update_daily_limit_for_paid_user, check_rate_limit_status

# Load environment variables
load_dotenv()


def main():
    """Update daily limit for paid users."""
    print("🎯 OpenRouter Daily Limit Updater")
    print("=" * 40)

    # Check if API key is set
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY environment variable is not set")
        return

    print(f"✅ API key found: {api_key[:8]}...{api_key[-4:]}")

    # Show current status
    print("\n📊 Current Rate Limit Status:")
    check_rate_limit_status()

    # Update to paid user limits
    print("\n🔄 Updating daily limit for paid user...")
    update_daily_limit_for_paid_user()

    # Show updated status
    print("\n📊 Updated Rate Limit Status:")
    check_rate_limit_status()

    print("\n✅ Daily limit updated successfully!")
    print("💡 You now have 1000 requests per day instead of 50.")


if __name__ == "__main__":
    main()
