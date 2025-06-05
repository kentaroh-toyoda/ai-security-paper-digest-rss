import os
import requests
from dotenv import load_dotenv

load_dotenv()

BASEROW_API_TOKEN = os.getenv("BASEROW_API_TOKEN")
BASEROW_DATABASE_ID = os.getenv("BASEROW_DATABASE_ID")

# Print token length and first few characters (for debugging)
print(f"Token length: {len(BASEROW_API_TOKEN) if BASEROW_API_TOKEN else 0}")
print(
    f"Token preview: {BASEROW_API_TOKEN[:10]}..." if BASEROW_API_TOKEN else "No token found")

# Try different authorization header formats
auth_formats = [
    f"Token {BASEROW_API_TOKEN}",  # Original format
    f"Bearer {BASEROW_API_TOKEN}",  # Bearer token format
    BASEROW_API_TOKEN,  # Raw token
    f"JWT {BASEROW_API_TOKEN}"  # JWT format
]

url = f"https://api.baserow.io/api/database/tables/database/{BASEROW_DATABASE_ID}/"

for auth_format in auth_formats:
    print(f"\nTrying authorization format: {auth_format}")
    headers = {
        "Authorization": auth_format,
        "Content-Type": "application/json"
    }

    response = requests.get(url, headers=headers)
    print(f"Status code: {response.status_code}")
    print(f"Response body: {response.text}")

    if response.status_code == 200:
        print("✅ Success with this format!")
        break

# Try a different endpoint to test if it's a specific endpoint issue
print("\nTrying a different endpoint...")
test_url = "https://api.baserow.io/api/user/token-auth/"
test_response = requests.post(test_url, json={"token": BASEROW_API_TOKEN})
print(f"Test endpoint status code: {test_response.status_code}")
print(f"Test endpoint response: {test_response.text}")
