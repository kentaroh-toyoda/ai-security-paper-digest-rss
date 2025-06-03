#!/usr/bin/env python3

import os
import json
import requests
from dotenv import load_dotenv
from typing import Dict, Any, List

# Load environment variables
load_dotenv()

# Get Baserow credentials from environment
BASEROW_API_TOKEN = os.getenv("BASEROW_API_TOKEN")
BASEROW_TABLE_ID = os.getenv("BASEROW_TABLE_ID")
BASEROW_API_URL = "https://api.baserow.io/api/database"


def get_table_schema(token: str, table_id: str) -> List[Dict[str, Any]]:
    """Fetch the schema of a Baserow table."""
    headers = {"Authorization": f"Token {token}"}
    fields_url = f"{BASEROW_API_URL}/fields/table/{table_id}/?user_field_names=true"

    try:
        response = requests.get(fields_url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching schema: {str(e)}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        return []


def format_field_info(field: Dict[str, Any]) -> Dict[str, Any]:
    """Format field information for display."""
    return {
        "name": field.get("name", "Unknown"),
        "type": field.get("type", "Unknown"),
        "id": field.get("id", "Unknown"),
        "primary": field.get("primary", False),
        "read_only": field.get("read_only", False),
        "text_default": field.get("text_default", None),
        "number_decimal_places": field.get("number_decimal_places", None),
        "number_negative": field.get("number_negative", None),
        "date_format": field.get("date_format", None),
        "date_include_time": field.get("date_include_time", None),
        "date_time_format": field.get("date_time_format", None),
        "boolean_default": field.get("boolean_default", None),
    }


def main():
    if not BASEROW_API_TOKEN or not BASEROW_TABLE_ID:
        print("❌ Error: BASEROW_API_TOKEN and BASEROW_TABLE_ID must be set in .env file")
        return

    print("\n🔍 Fetching Baserow table schema...")
    fields = get_table_schema(BASEROW_API_TOKEN, BASEROW_TABLE_ID)

    if not fields:
        print("❌ No fields found or error occurred")
        return

    print(f"\n📊 Table Schema (Total fields: {len(fields)})")
    print("=" * 80)

    for field in fields:
        field_info = format_field_info(field)
        print(f"\nField: {field_info['name']}")
        print("-" * 40)
        print(f"Type: {field_info['type']}")
        print(f"ID: {field_info['id']}")
        print(f"Primary: {field_info['primary']}")
        print(f"Read-only: {field_info['read_only']}")

        # Print type-specific information
        if field_info['type'] == 'text':
            if field_info['text_default']:
                print(f"Default value: {field_info['text_default']}")
        elif field_info['type'] == 'number':
            print(f"Decimal places: {field_info['number_decimal_places']}")
            print(f"Allow negative: {field_info['number_negative']}")
        elif field_info['type'] == 'date':
            print(f"Date format: {field_info['date_format']}")
            print(f"Include time: {field_info['date_include_time']}")
            if field_info['date_include_time']:
                print(f"Time format: {field_info['date_time_format']}")
        elif field_info['type'] == 'boolean':
            if field_info['boolean_default'] is not None:
                print(f"Default value: {field_info['boolean_default']}")


if __name__ == "__main__":
    main()
