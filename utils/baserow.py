# utils/baserow.py

import requests
import re
import json
from typing import Dict, Any, List
from datetime import datetime

# Field mappings for Baserow columns
FIELD_TITLE = "field_4496823"
FIELD_URL = "field_4496824"
FIELD_ABSTRACT = "field_4496825"
FIELD_TAGS = "field_4496826"
FIELD_AUTHORS = "field_4496827"
FIELD_DATE = "field_4496828"
FIELD_RELEVANCE = "field_4496829"
FIELD_SUMMARY = "field_4496830"
FIELD_PAPER_TYPE = "field_4496831"
FIELD_CLARITY = "field_4496832"
FIELD_NOVELTY = "field_4496833"
FIELD_SIGNIFICANCE = "field_4496834"
FIELD_TRY_WORTHINESS = "field_4496835"
FIELD_JUSTIFICATION = "field_4496836"
FIELD_CODE_REPO = "field_4496837"

BASEROW_API_URL = "https://api.baserow.io/api/database"


def paper_exists_in_baserow(paper_url: str, token: str, table_id: str) -> bool:
    headers = {"Authorization": f"Token {token}"}
    arxiv_id = extract_arxiv_id(paper_url)
    query = arxiv_id if arxiv_id else paper_url

    url = f"{BASEROW_API_URL}/rows/table/{table_id}/?user_field_names=true&filter__URL__contains={query}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        results = response.json().get("results", [])
        return any(query in row.get("URL", "") for row in results)
    print(f"❌ Failed to check paper existence: {response.text}")
    return False


def validate_row_data(row_data: Dict[str, Any]) -> List[str]:
    """Validate row data before sending to Baserow."""
    errors = []

    # Check required fields
    required_fields = ["Title", "URL", "Summary", "Tags", "Authors", "Date"]
    for field in required_fields:
        if field not in row_data:
            errors.append(f"Missing required field: {field}")
        elif not row_data[field]:
            errors.append(f"Empty required field: {field}")

    # Check field lengths
    max_lengths = {
        "Title": 255,
        "URL": 255,
        "Summary": 10000,
        "Tags": 255,
        "Date": 10,
        "Clarity": 1,
        "Novelty": 1,
        "Significance": 1,
        "Try-worthiness": 1,
        "Justification": 1000,
        "Code repository": 255
    }

    for field, max_len in max_lengths.items():
        if field in row_data and row_data[field]:
            if isinstance(row_data[field], str) and len(row_data[field]) > max_len:
                errors.append(
                    f"Field {field} exceeds maximum length of {max_len}")
            elif isinstance(row_data[field], (int, float)) and field in ["Clarity", "Novelty", "Significance"]:
                if not (1 <= row_data[field] <= 5):
                    errors.append(f"Field {field} must be between 1 and 5")

    return errors


def prepare_row_data(row_data: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare row data for Baserow by ensuring correct types and handling null values."""
    prepared_data = {}

    # Handle rating fields (1-5 scale)
    rating_fields = ["Clarity", "Novelty", "Significance", "Relevance"]
    for field in rating_fields:
        if field in row_data:
            value = row_data[field]
            if value is None or value == 0:
                prepared_data[field] = None
            else:
                # Ensure value is between 1 and 5
                value = int(value) if isinstance(value, (int, float)) else None
                if value is not None:
                    value = max(1, min(5, value))
                prepared_data[field] = value

    # Handle boolean fields
    boolean_fields = ["Try-worthiness"]
    for field in boolean_fields:
        if field in row_data:
            value = row_data[field]
            prepared_data[field] = bool(value) if value is not None else None

    # Handle URL fields
    url_fields = ["URL", "Code repository"]
    for field in url_fields:
        if field in row_data:
            value = row_data[field]
            prepared_data[field] = str(value) if value is not None else None

    # Handle date field
    if "Date" in row_data:
        value = row_data["Date"]
        prepared_data["Date"] = str(value) if value is not None else None

    # Handle text fields
    text_fields = ["Title", "Summary", "Tags", "Authors", "Justification"]
    for field in text_fields:
        if field in row_data:
            value = row_data[field]
            prepared_data[field] = str(value) if value is not None else None

    return prepared_data


def insert_to_baserow(row_data: Dict[str, Any], token: str, table_id: str) -> bool:
    """Insert a row into Baserow with validation and error handling."""
    # Map field IDs to names for validation
    field_mapping = {
        "field_4496823": "Title",
        "field_4496824": "URL",
        "field_4496825": "Abstract",
        "field_4496826": "Tags",
        "field_4496827": "Authors",
        "field_4496828": "Date",
        "field_4496829": "Relevance",
        "field_4496830": "Summary",
        "field_4496831": "Paper Type",
        "field_4496832": "Clarity",
        "field_4496833": "Novelty",
        "field_4496834": "Significance",
        "field_4496835": "Try-worthiness",
        "field_4496836": "Justification",
        "field_4496837": "Code repository"
    }

    # Convert field IDs to names for validation
    validation_data = {}
    for field_id, value in row_data.items():
        if field_id in field_mapping:
            validation_data[field_mapping[field_id]] = value
        else:
            validation_data[field_id] = value

    # Validate data first
    errors = validate_row_data(validation_data)
    if errors:
        print(
            f"❌ Validation errors for {validation_data.get('Title', 'Unknown paper')}:")
        for error in errors:
            print(f"  - {error}")
        return False

    # Prepare data for Baserow
    prepared_data = prepare_row_data(validation_data)

    # Debug print the prepared data
    print("\nPrepared data for Baserow:")
    print(json.dumps(prepared_data, indent=2))

    headers = {
        "Authorization": f"Token {token}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            f"{BASEROW_API_URL}/rows/table/{table_id}/?user_field_names=true",
            json=prepared_data,
            headers=headers
        )

        # Consider both 200 and 201 as success
        if response.status_code in [200, 201]:
            print(f"✅ Added to Baserow: {validation_data['Title']}")
            return True
        else:
            print(f"❌ Baserow push failed for: {validation_data['Title']}")
            print("Response status:", response.status_code)
            print("Response headers:", json.dumps(
                dict(response.headers), indent=2))
            print("Response body:", response.text)

            # Try to parse the response as JSON for better error display
            try:
                error_json = response.json()
                print("\nDetailed error information:")
                print(json.dumps(error_json, indent=2))
            except:
                pass

            return False

    except Exception as e:
        print(f"❌ Error pushing to Baserow: {str(e)}")
        return False


def extract_arxiv_id(url: str) -> str:
    match = re.search(r'arxiv\.org/(abs|pdf|html)/([0-9]+\.[0-9]+)', url)
    return match.group(2) if match else None


def ensure_baserow_fields_exist(token: str, table_id: str, required_fields: List[str]) -> None:
    headers = {"Authorization": f"Token {token}"}
    fields_url = f"{BASEROW_API_URL}/fields/table/{table_id}/?user_field_names=true"

    try:
        response = requests.get(fields_url, headers=headers)
        if response.status_code != 200:
            print(f"❌ Failed to fetch fields: {response.text}")
            return

        existing_fields = response.json()
        existing_names = [f["name"] for f in existing_fields]

        for field in required_fields:
            if field not in existing_names:
                create_field(field, headers, table_id)
    except Exception as e:
        print(f"❌ Error ensuring fields exist: {str(e)}")


def create_field(field_name: str, headers: Dict[str, str], table_id: str) -> None:
    print(f"⚙️ Creating missing field: {field_name}")

    # Determine field type based on name
    field_type = "long_text"
    if field_name in ["Clarity", "Novelty", "Significance", "Relevance"]:
        field_type = "number"
    elif field_name == "Try-worthiness":
        field_type = "boolean"
    elif field_name == "Date":
        field_type = "date"
    elif field_name in ["Title", "URL", "Tags", "Code repository"]:
        field_type = "text"

    payload = {
        "table_id": table_id,
        "name": field_name,
        "type": field_type
    }

    try:
        response = requests.post(
            f"{BASEROW_API_URL}/fields/table/{table_id}/?user_field_names=true",
            json=payload,
            headers=headers
        )
        if response.status_code == 200:
            print(f"✅ Field created: {field_name}")
        else:
            print(f"❌ Failed to create field: {field_name}")
            print("Response status:", response.status_code)
            print("Response body:", response.text)
    except Exception as e:
        print(f"❌ Error creating field {field_name}: {str(e)}")


def ensure_conversation_table_exists(api_token: str, database_id: str) -> str:
    """Ensure the conversations table exists in Baserow and return its ID."""
    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json"
    }

    # First, get all tables in the database
    url = f"https://api.baserow.io/api/database/tables/database/{database_id}/"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Failed to get tables: {response.text}")

    tables = response.json()

    # Check if conversations table exists
    conversations_table = next(
        (table for table in tables if table["name"] == "Paper Conversations"), None)

    if conversations_table:
        return conversations_table["id"]

    # Create conversations table if it doesn't exist
    url = f"https://api.baserow.io/api/database/tables/database/{database_id}/"
    data = {
        "name": "Paper Conversations",
        "fields": [
            {"name": "Paper URL", "type": "text"},
            {"name": "Question", "type": "text"},
            {"name": "Answer", "type": "text"},
            {"name": "Timestamp", "type": "date"}
        ]
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200:
        raise Exception(
            f"Failed to create conversations table: {response.text}")

    return response.json()["id"]


def store_conversation(api_token: str, table_id: str, paper_url: str, question: str, answer: str):
    """Store a conversation in Baserow."""
    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json"
    }

    url = f"https://api.baserow.io/api/database/rows/table/{table_id}/"
    data = {
        "Paper URL": paper_url,
        "Question": question,
        "Answer": answer,
        "Timestamp": datetime.now().strftime("%Y-%m-%d")
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200:
        print(f"Warning: Failed to store conversation: {response.text}")


def get_conversation_history(api_token: str, table_id: str, paper_url: str) -> list:
    """Get conversation history for a paper from Baserow."""
    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json"
    }

    # Construct filter to get conversations for this paper
    url = f"https://api.baserow.io/api/database/rows/table/{table_id}/"
    params = {
        "filter__Paper_URL__equal": paper_url,
        "order_by": "Timestamp"
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        print(f"Warning: Failed to get conversation history: {response.text}")
        return []

    return response.json()["results"]


def get_all_papers(token: str, table_id: str) -> List[Dict[str, Any]]:
    """Fetch all papers from Baserow."""
    headers = {"Authorization": f"Token {token}"}
    url = f"{BASEROW_API_URL}/rows/table/{table_id}/?user_field_names=true"
    all_papers = []
    page = 1
    while True:
        page_url = f"{url}&page={page}"
        response = requests.get(page_url, headers=headers)
        if response.status_code != 200:
            print(f"❌ Failed to fetch papers: {response.text}")
            break
        data = response.json()
        papers = data.get("results", [])
        if not papers:
            break
        all_papers.extend(papers)
        page += 1
    return all_papers


def update_paper_in_baserow(paper, api_token, table_id):
    """Update a paper in Baserow."""
    if "id" not in paper:
        print("Error: Paper must have an 'id' field to update")
        return False

    url = f"https://api.baserow.io/api/database/rows/table/{table_id}/{paper['id']}/"
    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json"
    }

    # Remove the id field as it's not needed in the update payload
    update_data = paper.copy()
    update_data.pop("id", None)

    # Debug: print the update payload
    print("\n[DEBUG] Update payload:")
    print(json.dumps(update_data, indent=2))

    try:
        response = requests.patch(url, headers=headers, json=update_data)
        if response.status_code != 200:
            print(f"[DEBUG] Response status: {response.status_code}")
            print(f"[DEBUG] Response body: {response.text}")
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error updating paper in Baserow: {e}")
        if hasattr(e.response, 'text'):
            print(f"[DEBUG] Error response: {e.response.text}")
        return False


def delete_paper_from_baserow(paper_id: str, token: str, table_id: str) -> bool:
    """Delete a paper from Baserow."""
    url = f"{BASEROW_API_URL}/rows/table/{table_id}/{paper_id}/"
    headers = {
        "Authorization": f"Token {token}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.delete(url, headers=headers)
        if response.status_code == 204:  # 204 No Content is success for DELETE
            print(f"✅ Successfully deleted paper with ID: {paper_id}")
            return True
        else:
            print(f"❌ Failed to delete paper with ID: {paper_id}")
            print(f"Response status: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error deleting paper: {str(e)}")
        return False
