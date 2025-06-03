# utils/baserow.py

import requests
import re

BASEROW_API_URL = "https://api.baserow.io/api/database"

def paper_exists_in_baserow(paper_url, token, table_id):
    headers = {"Authorization": f"Token {token}"}
    arxiv_id = extract_arxiv_id(paper_url)
    query = arxiv_id if arxiv_id else paper_url

    url = f"{BASEROW_API_URL}/rows/table/{table_id}/?user_field_names=true&filter__URL__contains={query}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        results = response.json().get("results", [])
        return any(query in row.get("URL", "") for row in results)
    return False

def insert_to_baserow(row_data, token, table_id):
    headers = {
        "Authorization": f"Token {token}",
        "Content-Type": "application/json"
    }
    response = requests.post(
        f"{BASEROW_API_URL}/rows/table/{table_id}/?user_field_names=true",
        json=row_data,
        headers=headers
    )
    if response.status_code == 201:
        print(f"✅ Added to Baserow: {row_data['Title']}")
    else:
        print(f"❌ Baserow push failed for: {row_data['Title']}")
        print(response.text)

def extract_arxiv_id(url):
    match = re.search(r'arxiv\.org/(abs|pdf|html)/([0-9]+\.[0-9]+)', url)
    return match.group(2) if match else None

def ensure_baserow_fields_exist(token, table_id, required_fields):
    headers = {"Authorization": f"Token {token}"}
    fields_url = f"{BASEROW_API_URL}/fields/table/{table_id}/?user_field_names=true"
    existing_fields = requests.get(fields_url, headers=headers).json()
    existing_names = [f["name"] for f in existing_fields]

    for field in required_fields:
        if field not in existing_names:
            create_field(field, headers, table_id)

def create_field(field_name, headers, table_id):
    print(f"⚙️ Creating missing field: {field_name}")
    payload = {
        "table_id": table_id,
        "name": field_name,
        "type": "long_text"
    }
    response = requests.post(
        f"{BASEROW_API_URL}/fields/table/{table_id}/?user_field_names=true",
        json=payload,
        headers=headers
    )
    if response.status_code == 200:
        print(f"✅ Field created: {field_name}")
    else:
        print(f"❌ Failed to create field: {field_name}")
        print(response.text)