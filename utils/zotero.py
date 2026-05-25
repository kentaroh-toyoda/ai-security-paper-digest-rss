# utils/zotero.py

import os
import time
from pyzotero import zotero


def init_zotero_client():
    """Initialize Zotero client from environment variables."""
    library_id = os.getenv("ZOTERO_LIBRARY_ID")
    api_key = os.getenv("ZOTERO_API_KEY")
    library_type = os.getenv("ZOTERO_LIBRARY_TYPE", "user")
    return zotero.Zotero(library_id, library_type, api_key)


def ensure_collection_exists(zot, collection_name):
    """Find or create a Zotero collection by name. Returns the collection key."""
    collections = zot.collections_top()
    for col in collections:
        if col["data"]["name"] == collection_name:
            return col["key"]

    resp = zot.create_collections([{"name": collection_name}])
    if resp and "success" in resp:
        return list(resp["success"].values())[0]
    raise RuntimeError(f"Failed to create Zotero collection: {collection_name}")


def paper_exists(zot, paper_url, collection_key):
    """Check if a paper with the given URL already exists in the collection.

    On any unexpected response shape or API error, returns False rather than
    crashing — the caller will simply attempt the insert, and Zotero will
    handle true duplicates on its end.
    """
    normalized = paper_url.replace("http://", "https://").rstrip("/")
    try:
        results = zot.collection_items(
            collection_key, q=normalized, qmode="everything", limit=10
        )
    except Exception as e:
        print(f"⚠️ Zotero search failed for {paper_url}: {e}")
        return False

    if not isinstance(results, list):
        print(f"⚠️ Unexpected Zotero response type {type(results).__name__} for {paper_url}")
        return False

    for item in results:
        if not isinstance(item, dict):
            continue
        data = item.get("data") if isinstance(item.get("data"), dict) else {}
        item_url = (data.get("url") or "").replace("http://", "https://").rstrip("/")
        if item_url == normalized:
            return True
    return False


def _split_author_name(full_name):
    """Split a full name into firstName and lastName."""
    parts = full_name.strip().split()
    if len(parts) == 0:
        return "", ""
    if len(parts) == 1:
        return "", parts[0]
    return " ".join(parts[:-1]), parts[-1]


def _build_extra_field(paper_data):
    """Build the Zotero extra field from paper metadata."""
    lines = []

    relevance_score = paper_data.get("relevance_score")
    if relevance_score:
        lines.append(f"Relevance Score: {relevance_score}/5")

    paper_type = paper_data.get("paper_type")
    if paper_type:
        lines.append(f"Paper Type: {paper_type}")

    code_repo = paper_data.get("code_repository")
    if code_repo:
        lines.append(f"Code Repository: {code_repo}")

    summary = paper_data.get("summary")
    if summary:
        if isinstance(summary, list):
            lines.append("Summary:")
            for bullet in summary:
                lines.append(f"- {bullet}")
        elif isinstance(summary, str) and summary:
            lines.append(f"Summary: {summary}")

    reason = paper_data.get("reason")
    if reason:
        lines.append(f"Relevance Reason: {reason}")

    return "\n".join(lines)


def _build_zotero_item(zot, paper_data, collection_key):
    """Build a Zotero item dict from paper data."""
    source = paper_data.get("source", "arxiv")
    item_type = "conferencePaper" if source == "acl" else "preprint"

    # Get a valid template and fill it
    template = zot.item_template(item_type)
    template["title"] = paper_data.get("title", "")
    template["url"] = paper_data.get("url", "")
    template["abstractNote"] = paper_data.get("abstract", "")
    template["date"] = paper_data.get("published_date", "")
    template["collections"] = [collection_key]

    if item_type == "preprint":
        template["repository"] = "ArXiv"
        paper_id = paper_data.get("paper_id", "")
        if paper_id:
            template["archiveID"] = f"arXiv:{paper_id}"
    else:
        template["libraryCatalog"] = "ACL Anthology"

    # Authors
    authors = paper_data.get("authors", [])
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(",")]
    creators = []
    for name in authors:
        first, last = _split_author_name(name)
        creators.append({
            "creatorType": "author",
            "firstName": first,
            "lastName": last
        })
    template["creators"] = creators

    # Tags (topics + modalities + star)
    tags = []
    for topic in paper_data.get("topics", []):
        tags.append({"tag": topic})
    for modality in paper_data.get("modalities", []):
        tags.append({"tag": f"modality:{modality}"})
    if paper_data.get("star"):
        tags.append({"tag": "starred"})
    template["tags"] = tags

    # Extra field for structured metadata
    template["extra"] = _build_extra_field(paper_data)

    return template


def insert_paper(zot, paper_data, collection_key):
    """Insert a paper as a Zotero item. Returns True on success."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            item = _build_zotero_item(zot, paper_data, collection_key)
            resp = zot.create_items([item])
            if resp.get("failed"):
                print(f"Failed to insert into Zotero: {resp['failed']}")
                return False
            return True
        except Exception as e:
            if "429" in str(e) or "Too Many" in str(e):
                wait = 2 ** attempt * 5
                print(f"Zotero rate limit hit, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"Zotero insertion error: {e}")
                return False
    return False
