import os
from dotenv import load_dotenv
from utils.qdrant import init_qdrant_client, ensure_collection_exists, get_all_papers
from qdrant_client.http import models
from utils.qdrant import generate_point_id

# Load environment variables
load_dotenv()


def fix_authors_format():
    """Fix author format in Qdrant points by converting comma-separated strings to arrays."""
    # Initialize Qdrant client
    client = init_qdrant_client()
    ensure_collection_exists(client)

    # Get all papers
    papers = get_all_papers(client)
    print(f"Found {len(papers)} papers to check")

    fixed_count = 0
    for paper in papers:
        # Debug: Print paper info
        print(f"\nChecking paper: {paper.get('title', 'Unknown')}")
        print(f"Authors type: {type(paper.get('authors'))}")
        print(f"Authors value: {paper.get('authors')}")

        # Check if authors is a string
        if isinstance(paper.get('authors'), str):
            # Convert comma-separated string to array
            authors = [author.strip()
                       for author in paper['authors'].split(',') if author.strip()]

            # Generate point ID from URL
            point_id = generate_point_id(paper['url'])
            print(f"Generated point ID: {point_id}")

            try:
                client.set_payload(
                    collection_name="ai_security_papers",
                    payload={"authors": authors},
                    points=[point_id]
                )
                print(
                    f"✅ Fixed authors for paper: {paper.get('title', 'Unknown')}")
                print(f"  Old format: {paper['authors']}")
                print(f"  New format: {authors}")
                fixed_count += 1
            except Exception as e:
                print(
                    f"❌ Error updating paper {paper.get('title', 'Unknown')}: {str(e)}")

    print(f"\n📊 Summary:")
    print(f"Total papers checked: {len(papers)}")
    print(f"Papers fixed: {fixed_count}")


if __name__ == "__main__":
    print("Starting author format fix...")
    fix_authors_format()
    print("Fix completed!")
