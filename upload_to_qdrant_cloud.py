import pandas as pd
from dotenv import load_dotenv
from utils.qdrant import init_qdrant_client, ensure_collection_exists, insert_paper
from search_paper import search_paper_by_title

# Load environment variables from .env file
load_dotenv()


def import_csv_to_qdrant():
    # Initialize Qdrant client
    client = init_qdrant_client()

    # Ensure collection exists
    ensure_collection_exists(client)

    try:
        # Read the entire CSV
        df = pd.read_csv('papers.csv')

        # Clean and prepare the data
        df = df.fillna('')  # Replace NaN with empty string

        # Process each paper
        for _, paper in df.iterrows():
            # Convert Series to dict
            paper_dict = paper.to_dict()

            # Convert column names to match expected format
            paper_dict = {k.lower().replace(' ', '_').replace('-', '_'): v
                          for k, v in paper_dict.items()}

            # Convert authors string to array
            if 'authors' in paper_dict and paper_dict['authors']:
                paper_dict['authors'] = [author.strip()
                                         for author in paper_dict['authors'].split(',')]

            # Convert url to string (revert to original behavior)
            if 'url' in paper_dict and paper_dict['url']:
                url_val = paper_dict['url']
                if ',' in url_val or ';' in url_val:
                    url_list = [u.strip() for part in url_val.split(';')
                                for u in part.split(',') if u.strip()]
                    paper_dict['url'] = '; '.join(url_list)
                else:
                    paper_dict['url'] = url_val.strip()

            # Convert tags to array if not empty
            if 'tags' in paper_dict and paper_dict['tags']:
                tag_val = paper_dict['tags']
                tag_list = [t.strip() for part in tag_val.split(';')
                            for t in part.split(',') if t.strip()]
                paper_dict['tags'] = tag_list

            # Check if paper already exists in KB
            title = paper_dict.get('title', '')
            if title and search_paper_by_title(client, title):
                print(f"⏭️  Skipping existing paper: {title}")
                continue

            # Insert into Qdrant
            success = insert_paper(client, paper_dict)
            if not success:
                print(
                    f"❌ Failed to insert paper: {paper_dict.get('title', 'Unknown title')}")
            else:
                print(f"✅ Added to Qdrant: {title}")

        print(f"Processed {len(df)} records...")

    except Exception as e:
        print(f"Error during import: {str(e)}")


if __name__ == "__main__":
    print("Starting CSV import to Qdrant (testing with first 2 papers)...")
    import_csv_to_qdrant()
    print("Import completed!")
