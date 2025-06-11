import pandas as pd
from utils.qdrant import init_qdrant_client, ensure_collection_exists, insert_paper


def import_csv_to_qdrant():
    # Initialize Qdrant client
    client = init_qdrant_client()

    # Ensure collection exists
    ensure_collection_exists(client)

    try:
        # Read CSV in chunks to handle large file
        chunk_size = 10000
        for chunk in pd.read_csv('papers.csv', chunksize=chunk_size):
            # Clean and prepare the data
            chunk = chunk.fillna('')  # Replace NaN with empty string

            # Process each paper in the chunk
            for _, paper in chunk.iterrows():
                # Convert Series to dict
                paper_dict = paper.to_dict()

                # Convert column names to match expected format
                paper_dict = {k.lower().replace(' ', '_').replace('-', '_'): v
                              for k, v in paper_dict.items()}

                # Insert into Qdrant
                success = insert_paper(client, paper_dict)
                if not success:
                    print(
                        f"Failed to insert paper: {paper_dict.get('title', 'Unknown title')}")

            print(f"Processed {len(chunk)} records...")

    except Exception as e:
        print(f"Error during import: {str(e)}")


if __name__ == "__main__":
    print("Starting CSV import to Qdrant...")
    import_csv_to_qdrant()
    print("Import completed!")
