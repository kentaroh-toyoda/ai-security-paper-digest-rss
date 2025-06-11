import sqlite3
import pandas as pd
import os


def create_database():
    # Create a new SQLite database
    conn = sqlite3.connect('papers.db')
    cursor = conn.cursor()

    # Drop the papers table if it exists
    cursor.execute('DROP TABLE IF EXISTS papers')

    # Create the papers table with the correct column structure
    cursor.execute('''
    CREATE TABLE papers (
        id INTEGER PRIMARY KEY,
        title TEXT,
        authors TEXT,
        url TEXT,
        summary TEXT,
        tags TEXT,
        date TEXT,
        relevance REAL,
        clarity REAL,
        novelty REAL,
        significance REAL,
        try_worthiness REAL,
        justification TEXT,
        code_repository TEXT,
        paper_type TEXT,
        read BOOLEAN
    )
    ''')

    return conn


def import_csv_to_sqlite():
    # Create database and get connection
    conn = create_database()

    try:
        # Read CSV in chunks to handle large file
        chunk_size = 10000
        for chunk in pd.read_csv('papers.csv', chunksize=chunk_size):
            # Clean and prepare the data
            chunk = chunk.fillna('')  # Replace NaN with empty string

            # Rename columns to match SQLite table
            chunk.columns = [col.lower().replace(' ', '_').replace('-', '_')
                             for col in chunk.columns]

            # Insert data into SQLite
            chunk.to_sql('papers', conn, if_exists='append', index=False)

            print(f"Imported {len(chunk)} records...")

    except Exception as e:
        print(f"Error during import: {str(e)}")
        conn.rollback()
    finally:
        conn.close()


if __name__ == "__main__":
    print("Starting CSV import to SQLite...")
    import_csv_to_sqlite()
    print("Import completed!")
