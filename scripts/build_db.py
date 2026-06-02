import sqlite3
import pandas as pd
import os

DB_PATH = 'data/data.db'
CSV_PATH = 'data/processed/link_prediction_test.csv'
META_PATH = 'data/raw/amazon-meta.txt'

def parse_and_insert_metadata(cursor):
    if not os.path.exists(META_PATH):
        print(f"Warning: Metadata file {META_PATH} not found. Skipping products table.")
        return

    print("Parsing amazon-meta.txt and inserting into database...")
    
    # Create products table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            product_id TEXT PRIMARY KEY,
            asin TEXT,
            title TEXT,
            group_name TEXT
        )
    ''')
    cursor.execute('DELETE FROM products')

    batch_size = 50000
    records = []
    
    current_id = None
    current_asin = None
    current_title = None
    current_group = None

    with open(META_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('Id:'):
                # Save previous product if exists
                if current_id is not None:
                    records.append((current_id, current_asin, current_title, current_group))
                    if len(records) >= batch_size:
                        cursor.executemany('INSERT INTO products VALUES (?, ?, ?, ?)', records)
                        records = []
                
                # Reset for new product
                current_id = line.split('Id:')[1].strip()
                current_asin = None
                current_title = None
                current_group = None
                
            elif line.startswith('ASIN:'):
                current_asin = line.split('ASIN:')[1].strip()
            elif line.startswith('title:'):
                current_title = line.split('title:')[1].strip()
            elif line.startswith('group:'):
                current_group = line.split('group:')[1].strip()

        # Insert the last product
        if current_id is not None:
            records.append((current_id, current_asin, current_title, current_group))
            
        if records:
            cursor.executemany('INSERT INTO products VALUES (?, ?, ?, ?)', records)

    print("Successfully built products table.")

def insert_recommendations(cursor):
    if not os.path.exists(CSV_PATH):
        print(f"Warning: Could not find real data at {CSV_PATH}. Skipping recommendations.")
        return

    print("Reading real recommendations data from CSV...")
    df = pd.read_csv(CSV_PATH)
    
    # Use Product_A as product_id, Product_B as recommended_id, and Adamic_Adar_Score as score
    records = df[['Product_A', 'Product_B', 'Adamic_Adar_Score']].values.tolist()

    # Create table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recommendations (
            product_id TEXT,
            recommended_id TEXT,
            score REAL
        )
    ''')
    cursor.execute('DELETE FROM recommendations')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_product_id ON recommendations(product_id)')

    print(f"Inserting {len(records)} recommendations into database...")
    cursor.executemany('''
        INSERT INTO recommendations (product_id, recommended_id, score)
        VALUES (?, ?, ?)
    ''', records)

def build_database():
    print(f"Connecting to SQLite database at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Enable PRAGMA for faster inserts
    cursor.execute('PRAGMA synchronous = OFF')
    cursor.execute('PRAGMA journal_mode = MEMORY')

    parse_and_insert_metadata(cursor)
    insert_recommendations(cursor)

    conn.commit()
    conn.close()
    
    print("Database build completed successfully!")

if __name__ == '__main__':
    build_database()
