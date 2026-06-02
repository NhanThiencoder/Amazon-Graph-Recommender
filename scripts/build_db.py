import sqlite3
import pandas as pd
import os

DB_PATH = 'data/data.db'
CSV_PATH = 'data/processed/link_prediction_test.csv'

def create_real_data_db():
    if not os.path.exists(CSV_PATH):
        print(f"Error: Could not find real data at {CSV_PATH}")
        return

    print("Reading real data from CSV...")
    df = pd.read_csv(CSV_PATH)
    
    # We will use Product_A as product_id, Product_B as recommended_id, and Adamic_Adar_Score as score
    # You can change the score column if you prefer Jaccard_Score
    records = df[['Product_A', 'Product_B', 'Adamic_Adar_Score']].values.tolist()

    print("Connecting to SQLite database...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recommendations (
            product_id TEXT,
            recommended_id TEXT,
            score REAL
        )
    ''')

    # Clear existing data
    cursor.execute('DELETE FROM recommendations')

    # Create index
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_product_id ON recommendations(product_id)
    ''')

    print(f"Inserting {len(records)} records into database...")
    cursor.executemany('''
        INSERT INTO recommendations (product_id, recommended_id, score)
        VALUES (?, ?, ?)
    ''', records)

    conn.commit()
    conn.close()
    
    print(f"Successfully built recommendations database at {DB_PATH} using real data!")

if __name__ == '__main__':
    create_real_data_db()
