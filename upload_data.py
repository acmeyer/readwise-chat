import time
import datetime
import os
import glob
import pandas as pd
import argparse
import sqlite3
from embeddings import (
    get_embeddings,
    save_embeddings,
    load_dataset_for_embeddings,
    add_embeddings_to_chroma,
)

def create_new_file(filepath: str):
    current_time = str(int(time.time() * 1000))
    file_name = filepath.split("/")[-1]
    file_name = file_name.split(".")[0]
    file_name = f"{file_name}_{current_time}.csv"
    new_file_path = f"data/highlights/{file_name}"
    # Copy the file to the new location
    os.system(f"cp {filepath} {new_file_path}")
    return new_file_path

def get_most_recently_updated_highlights_csv_file():
    # Get the most recent highlights data CSV file
    list_of_files = glob.glob('data/highlights/*.csv')
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def compare_csv_files(old_file_path: str, new_file_path: str):
    # Read in both CSV files
    old_data = pd.read_csv(old_file_path)
    new_data = pd.read_csv(new_file_path)

    # Find rows that have been removed
    removed_data = old_data.merge(new_data, on=list(old_data.columns), how='left', indicator=True)
    removed_data = removed_data[removed_data['_merge'] == 'left_only'].drop(columns='_merge')

    # Find rows that have been added
    added_data = new_data.merge(old_data, on=list(old_data.columns), how='left', indicator=True)
    added_data = added_data[added_data['_merge'] == 'left_only'].drop(columns='_merge')

    # Return the changed data
    return removed_data, added_data

def update_db(added_data, removed_data):
    # Connect to the database
    conn = sqlite3.connect("data/highlights.db")
    c = conn.cursor()

    # Remove the removed data from the database
    for _, row in removed_data.iterrows():
        c.execute("DELETE FROM highlights WHERE highlight = :highlight", {"highlight": row["Highlight"]})

    # Add the new data to the database
    for _, row in added_data.iterrows():
        c.execute("""INSERT INTO highlights 
                    (highlight, book, author, note, location, location_type) 
                    VALUES (?, ?, ?, ?, ?, ?)""", (
                      row["Highlight"], 
                      row["Book Title"], 
                      row["Book Author"], 
                      row["Note"], 
                      row["Location"], 
                      row["Location Type"]
                    )
                  )
        
    # Commit the changes
    conn.commit()
    conn.close()

def create_db(added_data: pd.DataFrame):
    # Create the database file if it doesn't exist
    if not os.path.exists("data/highlights.db"):
        os.system("touch data/highlights.db")
    # Connect to the database
    conn = sqlite3.connect("data/highlights.db")
    c = conn.cursor()

    # Create the table
    c.execute("""CREATE TABLE highlights
                (id INTEGER PRIMARY KEY,
                highlight TEXT,
                book TEXT,
                author TEXT,
                note TEXT,
                location TEXT,
                location_type TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit()
    
    for _, row in added_data.iterrows():
        c.execute("""INSERT INTO highlights 
                    (highlight, book, author, note, location, location_type) 
                    VALUES (?, ?, ?, ?, ?, ?)""", (
                      row["Highlight"], 
                      row["Book Title"], 
                      row["Book Author"], 
                      row["Note"], 
                      row["Location"], 
                      row["Location Type"]
                    )
                  )

    # Commit the changes
    conn.commit()
    conn.close()

def get_most_recently_added_data_from_db(since: str):
    # Connect to the database
    conn = sqlite3.connect("data/highlights.db")
    c = conn.cursor()
    # Get the most recently added data
    c.execute(f"SELECT id, highlight, book, author, note, location, location_type FROM highlights WHERE created_at >= '{since}'")
    data = c.fetchall()
    df = pd.DataFrame(data, columns=['id', 'highlight', 'book', 'author', 'note', 'location', 'location_type'])
    conn.close()
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="Specify the file to upload.", required=True)
    args = parser.parse_args()
    filepath = args.file
    # Get the most recently updated CSV file before uploading the new one
    most_recent_filepath = get_most_recently_updated_highlights_csv_file()
    # Upload the new file
    new_filepath = create_new_file(filepath)
    if most_recent_filepath:
        # Compare the new file to the most recently updated file
        removed_data, added_data = compare_csv_files(most_recent_filepath, new_filepath)
        # Process the data that changed
        update_db(added_data, removed_data)
    else:
        # No prior data, so create new database with data
        added_data = pd.read_csv(new_filepath)
        create_db(added_data)
    # Get the most recently changed data, then get and save the embeddings for it
    current_time = new_filepath.split('_')[-1].split('.')[0]
    timestamp = int(current_time)
    dt = datetime.datetime.fromtimestamp(timestamp / 1000)
    date = dt.strftime('%Y-%m-%d %H:%M:%S')
    df = get_most_recently_added_data_from_db(since=date)
    df = load_dataset_for_embeddings(df=df)
    df = get_embeddings(df)
    # Save the embeddings to a CSV file, just in case
    save_embeddings(df, 'data/embeddings/book_notes_w_embeddings.csv')
    # Upload embeddings to chroma
    add_embeddings_to_chroma(df)

