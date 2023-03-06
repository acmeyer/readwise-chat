import os
import pandas as pd
import openai
import tiktoken
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="data/chroma"))
import pandas as pd
import numpy as np
from typing import Iterator
from ast import literal_eval
from tenacity import retry, wait_random_exponential, stop_after_attempt
from dotenv import load_dotenv
load_dotenv()

EMBEDDINGS_MODEL = "text-embedding-ada-002"
OPENAI_EMBEDDING_ENCODING = "cl100k_base" # this the encoding for text-embedding-ada-002
MAX_EMBEDDING_TOKENS = 8191  # the maximum for text-embedding-ada-002 is 8191
EMBEDDINGS_INDEX_NAME = "book-notes"
BATCH_SIZE = 100

# Models a simple batch generator that make chunks out of an input DataFrame
class BatchGenerator:
    def __init__(self, batch_size: int = 10) -> None:
        self.batch_size = batch_size
    
    # Makes chunks out of an input DataFrame
    def to_batches(self, df: pd.DataFrame) -> Iterator[pd.DataFrame]:
        splits = self.splits_num(df.shape[0])
        if splits <= 1:
            yield df
        else:
            for chunk in np.array_split(df, splits):
                yield chunk

    # Determines how many chunks DataFrame contains
    def splits_num(self, elements: int) -> int:
        return round(elements / self.batch_size)
    
    __call__ = to_batches

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, model=EMBEDDINGS_MODEL) -> list[float]:
    text = text.replace("\n", " ") # OpenAI says removing newlines leads to better performance
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]

def get_embeddings(df: pd.DataFrame):
    print('Getting embeddings...')
    encoding = tiktoken.get_encoding(OPENAI_EMBEDDING_ENCODING)
    # omit any that are too long to embed
    df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))
    df = df[df.n_tokens <= MAX_EMBEDDING_TOKENS]
    df["embedding"] = df.combined.apply(lambda x: get_embedding(x, model=EMBEDDINGS_MODEL))
    print('Done getting embeddings.')
    return df

def compute_embeddings(df: pd.DataFrame):
    return {
        idx: get_embedding(r.combined) for idx, r in df.iterrows()
    }

def load_embeddings(filepath: str = 'data/embeddings/book_notes_w_embeddings.csv'):
    """Load the dataset with the embeddings from a CSV file."""
    df = pd.read_csv(filepath)
    # Convert embeddings to list
    df['embedding'] = df.embedding.apply(literal_eval)
    # Convert id to string
    df['id'] = df['id'].apply(str)
    return df

def load_dataset_for_embeddings(filepath: str = None, df: pd.DataFrame = None):
    """Load the dataset from a CSV file."""
    if filepath is not None:
        df = pd.read_csv(filepath)
    # Keep only the columns we need
    df = df[['id', 'highlight', 'book', 'author', 'note', 'location', 'location_type']]
    df['combined'] = (
        "Title: " + df['book'].str.strip().fillna('') + "; " +
        "Author: " + df['author'].str.strip().fillna('') + "; " +
        "Highlight: " + df['highlight'].str.strip().fillna('') +
        (("; Note: " + df['note'].str.strip()) if df['note'].notna().all() else '')
    )
    return df

def save_embeddings(df: pd.DataFrame, output_path: str = 'data/embeddings/book_notes_w_embeddings.csv'):
    """Save the dataset with the embeddings to a CSV file."""
    if os.path.exists(output_path):
        # Read in the existing file
        existing_df = pd.read_csv(output_path)
        # Append the new data to the existing data
        df = existing_df.append(df, ignore_index=True)
        df.to_csv(f'{output_path}', index=False)
        print(f"Saved embeddings to {output_path}.")
    else:  
        df.to_csv(f'{output_path}', index=False)
        print(f"Saved embeddings to {output_path}.")

# Using chromadb for embeddings search
def add_embeddings_to_chroma(df: pd.DataFrame):
    print(f'Adding {len(df)} embeddings to chromadb...')
    collection = chroma_client.get_or_create_collection(
        name=EMBEDDINGS_INDEX_NAME, 
        embedding_function=embedding_functions.OpenAIEmbeddingFunction(os.getenv('OPENAI_API_KEY'))
    )

    # Create a batch generator
    df_batcher = BatchGenerator(BATCH_SIZE)
    for batch_df in df_batcher(df):
        collection.add(
            embeddings=batch_df['embedding'].tolist(),
            documents=batch_df['combined'].tolist(),
            ids=batch_df['id'].tolist()
        )
    print('Done adding to chromadb.')

def query_embeddings_chroma(query: str, n_results: int = 5):
    query_embedding = get_embedding(query)
    collection = chroma_client.get_collection(
        name=EMBEDDINGS_INDEX_NAME, 
        embedding_function=embedding_functions.OpenAIEmbeddingFunction(os.getenv('OPENAI_API_KEY'))
    )
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    ids = results["ids"][0]
    distances = results["distances"][0]
    relevant_docs = [(distances[idx], id) for idx, id in enumerate(ids)]
    relevant_docs = sorted(relevant_docs, reverse=True)
    return relevant_docs