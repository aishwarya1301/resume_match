from __future__ import annotations

import os
import pickle
import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

STORAGE_DIR = os.path.join(os.path.dirname(__file__), '..', 'storage')
CHROMA_DIR = os.path.join(STORAGE_DIR, 'chroma')
BM25_PATH = os.path.join(STORAGE_DIR, 'bm25.pkl')
RECORDS_PATH = os.path.join(STORAGE_DIR, 'records.pkl')
COLLECTION_NAME = "resumes"


def get_chroma_client() -> chromadb.ClientAPI:
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR)


def get_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def build_and_save_index(records: list[dict], embeddings: np.ndarray) -> None:
    """
    Ingest all records into ChromaDB (dense vectors) and build a BM25 index
    (sparse/keyword) for hybrid search. Both are persisted to disk.
    """
    os.makedirs(STORAGE_DIR, exist_ok=True)

    # --- ChromaDB (dense) ---
    client = get_chroma_client()
    # Drop and recreate for clean re-ingestion
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = get_collection(client)

    print("Upserting into ChromaDB...")
    batch_size = 100
    for i in tqdm(range(0, len(records), batch_size)):
        batch_records = records[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]
        collection.add(
            ids=[r['id'] for r in batch_records],
            embeddings=batch_embeddings.tolist(),
            metadatas=[{
                'category': r['category'],
                'seniority': r['seniority'],
                'years_experience': r.get('years_experience') or -1,
                'skills': ','.join(r.get('skills', [])),
            } for r in batch_records],
            documents=[r['clean_text'] for r in batch_records],
        )

    # --- BM25 (sparse/keyword) ---
    print("Building BM25 index...")
    tokenized_corpus = [doc['embedding_doc'].lower().split() for doc in records]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_PATH, 'wb') as f:
        pickle.dump(bm25, f)

    # Save full records for result payload
    with open(RECORDS_PATH, 'wb') as f:
        pickle.dump(records, f)

    print(f"Index built: {len(records)} resumes stored.")


def is_index_built() -> bool:
    return os.path.exists(BM25_PATH) and os.path.exists(RECORDS_PATH)


def load_index() -> tuple[chromadb.Collection, BM25Okapi, list[dict]]:
    client = get_chroma_client()
    collection = get_collection(client)
    with open(BM25_PATH, 'rb') as f:
        bm25 = pickle.load(f)
    with open(RECORDS_PATH, 'rb') as f:
        records = pickle.load(f)
    return collection, bm25, records
