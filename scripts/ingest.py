"""
Ingestion pipeline — run once to build the search index.

Usage:
    python scripts/ingest.py [--csv path/to/resumes.csv]
"""
from __future__ import annotations

import sys
import os
import argparse

# Make sure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from ingestion.preprocessor import preprocess_dataset
from ingestion.embedder import embed_texts
from ingestion.vector_store import build_and_save_index

DEFAULT_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'resumes.csv')


def main():
    parser = argparse.ArgumentParser(description='Ingest resumes into the search index')
    parser.add_argument('--csv', default=DEFAULT_CSV, help='Path to resumes CSV file')
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found at {csv_path}")
        print("Run: python scripts/download_data.py   first")
        sys.exit(1)

    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} resumes across {df['Category'].nunique()} categories")
    print(f"Categories: {sorted(df['Category'].unique())}")

    print("\nPreprocessing...")
    records = preprocess_dataset(df)
    print(f"Preprocessed {len(records)} records")

    print("\nEmbedding documents (this takes ~1-2 minutes)...")
    embedding_docs = [r['embedding_doc'] for r in records]
    embeddings = embed_texts(embedding_docs, batch_size=64)
    print(f"Embedded {len(embeddings)} documents → shape {embeddings.shape}")

    print("\nBuilding index...")
    build_and_save_index(records, embeddings)

    print("\nDone! Index is ready. Start the server with:")
    print("  uvicorn api.main:app --reload --port 8000")


if __name__ == '__main__':
    main()
