"""
Download the resume dataset from HuggingFace (no auth required).
The dataset is the same Kaggle dataset mirrored publicly on HuggingFace.

Usage:
    python scripts/download_data.py
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'resumes.csv')


def download_from_huggingface():
    print("Downloading dataset from HuggingFace (Divyaamith/Kaggle-Resume)...")
    from datasets import load_dataset
    ds = load_dataset("Divyaamith/Kaggle-Resume", split="train")
    df = ds.to_pandas()

    # Normalize column names to match expected schema
    col_map = {}
    for col in df.columns:
        if 'resume' in col.lower() and 'str' in col.lower():
            col_map[col] = 'Resume_str'
        elif 'resume' in col.lower() and 'html' in col.lower():
            col_map[col] = 'Resume_html'
        elif 'category' in col.lower():
            col_map[col] = 'Category'
        elif col.lower() == 'id':
            col_map[col] = 'ID'
    if col_map:
        df = df.rename(columns=col_map)

    # Ensure required columns exist
    if 'Resume_str' not in df.columns:
        # Try to find any text column
        text_cols = [c for c in df.columns if df[c].dtype == object and c not in ('Category',)]
        if text_cols:
            df = df.rename(columns={text_cols[0]: 'Resume_str'})

    if 'ID' not in df.columns:
        df['ID'] = range(len(df))

    return df


def main():
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'data'), exist_ok=True)

    if os.path.exists(OUTPUT_PATH):
        df = pd.read_csv(OUTPUT_PATH)
        print(f"Dataset already exists: {len(df)} rows. Delete data/resumes.csv to re-download.")
        return

    df = download_from_huggingface()

    print(f"Downloaded {len(df)} resumes")
    print(f"Columns: {list(df.columns)}")
    print(f"Categories: {sorted(df['Category'].unique()) if 'Category' in df.columns else 'N/A'}")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
