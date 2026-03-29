from __future__ import annotations

from typing import Optional
import numpy as np
from rank_bm25 import BM25Okapi
import chromadb


def semantic_search(
    collection: chromadb.Collection,
    query_embedding: list[float],
    n_results: int = 100,
    where: Optional[dict] = None,
) -> list[dict]:
    """Dense vector search via ChromaDB."""
    kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
        include=["metadatas", "documents", "distances"],
    )
    if where:
        kwargs["where"] = where
    results = collection.query(**kwargs)

    hits = []
    for i, doc_id in enumerate(results['ids'][0]):
        hits.append({
            'id': doc_id,
            'score': 1.0 - results['distances'][0][i],  # cosine distance → similarity
            'metadata': results['metadatas'][0][i],
            'document': results['documents'][0][i],
        })
    return hits


def bm25_search(
    bm25: BM25Okapi,
    records: list[dict],
    query: str,
    n_results: int = 100,
) -> list[dict]:
    """Keyword search using BM25."""
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:n_results]
    hits = []
    for idx in top_indices:
        if scores[idx] > 0:
            hits.append({
                'id': records[idx]['id'],
                'score': float(scores[idx]),
                'record': records[idx],
            })
    return hits


def reciprocal_rank_fusion(
    dense_hits: list[dict],
    sparse_hits: list[dict],
    k: int = 60,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> list[tuple[str, float]]:
    """
    Combine dense and sparse results using Reciprocal Rank Fusion.
    k=60 is the standard RRF constant that avoids over-weighting top results.
    Returns list of (id, rrf_score) sorted descending.
    """
    scores: dict[str, float] = {}

    for rank, hit in enumerate(dense_hits):
        doc_id = hit['id']
        scores[doc_id] = scores.get(doc_id, 0.0) + dense_weight * (1.0 / (k + rank + 1))

    for rank, hit in enumerate(sparse_hits):
        doc_id = hit['id']
        scores[doc_id] = scores.get(doc_id, 0.0) + sparse_weight * (1.0 / (k + rank + 1))

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(
    collection: chromadb.Collection,
    bm25: BM25Okapi,
    records: list[dict],
    query_embedding: list[float],
    query_text: str,
    top_k: int = 50,
) -> list[dict]:
    """
    Full hybrid search: dense + sparse → RRF fusion.
    Returns top_k candidates with full record data for reranking.
    """
    # Build a fast lookup from id → record
    id_to_record: dict[str, dict] = {r['id']: r for r in records}

    # Run both searches in parallel (both are CPU-bound, sequential is fine at this scale)
    dense_hits = semantic_search(collection, query_embedding, n_results=top_k * 2)
    sparse_hits = bm25_search(bm25, records, query_text, n_results=top_k * 2)

    # Fuse
    fused = reciprocal_rank_fusion(dense_hits, sparse_hits)[:top_k]

    # Build result list with full payloads
    dense_id_to_hit = {h['id']: h for h in dense_hits}
    candidates = []
    for doc_id, rrf_score in fused:
        record = id_to_record.get(doc_id, {})
        dense_hit = dense_id_to_hit.get(doc_id, {})
        candidates.append({
            'id': doc_id,
            'rrf_score': rrf_score,
            'semantic_score': dense_hit.get('score', 0.0),
            'resume_text': record.get('clean_text', dense_hit.get('document', '')),
            'category': record.get('category', ''),
            'skills': record.get('skills', []),
            'seniority': record.get('seniority', ''),
            'years_experience': record.get('years_experience'),
        })

    return candidates
