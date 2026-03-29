from __future__ import annotations

from sentence_transformers import CrossEncoder
import numpy as np

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        print(f"Loading reranker model: {RERANK_MODEL}")
        _reranker = CrossEncoder(RERANK_MODEL, max_length=512)
    return _reranker


def rerank(
    query: str,
    candidates: list[dict],
    top_n: int = 10,
) -> list[dict]:
    """
    Cross-encoder reranking. The cross-encoder sees query + document together
    (not independently like bi-encoder embeddings), producing more accurate scores.

    Uses a 512-token window on each resume — enough to cover the most informative
    first ~400 words while keeping latency low.
    """
    if not candidates:
        return []

    reranker = get_reranker()

    # Truncate resume text to keep inference fast on CPU
    pairs = [
        (query, c['resume_text'][:1500])
        for c in candidates
    ]

    scores = reranker.predict(pairs, show_progress_bar=False)

    # Sort by reranker score
    scored = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True,
    )

    return [
        {**candidate, "rerank_score": float(score)}
        for score, candidate in scored[:top_n]
    ]
