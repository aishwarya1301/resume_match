from __future__ import annotations

from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384 dims, fast, good quality

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
    """Embed a list of texts. Returns array of shape (N, 384)."""
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # L2-normalize for cosine similarity via dot product
        convert_to_numpy=True,
    )
    return embeddings


def embed_query(text: str) -> list[float]:
    """Embed a single query string. Returns a plain list for Chroma."""
    model = get_model()
    embedding = model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embedding.tolist()
