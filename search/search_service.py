from __future__ import annotations

import re
import time
from typing import Optional
from functools import lru_cache
from ingestion.embedder import embed_query
from ingestion.vector_store import load_index, is_index_built
from search.query_expander import expand_query, build_rich_query
from search.hybrid_search import hybrid_search
from search.reranker import rerank

_QUOTED_TERMS = re.compile(r'"([^"]+)"')


def parse_query(raw: str) -> tuple:
    """
    Split a raw query into a base job title and a list of mandatory terms.
    Anything in double quotes is treated as a required phrase.

    Example:
        'Software Engineer "Python" "AWS"'
        → job_title="Software Engineer", required_terms=["Python", "AWS"]
    """
    required_terms = _QUOTED_TERMS.findall(raw)
    job_title = _QUOTED_TERMS.sub('', raw).strip()
    # Collapse any extra whitespace left after removing quoted segments
    job_title = re.sub(r'\s+', ' ', job_title).strip()
    return job_title, required_terms


def matches_required_terms(resume_text: str, terms: list) -> bool:
    """Case-insensitive check that every required phrase appears in the resume."""
    text_lower = resume_text.lower()
    return all(term.lower() in text_lower for term in terms)


# Section headers that separate the job title from resume body text
_SECTION_HEADERS = re.compile(
    r'\b(Professional Summary|Professional Profile|Professional|Summary|Objective|'
    r'Profile|Experience|Skills|Education|Overview|Executive Profile|'
    r'Career Objective|Work Experience)\b',
    re.IGNORECASE,
)


def infer_role(resume_text: str, dataset_category: str) -> str:
    """
    Extract the actual job title from the resume text.
    Resumes from this dataset consistently open with an ALL-CAPS role heading
    followed by a section header (e.g. 'SOFTWARE DEVELOPER Professional Summary ...').
    We extract whatever comes before that first section header.
    Falls back to the dataset category if no role heading is found.
    """
    first_chunk = resume_text[:300]
    match = _SECTION_HEADERS.search(first_chunk)
    if match:
        role = first_chunk[:match.start()].strip()
        # Discard if it's too long (likely not a title), too short, or numeric
        if 2 < len(role) < 80 and not role.isdigit():
            return role.title()  # Title-case for display
    return dataset_category.title()

# Module-level cache so models and index load once per process
_index = None


def _get_index():
    global _index
    if _index is None:
        if not is_index_built():
            raise RuntimeError(
                "Index not built. Run: python scripts/ingest.py"
            )
        _index = load_index()
    return _index


def get_matching_results(
    raw_query: str,
    limit: int = 10,
    seniority: Optional[str] = None,
    min_years: Optional[int] = None,
) -> dict:
    """
    Main search entrypoint.

    Args:
        raw_query:  Job title, optionally with quoted mandatory terms.
                    e.g. 'Software Engineer "Python" "AWS"'
        limit:      Number of results to return (default 10)
        seniority:  Optional filter: entry | mid | senior | lead | executive
        min_years:  Optional filter: minimum years of experience

    Returns a dict with query metadata and a ranked list of matching resumes.
    """
    t0 = time.perf_counter()

    job_title, required_terms = parse_query(raw_query)
    if not job_title:
        # Query was only quoted terms — use them as the title too
        job_title = ' '.join(required_terms)

    collection, bm25, records = _get_index()
    expansion = expand_query(job_title)

    # Build a rich query that embeds into the same space as our resume documents
    rich_query = build_rich_query(job_title)

    # Embed once (cached model, single inference call)
    t_embed_start = time.perf_counter()
    query_embedding = embed_query(rich_query)
    t_embed = time.perf_counter() - t_embed_start

    # Hybrid retrieval: dense semantic + BM25 keyword → RRF fusion
    t_search_start = time.perf_counter()
    candidates = hybrid_search(
        collection=collection,
        bm25=bm25,
        records=records,
        query_embedding=query_embedding,
        query_text=rich_query,
        top_k=50,
    )
    t_search = time.perf_counter() - t_search_start

    # Post-retrieval filtering — applied after fusion so both branches are covered
    if seniority:
        candidates = [c for c in candidates if c.get("seniority") == seniority]
    if min_years is not None:
        candidates = [c for c in candidates if (c.get("years_experience") or 0) >= min_years]
    if required_terms:
        candidates = [c for c in candidates if matches_required_terms(c["resume_text"], required_terms)]

    # Cross-encoder reranking: top-50 → top-N
    t_rerank_start = time.perf_counter()
    rerank_query = f"{job_title}: {expansion.get('query_text', '')}"
    results = rerank(rerank_query, candidates, top_n=limit)
    t_rerank = time.perf_counter() - t_rerank_start

    t_total = time.perf_counter() - t0

    return {
        "query": job_title,
        "required_terms": required_terms,
        "expanded_titles": [job_title] + expansion.get('related', [])[:3],
        "category_hint": expansion.get('category_hint'),
        "total_candidates": len(candidates),
        "returned": len(results),
        "timing_ms": {
            "embed": round(t_embed * 1000),
            "search": round(t_search * 1000),
            "rerank": round(t_rerank * 1000),
            "total": round(t_total * 1000),
        },
        "results": [
            {
                "rank": i + 1,
                "relevance_score": round(r["rerank_score"], 4),
                "hybrid_score": round(r["rrf_score"], 6),
                "role": infer_role(r["resume_text"], r["category"]),
                "dataset_category": r["category"],
                "skills": r.get("skills", []),
                "seniority": r.get("seniority"),
                "years_experience": r.get("years_experience"),
                "resume_text": r["resume_text"],
                "resume_preview": r["resume_text"][:400] + "...",
            }
            for i, r in enumerate(results)
        ],
    }
