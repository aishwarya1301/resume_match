# Resume Search

Search ~2,500 resumes by job title using hybrid semantic + keyword retrieval with cross-encoder reranking. No API keys required — all models run locally.

## How It Works

```
Query: 'Software Engineer "Python" "AWS"'
         │
         ▼
  Query Expansion          expand title → related roles + skills
         │
         ▼
  Hybrid Retrieval         dense (sentence-transformers) + sparse (BM25) → RRF fusion
         │
         ▼
  Hard Filters             seniority, min_years, required quoted terms
         │
         ▼
  Cross-Encoder Rerank     top-50 candidates → top-N accurate results
```

**Accuracy levers:**
- **Query expansion** — `"Software Engineer"` is expanded into related titles and skills before embedding, so the query vector lands in the same neighborhood as actual engineer resumes
- **Hybrid search** — BM25 catches exact keyword matches; dense vectors catch semantic equivalents (`"SDE"` ≈ `"software engineer"`); Reciprocal Rank Fusion combines both
- **Cross-encoder reranking** — unlike embedding models that score query and document independently, the cross-encoder sees both together and produces more accurate final rankings

## Stack

| Component | Choice |
|-----------|--------|
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers, 384 dims) |
| Vector store | ChromaDB (embedded, no server needed) |
| Keyword search | BM25 (rank-bm25) |
| Fusion | Reciprocal Rank Fusion |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| API | FastAPI + uvicorn |

## Setup

**Requirements:** Python 3.9+

```bash
git clone <repo>
cd resume_match

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Ingest

Downloads the dataset from HuggingFace and builds the search index (runs once, ~2 minutes):

```bash
python scripts/download_data.py
python scripts/ingest.py
```

## Run

```bash
uvicorn api.main:app --port 8000
```

Open **http://localhost:8000** for the search UI, or **http://localhost:8000/docs** for the API reference.

## API

### `GET /search`

| Parameter | Type | Description |
|-----------|------|-------------|
| `q` | string | Job title query. Wrap terms in `"quotes"` to require them. |
| `limit` | int | Results to return (1–50, default 10) |
| `seniority` | string | Filter: `entry` \| `mid` \| `senior` \| `lead` \| `executive` |
| `min_years` | int | Filter: minimum years of experience |

**Examples:**

```bash
# Basic search
curl "http://localhost:8000/search?q=Software+Engineer"

# Require specific skills
curl "http://localhost:8000/search?q=Software+Engineer+%22Python%22+%22AWS%22"

# Filter by seniority
curl "http://localhost:8000/search?q=Data+Scientist&seniority=senior"

# Combine filters
curl "http://localhost:8000/search?q=Financial+Analyst+%22CFA%22&min_years=5"
```

**Response shape:**

```json
{
  "query": "Software Engineer",
  "required_terms": ["Python", "AWS"],
  "expanded_titles": ["Software Engineer", "software developer", "SDE", "backend engineer"],
  "total_candidates": 12,
  "returned": 10,
  "timing_ms": { "embed": 22, "search": 14, "rerank": 310, "total": 346 },
  "results": [
    {
      "rank": 1,
      "relevance_score": 6.48,
      "hybrid_score": 0.014286,
      "role": "Software Developer",
      "dataset_category": "INFORMATION-TECHNOLOGY",
      "skills": ["Python", "AWS", "PostgreSQL", "Docker"],
      "seniority": "mid",
      "years_experience": 4,
      "resume_text": "...",
      "resume_preview": "..."
    }
  ]
}
```

> **`role`** is extracted from the resume text (the self-reported heading). **`dataset_category`** is the original dataset label, which can be noisy — e.g. a "Software Developer" resume may be tagged `AGRICULTURE` in the source data.

## Performance

Measured on CPU (Apple Silicon), 2,484 resumes:

| Stage | Latency |
|-------|---------|
| Query expansion (cached) | < 1 ms |
| Embed query | ~20 ms |
| Hybrid retrieval | ~15 ms |
| Cross-encoder rerank (50 → 10) | ~300 ms |
| **Total** | **~335 ms** |

