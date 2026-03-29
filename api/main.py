from __future__ import annotations

from typing import Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from search.search_service import get_matching_results
from ingestion.vector_store import is_index_built

app = FastAPI(
    title="Resume Search API",
    description="Search resumes by job title using hybrid vector + BM25 search with cross-encoder reranking.",
    version="1.0.0",
)

# Warm up models on startup so first request isn't slow
@app.on_event("startup")
async def startup():
    if is_index_built():
        print("Warming up models...")
        try:
            get_matching_results("software engineer", limit=1)
            print("Models ready.")
        except Exception as e:
            print(f"Warmup failed (non-fatal): {e}")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def ui():
    return HTMLResponse(DEMO_HTML)


@app.get("/search")
async def search(
    q: str = Query(..., description="Job title to search for", example="Software Engineer"),
    limit: int = Query(10, ge=1, le=50, description="Number of results"),
    seniority: Optional[str] = Query(None, description="Filter: entry | mid | senior | lead | executive"),
    min_years: Optional[int] = Query(None, description="Filter: minimum years of experience"),
):
    """
    Search resumes by job title.

    - **q**: Job title query (e.g. "Software Engineer", "Data Scientist", "Financial Analyst")
    - **limit**: Max results to return (1-50)
    - **seniority**: Optional seniority filter
    - **min_years**: Optional minimum years of experience filter
    """
    if not is_index_built():
        raise HTTPException(
            status_code=503,
            detail="Index not built. Run: python scripts/ingest.py"
        )
    try:
        result = get_matching_results(
            raw_query=q,
            limit=limit,
            seniority=seniority,
            min_years=min_years,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "index_built": is_index_built(),
    }


@app.get("/categories")
async def categories():
    """List available job categories in the dataset."""
    return {
        "categories": [
            "Information-Technology", "HR", "Finance", "Healthcare", "Teacher",
            "Advocate", "Business-Development", "Fitness", "Agriculture", "BPO",
            "Sales", "Consultant", "Digital-Media", "Automobile", "Chef",
            "Accountant", "Construction", "Public-Relations", "Banking", "Arts",
            "Aviation", "Engineering", "Designer",
        ]
    }


DEMO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Resume Search</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f5f5f5; color: #333; }
  .header { background: #1a1a2e; color: white; padding: 24px 32px; }
  .header h1 { font-size: 1.5rem; font-weight: 600; }
  .header p  { font-size: 0.85rem; color: #aaa; margin-top: 4px; }
  .search-bar { background: white; padding: 24px 32px; border-bottom: 1px solid #e0e0e0;
                display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  .search-bar input { flex: 1; min-width: 200px; padding: 10px 16px; font-size: 1rem;
                      border: 1.5px solid #ccc; border-radius: 8px; outline: none; }
  .search-bar input:focus { border-color: #4f46e5; }
  .search-bar select { padding: 10px 12px; border: 1.5px solid #ccc; border-radius: 8px;
                       font-size: 0.9rem; background: white; }
  .search-bar button { padding: 10px 24px; background: #4f46e5; color: white;
                       border: none; border-radius: 8px; font-size: 1rem;
                       cursor: pointer; font-weight: 500; }
  .search-bar button:hover { background: #4338ca; }
  .container { max-width: 1100px; margin: 0 auto; padding: 24px 32px; }
  .meta { font-size: 0.85rem; color: #666; margin-bottom: 16px; }
  .timing { display: inline-flex; gap: 12px; margin-left: 12px; }
  .timing span { background: #e8f4fd; color: #1a56db; padding: 2px 8px;
                 border-radius: 4px; font-size: 0.8rem; }
  .card { background: white; border-radius: 10px; padding: 20px 24px;
          margin-bottom: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
  .card-header { display: flex; justify-content: space-between; align-items: flex-start;
                 margin-bottom: 12px; gap: 12px; }
  .rank { font-size: 1.1rem; font-weight: 700; color: #4f46e5; min-width: 32px; }
  .badges { display: flex; gap: 6px; flex-wrap: wrap; }
  .badge { padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 500; }
  .badge-category { background: #ede9fe; color: #6d28d9; }
  .badge-seniority { background: #fef3c7; color: #92400e; }
  .badge-years { background: #d1fae5; color: #065f46; }
  .score { font-size: 0.8rem; color: #888; text-align: right; line-height: 1.6; }
  .skills { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
  .skill { background: #f3f4f6; color: #374151; padding: 2px 8px; border-radius: 4px;
           font-size: 0.75rem; }
  .preview { font-size: 0.85rem; color: #555; line-height: 1.6;
             border-top: 1px solid #f0f0f0; padding-top: 12px; }
  .expand-btn { background: none; border: none; color: #4f46e5; cursor: pointer;
                font-size: 0.8rem; margin-top: 6px; padding: 0; }
  .full-text { display: none; font-size: 0.82rem; color: #555; line-height: 1.6;
               margin-top: 8px; white-space: pre-wrap; word-break: break-word;
               max-height: 400px; overflow-y: auto; }
  .loading { text-align: center; padding: 40px; color: #888; }
  .error { background: #fef2f2; border: 1px solid #fca5a5; border-radius: 8px;
           padding: 16px; color: #dc2626; margin-top: 16px; }
  .examples { font-size: 0.82rem; color: #888; }
  .examples span { cursor: pointer; color: #4f46e5; text-decoration: underline; margin-left: 8px; }
</style>
</head>
<body>
<div class="header">
  <h1>Resume Search</h1>
  <p>Hybrid semantic + keyword search with cross-encoder reranking</p>
</div>
<div class="search-bar">
  <input type="text" id="query" placeholder='Job title — use "quotes" for required terms, e.g.: Software Engineer "Python" "AWS"'
         onkeydown="if(event.key==='Enter') doSearch()">
  <select id="limit">
    <option value="5">5 results</option>
    <option value="10" selected>10 results</option>
    <option value="20">20 results</option>
  </select>
  <select id="seniority">
    <option value="">Any seniority</option>
    <option value="entry">Entry</option>
    <option value="mid">Mid</option>
    <option value="senior">Senior</option>
    <option value="lead">Lead</option>
    <option value="executive">Executive</option>
  </select>
  <button onclick="doSearch()">Search</button>
</div>
<div class="container">
  <div class="examples">Try:
    <span onclick="setQuery('Software Engineer')">Software Engineer</span>
    <span onclick="setQuery('Data Scientist')">Data Scientist</span>
    <span onclick="setQuery('Financial Analyst')">Financial Analyst</span>
    <span onclick="setQuery('HR Manager')">HR Manager</span>
    <span onclick="setQuery('Nurse')">Nurse</span>
    <span onclick="setQuery('Teacher')">Teacher</span>
    <span onclick="setQuery('Lawyer')">Lawyer</span>
  </div>
  <div id="results" style="margin-top: 20px;"></div>
</div>

<script>
function setQuery(q) {
  document.getElementById('query').value = q;
  doSearch();
}

async function doSearch() {
  const q = document.getElementById('query').value.trim();
  if (!q) return;
  const limit = document.getElementById('limit').value;
  const seniority = document.getElementById('seniority').value;

  const resultsEl = document.getElementById('results');
  resultsEl.innerHTML = '<div class="loading">Searching...</div>';

  let url = `/search?q=${encodeURIComponent(q)}&limit=${limit}`;
  if (seniority) url += `&seniority=${seniority}`;

  try {
    const resp = await fetch(url);
    const data = await resp.json();
    if (!resp.ok) {
      resultsEl.innerHTML = `<div class="error">Error: ${data.detail || 'Unknown error'}</div>`;
      return;
    }
    renderResults(data);
  } catch(e) {
    resultsEl.innerHTML = `<div class="error">Request failed: ${e.message}</div>`;
  }
}

function renderResults(data) {
  const el = document.getElementById('results');
  const t = data.timing_ms;
  const expanded = data.expanded_titles.slice(1,4).join(', ');
  const reqTerms = (data.required_terms || []).map(t => `<code style="background:#fef9c3;padding:1px 5px;border-radius:3px">"${t}"</code>`).join(' ');

  let html = `<div class="meta">
    Found <strong>${data.total_candidates}</strong> candidates → ranked top <strong>${data.returned}</strong>
    &nbsp;|&nbsp; Expanded: <em>${data.expanded_titles[0]}${expanded ? ', ' + expanded : ''}</em>
    ${reqTerms ? `&nbsp;|&nbsp; Required: ${reqTerms}` : ''}
    <span class="timing">
      <span>embed ${t.embed}ms</span>
      <span>search ${t.search}ms</span>
      <span>rerank ${t.rerank}ms</span>
      <span>total ${t.total}ms</span>
    </span>
  </div>`;

  for (const r of data.results) {
    const skills = (r.skills || []).slice(0,8).map(s => `<span class="skill">${s}</span>`).join('');
    const yr = r.years_experience ? `${r.years_experience}yr` : '';
    const dimmed = r.relevance_score < 0 ? 'opacity:0.45;filter:grayscale(0.4)' : '';
    html += `
    <div class="card" style="${dimmed}">
      <div class="card-header">
        <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
          <span class="rank">#${r.rank}</span>
          <div class="badges">
            <span class="badge badge-category">${r.role}</span>
            <span class="badge" style="background:#f1f5f9;color:#64748b;font-size:0.7rem" title="Original dataset label">📂 ${r.dataset_category}</span>
            ${r.seniority ? `<span class="badge badge-seniority">${r.seniority}</span>` : ''}
            ${yr ? `<span class="badge badge-years">${yr}</span>` : ''}
          </div>
        </div>
        <div class="score">
          relevance: ${(r.relevance_score).toFixed(3)}<br>
          hybrid: ${(r.hybrid_score * 1000).toFixed(2)}
        </div>
      </div>
      ${skills ? `<div class="skills">${skills}</div>` : ''}
      <div class="preview">${r.resume_preview.replace(/</g,'&lt;')}
        <button class="expand-btn" onclick="toggleFull(this)">Show full resume ▼</button>
        <div class="full-text">${r.resume_text.replace(/</g,'&lt;')}</div>
      </div>
    </div>`;
  }

  el.innerHTML = html;
}

function toggleFull(btn) {
  const full = btn.nextElementSibling;
  if (full.style.display === 'block') {
    full.style.display = 'none';
    btn.textContent = 'Show full resume ▼';
  } else {
    full.style.display = 'block';
    btn.textContent = 'Hide ▲';
  }
}
</script>
</body>
</html>"""
