from __future__ import annotations

from typing import Optional
import json
import os
from functools import lru_cache

# ---------------------------------------------------------------------------
# Static expansion map — covers the 24 dataset categories + common variations
# ---------------------------------------------------------------------------
TITLE_EXPANSIONS: dict[str, dict] = {
    # IT / Engineering
    "software engineer": {
        "related": ["software developer", "SDE", "backend engineer", "full-stack developer",
                    "programmer", "application developer", "software programmer"],
        "skills": ["Python", "Java", "algorithms", "APIs", "databases", "Git", "Agile"],
        "category_hint": "INFORMATION-TECHNOLOGY",
    },
    "data scientist": {
        "related": ["machine learning engineer", "ML engineer", "AI researcher",
                    "data analyst", "research scientist", "data engineer"],
        "skills": ["Python", "Machine Learning", "PyTorch", "TensorFlow", "statistics",
                   "pandas", "scikit-learn", "SQL"],
        "category_hint": "INFORMATION-TECHNOLOGY",
    },
    "data engineer": {
        "related": ["data pipeline engineer", "ETL engineer", "analytics engineer",
                    "big data engineer", "data architect"],
        "skills": ["Spark", "Kafka", "Airflow", "SQL", "Python", "AWS", "data pipelines"],
        "category_hint": "INFORMATION-TECHNOLOGY",
    },
    "product manager": {
        "related": ["PM", "product owner", "product lead", "program manager",
                    "product director", "technical product manager"],
        "skills": ["roadmap", "stakeholder", "Agile", "Scrum", "user stories", "KPIs"],
        "category_hint": "BUSINESS-DEVELOPMENT",
    },
    "devops engineer": {
        "related": ["SRE", "site reliability engineer", "platform engineer",
                    "cloud engineer", "infrastructure engineer", "systems engineer"],
        "skills": ["Docker", "Kubernetes", "AWS", "Terraform", "CI/CD", "Linux", "monitoring"],
        "category_hint": "INFORMATION-TECHNOLOGY",
    },
    "frontend developer": {
        "related": ["UI developer", "React developer", "web developer", "frontend engineer",
                    "JavaScript developer", "Angular developer"],
        "skills": ["React", "JavaScript", "TypeScript", "CSS", "HTML", "Vue", "UX"],
        "category_hint": "INFORMATION-TECHNOLOGY",
    },
    "backend developer": {
        "related": ["backend engineer", "API developer", "server-side developer",
                    "software engineer", "Java developer", "Python developer"],
        "skills": ["Python", "Java", "Node.js", "REST APIs", "databases", "microservices"],
        "category_hint": "INFORMATION-TECHNOLOGY",
    },
    # Finance
    "financial analyst": {
        "related": ["finance analyst", "investment analyst", "equity analyst",
                    "business analyst", "FP&A analyst", "portfolio analyst"],
        "skills": ["financial modeling", "Excel", "valuation", "Bloomberg", "CFA", "SQL"],
        "category_hint": "FINANCE",
    },
    "accountant": {
        "related": ["CPA", "staff accountant", "senior accountant", "controller",
                    "financial accountant", "tax accountant", "auditor"],
        "skills": ["accounting", "GAAP", "QuickBooks", "Excel", "tax", "auditing", "SAP"],
        "category_hint": "ACCOUNTANT",
    },
    "banker": {
        "related": ["investment banker", "commercial banker", "bank manager",
                    "loan officer", "credit analyst", "relationship manager"],
        "skills": ["banking", "credit analysis", "financial products", "risk management"],
        "category_hint": "BANKING",
    },
    # Healthcare
    "nurse": {
        "related": ["registered nurse", "RN", "clinical nurse", "staff nurse",
                    "nurse practitioner", "LPN", "charge nurse"],
        "skills": ["patient care", "clinical", "HIPAA", "EMR", "medication administration"],
        "category_hint": "HEALTHCARE",
    },
    "doctor": {
        "related": ["physician", "medical doctor", "MD", "clinician", "specialist",
                    "general practitioner", "attending physician"],
        "skills": ["clinical", "patient care", "diagnosis", "medical", "EHR", "HIPAA"],
        "category_hint": "HEALTHCARE",
    },
    # Business
    "sales manager": {
        "related": ["sales executive", "account executive", "sales representative",
                    "business development manager", "regional sales manager"],
        "skills": ["sales", "CRM", "Salesforce", "pipeline", "quota", "B2B", "negotiation"],
        "category_hint": "SALES",
    },
    "hr manager": {
        "related": ["human resources manager", "HR business partner", "people manager",
                    "talent acquisition manager", "HRBP", "HR director"],
        "skills": ["recruiting", "talent acquisition", "HRIS", "employee relations", "payroll"],
        "category_hint": "HR",
    },
    "recruiter": {
        "related": ["talent acquisition specialist", "technical recruiter", "HR recruiter",
                    "staffing specialist", "sourcing specialist"],
        "skills": ["recruiting", "sourcing", "ATS", "LinkedIn", "interviews", "hiring"],
        "category_hint": "HR",
    },
    "teacher": {
        "related": ["educator", "instructor", "professor", "lecturer", "tutor",
                    "academic", "school teacher", "K-12 teacher"],
        "skills": ["curriculum", "lesson planning", "classroom management", "assessment"],
        "category_hint": "TEACHER",
    },
    "lawyer": {
        "related": ["attorney", "counsel", "advocate", "solicitor", "legal advisor",
                    "paralegal", "associate attorney", "partner"],
        "skills": ["litigation", "contracts", "legal research", "Westlaw", "compliance", "JD"],
        "category_hint": "ADVOCATE",
    },
    "consultant": {
        "related": ["management consultant", "strategy consultant", "business consultant",
                    "advisor", "engagement manager", "associate consultant"],
        "skills": ["strategy", "analysis", "PowerPoint", "Excel", "stakeholder management"],
        "category_hint": "CONSULTANT",
    },
    "designer": {
        "related": ["graphic designer", "UX designer", "UI designer", "visual designer",
                    "product designer", "web designer", "creative director"],
        "skills": ["Figma", "Photoshop", "Illustrator", "UX", "UI", "design systems", "Adobe"],
        "category_hint": "DESIGNER",
    },
    "chef": {
        "related": ["head chef", "sous chef", "executive chef", "cook", "culinary artist",
                    "pastry chef", "line cook"],
        "skills": ["culinary", "food preparation", "menu development", "kitchen management"],
        "category_hint": "CHEF",
    },
    "fitness trainer": {
        "related": ["personal trainer", "gym instructor", "fitness coach", "strength coach",
                    "health coach", "group fitness instructor"],
        "skills": ["personal training", "nutrition", "exercise programming", "CPR", "NASM"],
        "category_hint": "FITNESS",
    },
    "pilot": {
        "related": ["commercial pilot", "airline pilot", "captain", "first officer",
                    "flight instructor", "aviation professional"],
        "skills": ["ATP", "instrument rating", "aviation", "FAA", "flight operations"],
        "category_hint": "AVIATION",
    },
    "mechanical engineer": {
        "related": ["engineer", "manufacturing engineer", "product engineer",
                    "design engineer", "process engineer"],
        "skills": ["CAD", "SolidWorks", "AutoCAD", "manufacturing", "thermodynamics"],
        "category_hint": "ENGINEERING",
    },
    "digital marketer": {
        "related": ["marketing manager", "SEO specialist", "content marketer",
                    "social media manager", "growth marketer", "performance marketer"],
        "skills": ["SEO", "Google Analytics", "Facebook Ads", "content strategy", "email marketing"],
        "category_hint": "DIGITAL-MEDIA",
    },
}


def _normalize(title: str) -> str:
    return title.lower().strip()


def _fuzzy_match(title: str) -> Optional[str]:
    """Find the best matching key in TITLE_EXPANSIONS using substring matching."""
    norm = _normalize(title)
    # Exact match first
    if norm in TITLE_EXPANSIONS:
        return norm
    # Substring: key is in the query title
    for key in TITLE_EXPANSIONS:
        if key in norm or norm in key:
            return key
    # Word overlap
    query_words = set(norm.split())
    best_key, best_overlap = None, 0
    for key in TITLE_EXPANSIONS:
        key_words = set(key.split())
        overlap = len(query_words & key_words)
        if overlap > best_overlap:
            best_key, best_overlap = key, overlap
    return best_key if best_overlap > 0 else None


@lru_cache(maxsize=1000)
def expand_query(job_title: str) -> dict:
    """
    Expand a job title into related terms and skills.
    Returns a dict with 'related', 'skills', 'query_text', 'category_hint'.
    Falls back gracefully if no match found.
    """
    key = _fuzzy_match(job_title)
    if key and key in TITLE_EXPANSIONS:
        exp = TITLE_EXPANSIONS[key]
        query_text = (
            f"A {job_title} professional with experience in "
            f"{', '.join(exp['skills'][:5])}. "
            f"Also known as: {', '.join(exp['related'][:4])}."
        )
        return {**exp, "query_text": query_text}

    # Generic fallback
    return {
        "related": [job_title],
        "skills": [],
        "category_hint": None,
        "query_text": f"A professional working as a {job_title} with relevant skills and experience.",
    }


def build_rich_query(job_title: str) -> str:
    """Build a rich query document that embeds into the same space as resumes."""
    exp = expand_query(job_title)
    parts = [f"ROLES: {job_title}"]
    if exp.get('related'):
        parts[0] += ', ' + ', '.join(exp['related'][:4])
    if exp.get('skills'):
        parts.append(f"SKILLS: {', '.join(exp['skills'])}")
    parts.append(exp['query_text'])
    return '\n'.join(parts)
