from __future__ import annotations

import re
from typing import Optional, List
import pandas as pd


def clean_text(text: str) -> str:
    """Normalize whitespace and remove non-printable characters."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    return text[:6000]  # ~4500 token cap, well within embedding limits


def extract_skills_heuristic(text: str) -> list[str]:
    """
    Regex-based skill extraction. Looks for common tech/domain keywords.
    Fast, no API needed. Good enough for metadata filtering.
    """
    SKILL_PATTERNS = [
        # Languages
        r'\b(Python|Java|JavaScript|TypeScript|C\+\+|C#|Go|Rust|Ruby|PHP|Swift|Kotlin|Scala|R)\b',
        # Web/Backend
        r'\b(React|Angular|Vue|Node\.js|Django|Flask|Spring|FastAPI|Rails|Laravel)\b',
        # Data/ML
        r'\b(TensorFlow|PyTorch|scikit-learn|Pandas|NumPy|Spark|Hadoop|Kafka|Airflow)\b',
        # Cloud/Infra
        r'\b(AWS|GCP|Azure|Docker|Kubernetes|Terraform|CI/CD|Jenkins|GitHub Actions)\b',
        # Databases
        r'\b(PostgreSQL|MySQL|MongoDB|Redis|Elasticsearch|DynamoDB|BigQuery|Snowflake)\b',
        # Domains
        r'\b(Machine Learning|Deep Learning|NLP|Computer Vision|Data Science|Analytics)\b',
        # Finance
        r'\b(accounting|auditing|financial analysis|Excel|QuickBooks|SAP|CPA|CFA)\b',
        # Healthcare
        r'\b(EHR|EMR|HIPAA|clinical|nursing|patient care|medical|pharmacy)\b',
        # Business
        r'\b(project management|Agile|Scrum|PMP|stakeholder|business development|CRM|Salesforce)\b',
        # Design
        r'\b(Figma|Photoshop|Illustrator|UX|UI design|InDesign|Sketch|Adobe)\b',
    ]
    found = set()
    for pattern in SKILL_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        found.update(m.strip() for m in matches if m.strip())
    return list(found)[:15]


def extract_years_experience(text: str) -> Optional[int]:
    """Extract total years of experience from resume text."""
    patterns = [
        r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
        r'(\d+)\+?\s*years?\s+(?:in|with)\b',
        r'over\s+(\d+)\s+years?',
        r'(\d+)\s*\+\s*years?',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            years = int(match.group(1))
            if 0 < years < 50:
                return years
    return None


def classify_seniority(text: str, years: Optional[int]) -> str:
    text_lower = text.lower()
    if any(w in text_lower for w in ['vp ', 'vice president', 'chief ', 'cto', 'ceo', 'cfo', 'director', 'head of']):
        return 'executive'
    if any(w in text_lower for w in ['lead ', 'principal ', 'staff ', 'architect']):
        return 'lead'
    if any(w in text_lower for w in ['senior ', 'sr.', 'sr ']):
        return 'senior'
    if any(w in text_lower for w in ['junior ', 'jr.', 'jr ', 'intern', 'entry']):
        return 'entry'
    if years is not None:
        if years >= 8:
            return 'senior'
        if years >= 3:
            return 'mid'
        return 'entry'
    return 'mid'


CATEGORY_TO_TITLES = {
    'INFORMATION-TECHNOLOGY': ['software engineer', 'developer', 'IT professional', 'programmer', 'systems engineer'],
    'HR': ['HR manager', 'human resources', 'recruiter', 'talent acquisition', 'people operations'],
    'FINANCE': ['financial analyst', 'accountant', 'finance manager', 'investment analyst', 'CFO'],
    'HEALTHCARE': ['healthcare professional', 'nurse', 'doctor', 'clinical specialist', 'medical professional'],
    'TEACHER': ['teacher', 'educator', 'instructor', 'professor', 'academic'],
    'ADVOCATE': ['lawyer', 'attorney', 'advocate', 'legal counsel', 'paralegal'],
    'BUSINESS-DEVELOPMENT': ['business developer', 'sales manager', 'account executive', 'BD manager'],
    'FITNESS': ['fitness trainer', 'personal trainer', 'gym instructor', 'health coach'],
    'AGRICULTURE': ['agricultural specialist', 'agronomist', 'farm manager'],
    'BPO': ['BPO specialist', 'customer service', 'call center agent', 'support specialist'],
    'SALES': ['sales representative', 'account manager', 'sales executive', 'business development'],
    'CONSULTANT': ['consultant', 'advisor', 'management consultant', 'strategy consultant'],
    'DIGITAL-MEDIA': ['digital marketing', 'content creator', 'social media manager', 'SEO specialist'],
    'AUTOMOBILE': ['automotive engineer', 'mechanic', 'vehicle technician'],
    'CHEF': ['chef', 'cook', 'culinary artist', 'head chef', 'sous chef'],
    'ACCOUNTANT': ['accountant', 'CPA', 'auditor', 'tax professional', 'bookkeeper'],
    'CONSTRUCTION': ['construction manager', 'civil engineer', 'site manager', 'project manager'],
    'PUBLIC-RELATIONS': ['PR specialist', 'communications manager', 'media relations'],
    'BANKING': ['banker', 'financial advisor', 'bank manager', 'credit analyst', 'loan officer'],
    'ARTS': ['artist', 'graphic designer', 'creative director', 'illustrator'],
    'AVIATION': ['pilot', 'aviation specialist', 'flight attendant', 'aerospace engineer'],
    'ENGINEERING': ['engineer', 'mechanical engineer', 'electrical engineer', 'civil engineer'],
    'DESIGNER': ['designer', 'UX designer', 'product designer', 'graphic designer'],
    'APPAREL': ['fashion designer', 'apparel specialist', 'clothing designer', 'textile professional'],
}


def build_embedding_document(row: dict) -> str:
    """
    Construct a structured text for embedding. Front-load high-signal fields
    so the embedding vector is biased toward job-title semantics.
    """
    parts = []

    category = row.get('Category', '')
    titles = CATEGORY_TO_TITLES.get(category, [category])
    if titles:
        parts.append(f"ROLES: {', '.join(titles[:4])}")

    skills = row.get('skills', [])
    if skills:
        parts.append(f"SKILLS: {', '.join(skills[:10])}")

    if category:
        parts.append(f"CATEGORY: {category}")

    seniority = row.get('seniority', '')
    years = row.get('years_experience')
    if seniority and seniority != 'mid':
        parts.append(f"LEVEL: {seniority}")
    if years:
        parts.append(f"EXPERIENCE: {years} years")

    parts.append(row.get('clean_text', '')[:3500])

    return '\n'.join(parts)


def preprocess_dataset(df: pd.DataFrame) -> list[dict]:
    """Full preprocessing pipeline for a loaded dataframe."""
    records = []
    for _, row in df.iterrows():
        clean = clean_text(str(row.get('Resume_str', '')))
        skills = extract_skills_heuristic(clean)
        years = extract_years_experience(clean)
        seniority = classify_seniority(clean, years)
        record = {
            'id': str(row.get('ID', '')),
            'category': str(row.get('Category', 'Unknown')),
            'clean_text': clean,
            'skills': skills,
            'years_experience': years,
            'seniority': seniority,
        }
        record['embedding_doc'] = build_embedding_document(record)
        records.append(record)
    return records
