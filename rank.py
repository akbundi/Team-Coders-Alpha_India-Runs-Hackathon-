#!/usr/bin/env python3
"""
rank.py - Top-tier Candidate Ranking System for Redrob Hackathon
Usage: python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
import numpy as np

def fast_parse_date(dt_str):
    if not dt_str:
        return None
    try:
        parts = dt_str.split("-")
        return datetime(int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        return datetime.strptime(dt_str, "%Y-%m-%d")

# ── Regex and Lists for Fallback Feature Extraction ──────────────────────────
IMPACT_METRICS_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(%|x|percent|times|ms|milliseconds|seconds|minutes|gb|tb|requests|qps|rps|users|candidates|results|latency|accuracy|ndcg|mrr|map|f1|precision|recall)",
    re.IGNORECASE
)

CONSULTING_FIRMS = {"tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant", "capgemini"}

BUZZWORDS = {"ai", "ml", "machine learning", "deep learning", "llm", "rag", "langchain", "pinecone", "weaviate", "vector", "embedding", "nlp"}
FRAMEWORKS = {"langchain", "llamaindex", "flowise", "autogpt", "huggingface", "openai api", "langgraph"}

TUTORIAL_KWS = [
    "movie recommendation system using pandas", "titanic dataset classifier",
    "chatbot demo using langchain tutorial", "mnist handwritten digit", "housing price prediction",
    "tutorial project", "langchain chatbot demo"
]

PRODUCTION_WORDS = {
    "shipped", "deployed", "launched", "production", "prod", "real users", "served", "scale", 
    "latency", "throughput", "ndcg", "mrr", "map", "ctr", "revenue", "qps", "rps",
    "reduced latency", "improved ndcg", "served requests"
}

STARTUP_KEYWORDS = {"startup", "early-stage", "founding", "series a", "series b", "seed", "co-founder", "founding engineer"}
DELIVERY_KEYWORDS = {"ownership", "owned", "end-to-end", "delivery", "delivered", "shipped", "built", "designed", "architected", "led", "responsible", "drove", "spearheaded", "managed"}

WEAK_KEYWORDS = {"langchain", "rag", "openai", "chatbot", "gpt", "weaviate", "pinecone"}
STRONG_PHRASES = [
    "built retrieval pipeline", "built search relevance", "improved search", 
    "reduced latency", "improved mrr", "improved ndcg", "scaled database", 
    "implemented hybrid search", "deployed sentence-transformers", "optimized embeddings"
]

EVIDENCE_VERBS = {"built", "implemented", "shipped", "deployed", "scaled", "optimized", "improved", "designed", "architected"}

# TF-IDF Fallback globals
_tfidf_vectorizer_jd = None
_tfidf_matrix_jd = None
_cand_ids_map = {}
_jd_sim_dict = {}
_ideal_sim_dict = {}

# ── Skill scoring functions ───────────────────────────────────────────────────
def get_retrieval_score(cand: dict) -> float:
    skills = cand.get("skills", [])
    ret_skills = {"weaviate", "pinecone", "milvus", "qdrant", "faiss", "vector search", "dense retrieval", 
                  "elasticsearch", "opensearch", "embeddings", "embedding", "vector database", "vector store", "semantic search"}
    hits = 0.0
    has_pinecone = False
    has_qdrant = False
    for s in skills:
        name = s.get("name", "").lower()
        prof = s.get("proficiency", "beginner")
        prof_w = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.5, "beginner": 0.2}.get(prof, 0.2)
        if any(kw in name for kw in ret_skills):
            hits += prof_w
        if "pinecone" in name:
            has_pinecone = True
        if "qdrant" in name:
            has_qdrant = True
            
    # Skill Assessment Override
    sig = cand.get("redrob_signals", {})
    assessments = sig.get("skill_assessment_scores", {})
    assessment_override = 0.0
    for s_name, score in assessments.items():
        if any(kw in s_name.lower() for kw in ret_skills) and score >= 80:
            assessment_override += 0.5

    base_score = min(hits / 4.0 + assessment_override, 1.0)
    
    # Check descriptions for Pinecone and Qdrant
    career_history = cand.get("career_history", [])
    desc_text = " ".join(job.get("description", "") for job in career_history).lower()
    in_desc_pinecone = "pinecone" in desc_text
    in_desc_qdrant = "qdrant" in desc_text
    
    boost = 0.0
    if has_pinecone or in_desc_pinecone:
        boost += 0.15
    if has_qdrant or in_desc_qdrant:
        boost += 0.15
    if (has_pinecone or in_desc_pinecone) and (has_qdrant or in_desc_qdrant):
        boost += 0.20
        
    return min(base_score + boost, 1.0)

def get_ranking_score(cand: dict) -> float:
    skills = cand.get("skills", [])
    rank_skills = {"learning to rank", "learning-to-rank", "ltr", "ranking", "re-ranking", "reranking", 
                   "xgboost", "lightgbm", "bm25", "sparse retrieval", "ndcg", "mrr", "map", "search relevance"}
    hits = 0.0
    has_bm25 = False
    for s in skills:
        name = s.get("name", "").lower()
        prof = s.get("proficiency", "beginner")
        prof_w = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.5, "beginner": 0.2}.get(prof, 0.2)
        if any(kw in name for kw in rank_skills):
            hits += prof_w
        if "bm25" in name:
            has_bm25 = True
            
    # Skill Assessment Override
    sig = cand.get("redrob_signals", {})
    assessments = sig.get("skill_assessment_scores", {})
    assessment_override = 0.0
    for s_name, score in assessments.items():
        if any(kw in s_name.lower() for kw in rank_skills) and score >= 80:
            assessment_override += 0.5

    base_score = min(hits / 3.0 + assessment_override, 1.0)
    
    # Check career descriptions for BM25 and search relevance
    career_history = cand.get("career_history", [])
    desc_text = " ".join(job.get("description", "") for job in career_history).lower()
    in_desc_bm25 = "bm25" in desc_text
    has_relevance = "search relevance" in desc_text or "search-relevance" in desc_text or "relevance ranking" in desc_text
    
    boost = 0.0
    if has_bm25 or in_desc_bm25:
        boost += 0.25
    if has_relevance:
        boost += 0.15
        
    return min(base_score + boost, 1.0)

# ── TF-IDF Fallback Index Builder ─────────────────────────────────────────────
def build_tfidf_index(candidates: list):
    """Build online TF-IDF index for candidates and precalculate similarity dictionaries."""
    global _tfidf_vectorizer_jd, _tfidf_matrix_jd, _cand_ids_map, _jd_sim_dict, _ideal_sim_dict
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = []
    for idx, cand in enumerate(candidates):
        p = cand.get("profile", {})
        career = cand.get("career_history", [])
        skills = cand.get("skills", [])
        skill_text = " ".join(s.get("name", "") for s in skills)
        desc_text = " ".join(j.get("description", "") for j in career[:3])
        text = f"{p.get('current_title', '')} {p.get('headline', '')} {p.get('summary', '')} {skill_text} {desc_text}".lower()
        texts.append(text)
        _cand_ids_map[cand["candidate_id"]] = idx

    _tfidf_vectorizer_jd = TfidfVectorizer(ngram_range=(1, 2), max_features=30000, sublinear_tf=True, min_df=2)
    _tfidf_matrix_jd = _tfidf_vectorizer_jd.fit_transform(texts)

    jd_text = (
        "senior ai engineer founding team retrieval ranking embeddings embedding vector "
        "database semantic search hybrid search faiss milvus qdrant weaviate pinecone "
        "elasticsearch opensearch ndcg mrr map precision recall learning to rank ltr "
        "lightgbm xgboost bm25 sparse retrieval reranking lora qlora peft fine-tuning "
        "finetuning sentence transformers bge e5 instructor rag retrieval augmented generation "
        "production deployment scaling python pytorch tensorflow evaluation a/b testing "
        "nlp natural language processing information retrieval recommendation system "
        "vector index embedding drift latency throughput infrastructure pipeline "
        "applied ml engineer machine learning engineer search engineer nlp engineer "
        "startup product company series a founding engineer architecture system design "
        "python strong code quality mentor team lead owned scaled shipped"
    )

    ideal_text = (
        "ideal candidate profile senior ai engineer founding team retrieval systems "
        "ranking systems search relevance embeddings vector databases startup mindset "
        "product engineering production ml ownership evaluation metrics ndcg mrr map "
        "precision recall learning to rank lightgbm xgboost bm25 sparse retrieval "
        "reranking lora qlora peft fine-tuning sentence transformers bge e5 instructor "
        "rag production deployment scaling python pytorch evaluation framework a/b testing "
        "nlp natural language processing search engineer product company owned scaled shipped"
    )

    # Cosine similarity for normalized vectors is just the dot product
    jd_vec = _tfidf_vectorizer_jd.transform([jd_text.lower()])
    ideal_vec = _tfidf_vectorizer_jd.transform([ideal_text.lower()])

    jd_sims = _tfidf_matrix_jd.dot(jd_vec.T).toarray().ravel()
    ideal_sims = _tfidf_matrix_jd.dot(ideal_vec.T).toarray().ravel()

    for idx, cand in enumerate(candidates):
        cid = cand["candidate_id"]
        _jd_sim_dict[cid] = round(float(jd_sims[idx]), 4)
        _ideal_sim_dict[cid] = round(float(ideal_sims[idx]), 4)

def get_tfidf_similarity(cand_id: str, query_text: str, is_jd: bool = True) -> float:
    if is_jd:
        return _jd_sim_dict.get(cand_id, 0.5)
    else:
        return _ideal_sim_dict.get(cand_id, 0.5)

# ── Fallback Online Feature Extractor ─────────────────────────────────────────
def extract_online_features(cand: dict, current_date: datetime, jd_text: str, ideal_text: str, use_bge: bool = False, model = None) -> dict:
    cid = cand["candidate_id"]
    profile = cand.get("profile", {})
    career = cand.get("career_history", [])
    skills = cand.get("skills", [])
    education = cand.get("education", [])
    sig = cand.get("redrob_signals", {})

    # 1. Honeypots
    skill_hp = any(s.get("proficiency") in ("expert", "advanced") and s.get("duration_months", 1) == 0 for s in skills)
    date_hp = False
    for job in career:
        s_str = job.get("start_date", "")
        e_str = job.get("end_date")
        dur = job.get("duration_months", 0)
        try:
            s_dt = fast_parse_date(s_str)
            e_dt = fast_parse_date(e_str) if e_str else current_date
            if s_dt > e_dt: date_hp = True
            calc = (e_dt.year - s_dt.year) * 12 + (e_dt.month - s_dt.month)
            if abs(dur - calc) > 3: date_hp = True
        except:
            pass
    hist_months = sum(j.get("duration_months", 0) for j in career)
    years_claimed = profile.get("years_of_experience", 0)
    exp_hp = abs(years_claimed - hist_months / 12.0) > 2.0

    edu_hp = False
    if education:
        max_edu_end = max(e.get("end_year") for e in education if e.get("end_year") is not None)
        if max_edu_end:
            max_possible_exp = 2026 - max_edu_end + 1
            if years_claimed > max_possible_exp + 9.8: edu_hp = True

    is_honeypot = int(skill_hp or date_hp or exp_hp or edu_hp)

    consulting_only = int(len(career) > 0 and all(any(cf in j.get("company", "").lower() for cf in CONSULTING_FIRMS) for j in career))

    ai_months = 0
    for job in career:
        t_low = job.get("title", "").lower()
        d_low = job.get("description", "").lower()
        dur = job.get("duration_months", 0)
        if any(kw in t_low for kw in {"ai", "ml", "machine learning", "deep learning", "nlp", "retrieval", "search", "ranking", "recommendation"}):
            ai_months += dur
        elif any(kw in d_low for kw in {"retrieval", "ranking", "embedding", "vector", "nlp", "ml model", "machine learning", "llm"}):
            ai_months += dur * 0.5
    ai_years = round(ai_months / 12.0, 2)

    response_probability = sig.get("recruiter_response_rate", 0.0)

    # Activity score
    last_active_s = sig.get("last_active_date", "2020-01-01")
    try:
        last_active = fast_parse_date(last_active_s)
        days_since = (current_date - last_active).days
        activity_score = 1.0 if days_since <= 30 else (0.7 if days_since <= 90 else (0.4 if days_since <= 180 else 0.1))
    except:
        activity_score = 0.5

    # Engagement score
    views = sig.get("profile_views_received_30d", 0)
    apps = sig.get("applications_submitted_30d", 0)
    searches = sig.get("search_appearance_30d", 0)
    saves = sig.get("saved_by_recruiters_30d", 0)
    engagement_score = min((views * 2 + apps * 5 + searches * 0.5 + saves * 10) / 100.0, 1.0)

    # Openness score
    openness_score = 1.0 if sig.get("open_to_work_flag") else 0.4
    if sig.get("willing_to_relocate"): openness_score = min(openness_score + 0.2, 1.0)

    # Stability score
    if career:
        avg_tenure = sum(j.get("duration_months", 0) for j in career) / len(career)
    else:
        avg_tenure = 0
    tenure_score = min(avg_tenure / 36.0, 1.0)
    notice = sig.get("notice_period_days", 90)
    notice_score = 1.0 if notice <= 30 else (0.7 if notice <= 60 else (0.4 if notice <= 90 else 0.2))
    stability_score = 0.5 * tenure_score + 0.5 * notice_score

    # Career growth score
    progression_score = 0.5
    if len(career) > 1:
        chron_career = list(reversed(career))
        levels = []
        for j in chron_career:
            t = j.get("title", "").lower()
            if any(w in t for w in ["intern", "junior", "jr", "associate", "trainee"]): levels.append(1)
            elif any(w in t for w in ["lead", "principal", "staff", "manager", "head", "architect"]): levels.append(4)
            elif "senior" in t or "sr" in t: levels.append(3)
            else: levels.append(2)
        upward = 0
        for i in range(len(levels) - 1):
            if levels[i+1] > levels[i]: upward += 1
            elif levels[i+1] < levels[i]: upward -= 0.5
        progression_score = min(max(0.5 + upward * 0.25, 0.0), 1.0)
    career_growth_score = progression_score

    # Ownership score
    all_desc = " ".join(j.get("description", "") for j in career).lower()
    hits = sum(1 for kw in DELIVERY_KEYWORDS if kw in all_desc)
    ownership_score = min(hits / 4.0, 1.0)

    # Promotion score
    promotion_count = 0
    if len(career) > 1:
        for i in range(len(career) - 1):
            c1 = career[i].get("company", "").lower()
            c2 = career[i+1].get("company", "").lower()
            if c1 and c2 and (c1 in c2 or c2 in c1):
                t1 = career[i].get("title", "").lower()
                t2 = career[i+1].get("title", "").lower()
                def get_title_rank(t):
                    if any(w in t for w in ["lead", "principal", "staff", "manager", "head", "architect"]): return 3
                    if "senior" in t or "sr" in t: return 2
                    return 1
                if get_title_rank(t1) > get_title_rank(t2): promotion_count += 1
    promotion_score = min(promotion_count * 0.5, 1.0)

    # Production score
    metric_count = len(IMPACT_METRICS_PATTERN.findall(all_desc))
    boost = 0.0
    if re.search(r'improved\s+(?:ndcg|mrr|map|ctr|precision|recall|accuracy)\s+by?\s*\d+', all_desc): boost += 0.3
    if re.search(r'served\s+\d+\s*(?:m|million|b|billion)\s+requests', all_desc): boost += 0.3
    if re.search(r'reduced\s+latency\s+by?\s*\d+', all_desc): boost += 0.3
    if re.search(r'increased\s+(?:ctr|revenue|conversion|mrr)\s+by?\s*\d+', all_desc): boost += 0.3
    prod_hit_count = sum(1 for w in PRODUCTION_WORDS if w in all_desc)
    production_score = min(prod_hit_count * 0.05 + metric_count * 0.1 + boost, 1.0)

    # Startup mindset
    startup_pos = 0.0
    for kw in STARTUP_KEYWORDS:
        if kw in all_desc or kw in profile.get("summary", "").lower() or kw in profile.get("headline", "").lower(): startup_pos += 0.2
    for job in career:
        if job.get("company_size", "") in ["1-10", "11-50", "51-200"]: startup_pos += 0.15
    for kw in DELIVERY_KEYWORDS:
        if kw in all_desc: startup_pos += 0.1

    startup_neg = 0.0
    if consulting_only: startup_neg += 0.5
    title_low = profile.get("current_title", "").lower()
    if "architect" in title_low and not any(w in all_desc for w in ["python", "pytorch", "code", "programming"]): startup_neg += 0.3
    is_research = any(w in title_low for w in ["researcher", "research scientist", "postdoc", "phd scholar", "academic"])
    if is_research and not any(w in all_desc for w in ["shipped", "production", "deployed"]): startup_neg += 0.4
    startup_score = min(max(startup_pos - startup_neg, 0.0), 1.0)

    # Evidence Quality Score
    weak_count = sum(1 for kw in WEAK_KEYWORDS if kw in all_desc)
    strong_count = sum(1 for p in STRONG_PHRASES if p in all_desc)
    has_verb = any(v in all_desc for v in EVIDENCE_VERBS)
    evidence_quality_score = 0.2
    if has_verb: evidence_quality_score += 0.2
    evidence_quality_score += min(strong_count * 0.2, 0.6)
    if weak_count > 0 and strong_count == 0 and not has_verb: evidence_quality_score = max(evidence_quality_score - 0.2, 0.05)
    evidence_quality_score = min(evidence_quality_score, 1.0)

    # Honeypot Penalties
    keyword_stuffing_score = 1.0 if (sum(1 for bw in BUZZWORDS if bw in all_desc) >= 10 and metric_count == 0) else 0.0
    llm_tools = sum(1 for kw in {"langchain", "openai", "rag", "crewai", "autogen"} if kw in all_desc.lower())
    real_retrieval = sum(1 for kw in {"weaviate", "pinecone", "qdrant", "milvus", "bm25"} if kw in all_desc.lower())
    framework_collector_score = 1.0 if ((sum(1 for fw in FRAMEWORKS if fw in all_desc) >= 4 and strong_count == 0) or (llm_tools >= 2 and real_retrieval == 0)) else 0.0
    research_only_score = 1.0 if (is_research and not any(w in all_desc for w in ["shipped", "production", "deployed"])) else 0.0
    inflated_seniority_score = 1.0 if (years_claimed > hist_months / 12.0 + 3.0 or (years_claimed > 10 and hist_months / 12.0 < 4.0)) else 0.0
    tutorial_project_score = 1.0 if sum(1 for kw in TUTORIAL_KWS if kw in all_desc or kw in profile.get("summary", "").lower()) >= 1 else 0.0

    # Similarities
    if use_bge and model is not None:
        skill_text = " ".join(s.get("name", "") for s in skills)
        desc_text = " ".join(j.get("description", "") for j in career[:3])
        text = f"{profile.get('current_title', '')} {profile.get('headline', '')} {profile.get('summary', '')} {skill_text} {desc_text}".lower()
        cand_emb = model.encode(text, normalize_embeddings=True)
        jd_emb = model.encode(jd_text, normalize_embeddings=True)
        ideal_emb = model.encode(ideal_text, normalize_embeddings=True)
        semantic_similarity = round(float(np.dot(cand_emb, jd_emb)), 4)
        archetype_match_score = round(float(np.dot(cand_emb, ideal_emb)), 4)
    else:
        semantic_similarity = get_tfidf_similarity(cid, jd_text, is_jd=True)
        archetype_match_score = get_tfidf_similarity(cid, ideal_text, is_jd=False)

    return {
        "candidate_id":              cid,
        "is_honeypot":               is_honeypot,
        "consulting_only":           consulting_only,
        "ai_years":                  ai_years,
        "years_claimed":             years_claimed,
        "hist_years":                round(hist_months / 12.0, 2),
        "response_probability":      round(response_probability, 4),
        "activity_score":            round(activity_score, 4),
        "engagement_score":          round(engagement_score, 4),
        "openness_score":            round(openness_score, 4),
        "stability_score":           round(stability_score, 4),
        "career_growth_score":       round(career_growth_score, 4),
        "ownership_score":           round(ownership_score, 4),
        "promotion_score":           round(promotion_score, 4),
        "production_score":          round(production_score, 4),
        "startup_score":             round(startup_score, 4),
        "evidence_quality_score":    round(evidence_quality_score, 4),
        "keyword_stuffing_score":    keyword_stuffing_score,
        "framework_collector_score": framework_collector_score,
        "research_only_score":       research_only_score,
        "inflated_seniority_score":  inflated_seniority_score,
        "tutorial_project_score":    tutorial_project_score,
        "semantic_similarity":       semantic_similarity,
        "archetype_match_score":     archetype_match_score,
        "archetype_distance":        round(1.0 - archetype_match_score, 4),
    }

# ── Cohort Normalization Helper ───────────────────────────────────────────────
def normalize_features(cohort: list) -> list:
    if not cohort:
        return cohort
    keys = list(cohort[0]["features"].keys())
    min_vals = {k: float("inf") for k in keys}
    max_vals = {k: float("-inf") for k in keys}
    for item in cohort:
        for k in keys:
            val = item["features"][k]
            if val < min_vals[k]: min_vals[k] = val
            if val > max_vals[k]: max_vals[k] = val
    for item in cohort:
        for k in keys:
            min_v = min_vals[k]
            max_v = max_vals[k]
            diff = max_v - min_v
            if diff > 1e-6:
                item["features"][k] = (item["features"][k] - min_v) / diff
            else:
                item["features"][k] = 0.5
    return cohort

# ── Reasoning Generator ───────────────────────────────────────────────────────
def build_reasoning(cand: dict, entry: dict) -> str:
    profile = cand.get("profile", {})
    career = cand.get("career_history", [])
    skills = cand.get("skills", [])
    sig = cand.get("redrob_signals", {})
    
    title = profile.get("current_title", "AI/ML Engineer")
    yoe = profile.get("years_of_experience", 0)
    
    # Capitalize title words nicely, preserving known acronyms
    _acronyms = {"ai", "ml", "nlp", "cv", "dl", "ir", "llm", "sde", "sre", "devops", "ii", "iii", "iv"}
    title = " ".join([w.upper() if w.lower() in _acronyms else w.capitalize() for w in title.split()])
    if not title:
        title = "AI/ML Engineer"
    
    # 1. YOE Sentence
    yoe_sentence = f"{title} with {yoe:.1f} years experience."
    
    # 2. Retrieval Systems Sentence
    skill_names = {s.get("name", "").lower(): s.get("name") for s in skills}
    retrieval_list = [skill_names[k] for k in ["weaviate", "pinecone", "milvus", "qdrant", "faiss", "vector search", "dense retrieval", "elasticsearch", "opensearch", "embeddings"] if k in skill_names]
    
    # Check career history descriptions for retrieval skills
    career_history_desc = " ".join(job.get("description", "") for job in career).lower()
    for s_name in ["weaviate", "pinecone", "milvus", "qdrant", "faiss", "vector search", "elasticsearch", "opensearch"]:
        if s_name in career_history_desc and s_name not in [r.lower() for r in retrieval_list]:
            retrieval_list.append(s_name.capitalize() if s_name != "faiss" else "FAISS")
            
    # Remove duplicates preserving order
    seen = set()
    retrieval_list = [x for x in retrieval_list if not (x.lower() in seen or seen.add(x.lower()))]
    
    if retrieval_list:
        if len(retrieval_list) > 2:
            techs = f"{', '.join(retrieval_list[:2])}, and {retrieval_list[2]}"
        elif len(retrieval_list) == 2:
            techs = f"{retrieval_list[0]} and {retrieval_list[1]}"
        else:
            techs = retrieval_list[0]
        
        # Vary the retrieval sentence phrasing based on rank to avoid repetition
        rank_num = entry.get("rank", 1)
        _ret_templates = [
            f"Strong retrieval expertise through {techs}.",
            f"Built production retrieval systems using {techs}.",
            f"Hands-on retrieval experience with {techs}.",
            f"Demonstrated search infrastructure expertise with {techs}.",
            f"Production retrieval experience spanning {techs}.",
        ]
        retrieval_sentence = _ret_templates[rank_num % len(_ret_templates)]
    else:
        retrieval_sentence = "Demonstrated familiarity with dense retrieval concepts and semantic search."

    # 3. Ranking/Relevance/Metrics Sentence
    ranking_list = [skill_names[k] for k in ["learning to rank", "ltr", "ranking", "re-ranking", "xgboost", "lightgbm", "bm25", "sparse retrieval"] if k in skill_names]
    for s_name in ["learning to rank", "learning-to-rank", "ltr", "bm25", "reranking", "re-ranking"]:
        if s_name in career_history_desc and s_name not in [r.lower() for r in ranking_list]:
            ranking_list.append(s_name.upper() if s_name in ["ltr", "bm25"] else s_name.capitalize())
            
    seen_rank = set()
    ranking_list = [x for x in ranking_list if not (x.lower() in seen_rank or seen_rank.add(x.lower()))]

    # Metrics — build contextual, human-readable metric phrases
    all_desc = " ".join(j.get("description", "") for j in career)
    metrics = IMPACT_METRICS_PATTERN.findall(all_desc.lower())
    
    pct_metrics = [m for m in metrics if m[1].lower() in ('%', 'percent')]
    ms_metrics = [m for m in metrics if m[1].lower() in ('ms', 'milliseconds')]
    x_metrics = [m for m in metrics if m[1].lower() in ('x', 'times')]
    
    # Build a natural-language metric phrase
    metric_phrase = ""
    if ms_metrics and pct_metrics:
        metric_phrase = f"Delivered measurable production improvements, including {pct_metrics[0][0]}% relevance gains and sub-{ms_metrics[0][0]}ms latency"
    elif pct_metrics and len(pct_metrics) >= 2:
        metric_phrase = f"Delivered measurable production improvements, including {pct_metrics[0][0]}% relevance gains and {pct_metrics[1][0]}% performance improvements"
    elif pct_metrics:
        metric_phrase = f"Delivered measurable production improvements, including {pct_metrics[0][0]}% relevance gains"
    elif ms_metrics:
        metric_phrase = f"Delivered measurable production improvements, including sub-{ms_metrics[0][0]}ms latency"
    elif x_metrics:
        metric_phrase = f"Delivered measurable production improvements, including {x_metrics[0][0]}x performance gains"

    if ranking_list:
        rank_techs = ", ".join(ranking_list[:2])
        if len(ranking_list) > 1:
            rank_techs = f"{ranking_list[0]} and {ranking_list[1]}"
        else:
            rank_techs = ranking_list[0]
        
        # Vary ranking sentence phrasing
        rank_num = entry.get("rank", 1)
        _rank_base = [
            f"Strong ranking expertise through {rank_techs}.",
            f"Demonstrated search relevance expertise with {rank_techs}.",
            f"Production ranking experience with {rank_techs}.",
            f"Designed ranking systems leveraging {rank_techs}.",
            f"Applied {rank_techs} for search ranking.",
        ]
        ranking_sentence = _rank_base[rank_num % len(_rank_base)]
        if metric_phrase:
            ranking_sentence += f" {metric_phrase}."
    else:
        if metric_phrase:
            ranking_sentence = f"{metric_phrase}."
        elif entry["raw_scores"].get("production_score", 0.0) > 0.4:
            ranking_sentence = "Quantified impact on relevance and retrieval quality."
        else:
            ranking_sentence = "Exposure to search relevance algorithms and ranking systems."

    # 4. Behavioral/Progression/Stability Sentence
    notice = sig.get("notice_period_days", 90)
    resp_rate = sig.get("recruiter_response_rate", 0.0)
    
    if career:
        avg_tenure = sum(j.get("duration_months", 0) for j in career) / len(career)
    else:
        avg_tenure = 0
    tenure_raw = min(avg_tenure / 36.0, 1.0)
    notice_raw = 1.0 if notice <= 30 else (0.7 if notice <= 60 else (0.4 if notice <= 90 else 0.2))
    raw_stability = 0.5 * tenure_raw + 0.5 * notice_raw
    
    progression_score = 0.5
    if len(career) > 1:
        chron_career = list(reversed(career))
        levels = []
        for j in chron_career:
            t = j.get("title", "").lower()
            if any(w in t for w in ["intern", "junior", "jr", "associate", "trainee"]): levels.append(1)
            elif any(w in t for w in ["lead", "principal", "staff", "manager", "head", "architect"]): levels.append(4)
            elif "senior" in t or "sr" in t: levels.append(3)
            else: levels.append(2)
        upward = 0
        for i in range(len(levels) - 1):
            if levels[i+1] > levels[i]: upward += 1
            elif levels[i+1] < levels[i]: upward -= 0.5
        progression_score = min(max(0.5 + upward * 0.25, 0.0), 1.0)
    
    behavior_score = entry.get("features", {}).get("behavior_score", 0.0)
    
    # Location signal
    location = profile.get("location", "").strip()
    loc_lower = location.lower()
    preferred_loc = any(city in loc_lower for city in ["pune", "noida", "delhi", "ncr", "gurgaon", "gurugram", "hyderabad", "mumbai", "bangalore", "bengaluru"])
    loc_suffix = f" Based in {location}." if location and preferred_loc else ""
    
    if behavior_score > 0.75:
        if notice <= 15:
            notice_txt = "immediately available" if notice == 0 else f"with {notice}-day notice period"
            behavior_sentence = f"High responsiveness {notice_txt}.{loc_suffix}"
        elif notice <= 30:
            behavior_sentence = f"Good responsiveness with {notice}-day notice period.{loc_suffix}"
        else:
            # 60+ days — neutral, don't frame notice as positive
            behavior_sentence = f"Strong recruiter engagement signals.{loc_suffix}"
    elif resp_rate > 0.8 and notice <= 15:
        notice_txt = "immediately available" if notice == 0 else f"with {notice}-day notice"
        behavior_sentence = f"Demonstrates {resp_rate:.0%} recruiter response rate {notice_txt}.{loc_suffix}"
    elif resp_rate > 0.8 and notice <= 30:
        behavior_sentence = f"Demonstrates {resp_rate:.0%} recruiter response rate with {notice}-day notice.{loc_suffix}"
    elif resp_rate > 0.8:
        behavior_sentence = f"Demonstrates strong recruiter engagement rate ({resp_rate:.0%}).{loc_suffix}"
    elif resp_rate > 0.6 and notice <= 30:
        behavior_sentence = f"Good recruiter responsiveness ({resp_rate:.0%}) and available on short notice.{loc_suffix}"
    elif notice <= 15:
        notice_txt = "immediately available" if notice == 0 else f"{notice}-day notice period"
        behavior_sentence = f"Available immediately ({notice_txt}).{loc_suffix}"
    elif notice <= 30:
        behavior_sentence = f"Available on short notice ({notice}-day notice period).{loc_suffix}"
    elif progression_score > 0.75 and raw_stability > 0.75:
        behavior_sentence = f"Strong career progression with consistent tenure.{loc_suffix}"
    elif raw_stability > 0.75:
        behavior_sentence = f"Demonstrates positive career stability (avg {avg_tenure:.0f} months tenure).{loc_suffix}"
    elif progression_score > 0.75:
        behavior_sentence = f"Shows upward career progression and history of promotions.{loc_suffix}"
    elif resp_rate > 0.5:
        behavior_sentence = f"Active candidate with {resp_rate:.0%} recruiter response rate.{loc_suffix}"
    else:
        behavior_sentence = f"Positive engagement signals on the platform.{loc_suffix}"

    # 5. Concern Sentence
    eval_list = [skill_names[k].upper() for k in ["ndcg", "mrr", "map", "a/b testing"] if k in skill_names]
    has_eval_in_desc = any(term in career_history_desc for term in ["ndcg", "mrr", "map", "a/b testing", "ab testing", "evaluation framework"])
    
    features = entry.get("features", {})
    raw = entry.get("raw_scores", {})
    
    ranking_gap = 1.0
    if eval_list or has_eval_in_desc:
        ranking_gap = 0.0
    elif features.get("ranking_score", 0.0) > 0.3:
        ranking_gap = max(1.0 - features.get("ranking_score", 0.0), 0.0)
        
    # Generate multiple candidate weaknesses
    weaknesses = {
        "ownership": 1.0 - raw.get("ownership_score", 0.0),
        "startup": 1.0 - raw.get("startup_score", 0.0),
        "behavioral": 1.0 - features.get("behavior_score", 0.0),
        "ranking_metrics": ranking_gap,
        "production_scale": 1.0 - raw.get("production_score", 0.0),
        "career_growth": 1.0 - features.get("career_growth_score", 0.0)
    }

    # Pick the strongest actual weakness
    primary_weakness = max(weaknesses, key=weaknesses.get)
    
    # Concern text mapping — standard (for Concern tier)
    concern_text_map = {
        "ownership": "limited ownership of end-to-end systems",
        "startup": "limited startup experience",
        "behavioral": "moderate recruiter engagement signals",
        "ranking_metrics": "limited evidence of ranking evaluation metrics",
        "production_scale": "limited quantifiable production scale metrics",
        "career_growth": "moderate career progression velocity"
    }
    
    # Softer text mapping — for Minor concern tier (more nuanced phrasing)
    minor_concern_text_map = {
        "ownership": "ownership evidence is less extensive than other strengths",
        "startup": "limited startup experience",
        "behavioral": "recruiter engagement could be stronger",
        "ranking_metrics": "ranking evaluation metrics not prominently featured",
        "production_scale": "production scale metrics are less documented",
        "career_growth": "career progression velocity is moderate relative to experience"
    }
    
    primary_text = concern_text_map[primary_weakness]
    primary_minor_text = minor_concern_text_map[primary_weakness]
    
    # Get secondary weakness for bottom-tier candidates
    temp_weaknesses = weaknesses.copy()
    del temp_weaknesses[primary_weakness]
    secondary_weakness = max(temp_weaknesses, key=temp_weaknesses.get)
    secondary_text = concern_text_map[secondary_weakness]
    
    # Introduce concern tiers — target distribution:
    #   No major concerns: 8-12 candidates
    #   Minor concern:     25-40 candidates
    #   Concern:           50-65 candidates
    composite = entry.get("composite", 0.0)
    worst_gap = weaknesses[primary_weakness]
    rank = entry.get("rank", 100)
    
    # 1. Elite — top ~10 candidates
    if rank <= 10:
        concern_sentence = "No major concerns identified; strong overall alignment with role requirements."
    # 2. Minor concern — strong candidates (ranks ~11-40)
    elif composite >= 0.75:
        concern_sentence = f"Minor concern: {primary_minor_text}."
    # 3. Concern — remaining candidates (ranks ~41-100)
    elif composite >= 0.6:
        concern_sentence = f"Concern: {primary_text}."
    # 4. Bottom tier
    else:
        concern_sentence = f"Concern: {primary_text} and {secondary_text}."

    # Assemble reasoning
    sentences = [
        yoe_sentence,
        retrieval_sentence,
        ranking_sentence,
        behavior_sentence
    ]
    
    # Startup experience signal
    startup_kws = ["startup", "founding engineer", "seed", "series a", "early-stage", "first engineer"]
    has_startup = any(kw in all_desc.lower() for kw in startup_kws) or entry.get("raw_scores", {}).get("startup_score", 0.0) > 0.5
    if has_startup:
        if entry.get("raw_scores", {}).get("ownership_score", 0.0) > 0.6:
            sentences.append("Strong ownership in early-stage product environments.")
        else:
            sentences.append("Demonstrated startup execution experience.")
    
    # JD alignment signal — reference the actual role requirements
    ret_f = entry.get("features", {}).get("retrieval_score", 0)
    rank_f = entry.get("features", {}).get("ranking_score", 0)
    if ret_f > 0.7 and rank_f > 0.7:
        sentences.append("Closely matches the JD's core need for production retrieval and ranking expertise.")
    elif ret_f > 0.6 or rank_f > 0.6:
        sentences.append("Profile aligns with the JD's emphasis on applied search and ranking systems.")
    
    sentences.append(concern_sentence)
    
    reasoning = " ".join(sentences)
    # Smart truncation: always preserve the concern sentence (last sentence).
    # If too long, drop optional middle sentences (JD alignment, startup) first.
    if len(reasoning) > 500:
        # The concern is always the last sentence — preserve it
        # Try removing optional sentences from the middle until it fits
        core_sentences = sentences[:4]  # title, retrieval, ranking, behavior (always keep)
        optional_sentences = sentences[4:-1]  # startup, JD alignment (can drop)
        concern = sentences[-1]  # always keep
        
        # Try with all optional
        candidate = " ".join(core_sentences + optional_sentences + [concern])
        if len(candidate) > 500:
            # Drop optional sentences one by one from the end
            while optional_sentences and len(candidate) > 500:
                optional_sentences.pop()
                candidate = " ".join(core_sentences + optional_sentences + [concern])
        reasoning = candidate
    
    # Final safety truncation at sentence boundary
    if len(reasoning) > 500:
        truncated = reasoning[:500]
        last_period = truncated.rfind(".")
        if last_period > 200:
            reasoning = truncated[:last_period + 1]
        else:
            reasoning = truncated
    return reasoning

# ── Main Entry Point ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--features", default=None, help="Path to precomputed_features.csv (optional)")
    args = parser.parse_args()

    current_date = datetime(2026, 6, 15)
    base_dir = Path(args.candidates).parent

    jd_text = (
        "senior ai engineer founding team retrieval ranking embeddings embedding vector "
        "database semantic search hybrid search faiss milvus qdrant weaviate pinecone "
        "elasticsearch opensearch ndcg mrr map precision recall learning to rank ltr "
        "lightgbm xgboost bm25 sparse retrieval reranking lora qlora peft fine-tuning "
        "finetuning sentence transformers bge e5 instructor rag retrieval augmented generation "
        "production deployment scaling python pytorch tensorflow evaluation a/b testing "
        "nlp natural language processing information retrieval recommendation system "
        "vector index embedding drift latency throughput infrastructure pipeline "
        "applied ml engineer machine learning engineer search engineer nlp engineer "
        "startup product company series a founding engineer architecture system design "
        "python strong code quality mentor team lead owned scaled shipped"
    )

    ideal_text = (
        "ideal candidate profile senior ai engineer founding team retrieval systems "
        "ranking systems search relevance embeddings vector databases startup mindset "
        "product engineering production ml ownership evaluation metrics ndcg mrr map "
        "precision recall learning to rank lightgbm xgboost bm25 sparse retrieval "
        "reranking lora qlora peft fine-tuning sentence transformers bge e5 instructor "
        "rag production deployment scaling python pytorch evaluation framework a/b testing "
        "nlp natural language processing search engineer product company owned scaled shipped"
    )

    # ── Load precomputed features if available ────────────────────────────────
    precomputed = {}
    features_path = args.features or (base_dir / "precomputed_features.csv")
    if Path(features_path).exists():
        with open(features_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                precomputed[row["candidate_id"]] = row
        print(f"Loaded precomputed features for {len(precomputed)} candidates.")
    else:
        print("No precomputed features found — will use fallback mode.")

    # ── Load candidates ───────────────────────────────────────────────────────
    print("Loading candidates...")
    candidates = []
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
    print(f"Loaded {len(candidates)} candidates.")

    # ── Online Fallback Setup ─────────────────────────────────────────────────
    use_bge_online = False
    model_online = None
    if len(precomputed) == 0:
        model_dir = Path("bge-small-en-v1.5")
        if model_dir.exists():
            try:
                from sentence_transformers import SentenceTransformer
                print("Loading local BGE model weights for online fallback...")
                model_online = SentenceTransformer(str(model_dir), device="cpu")
                use_bge_online = True
            except Exception as e:
                print(f"Could not load local BGE model: {e}. Falling back to TF-IDF.")
        if not use_bge_online:
            print("Building TF-IDF indices for online fallback...")
            build_tfidf_index(candidates)

    # ── Feature Extraction & Cohort Preparation ────────────────────────────────
    print("Preparing features...")
    cohort = []
    candidates_dict = {}
    for cand in candidates:
        cid = cand["candidate_id"]
        candidates_dict[cid] = cand
        
        if cid in precomputed:
            row = precomputed[cid]
            is_hp = int(row["is_honeypot"])
            is_consult = int(row["consulting_only"])
            
            # Raw features for LTR
            features = {
                "semantic_score":      float(row["semantic_similarity"]),
                "retrieval_score":     get_retrieval_score(cand),
                "ranking_score":       get_ranking_score(cand),
                "production_score":    float(row["production_score"]),
                "behavior_score":      (0.40 * (1.0 if cand.get("redrob_signals", {}).get("notice_period_days", 90) <= 30 else 
                                                (0.7 if cand.get("redrob_signals", {}).get("notice_period_days", 90) <= 60 else 
                                                 (0.4 if cand.get("redrob_signals", {}).get("notice_period_days", 90) <= 90 else 0.2))) +
                                        0.40 * float(row["response_probability"]) +
                                        0.10 * float(row["activity_score"]) +
                                        0.10 * ((float(row["engagement_score"]) + float(row["openness_score"])) / 2.0)),
                "startup_score":       float(row["startup_score"]),
                "career_growth_score": float(row["career_growth_score"]),
                "evidence_score":      float(row["evidence_quality_score"]),
                "archetype_score":     float(row["archetype_match_score"]) - float(row["archetype_distance"]),
            }
            
            penalties = {
                "keyword_stuffing":    float(row["keyword_stuffing_score"]),
                "framework_collector": float(row["framework_collector_score"]),
                "research_only":       float(row["research_only_score"]),
                "inflated_seniority":  float(row["inflated_seniority_score"]),
                "tutorial_project":    float(row["tutorial_project_score"]),
            }
            
            raw_scores = {
                "production_score": float(row["production_score"]),
                "startup_score":    float(row["startup_score"]),
                "ownership_score":  float(row["ownership_score"]),
            }
        else:
            row = extract_online_features(cand, current_date, jd_text, ideal_text, use_bge=use_bge_online, model=model_online)
            is_hp = row["is_honeypot"]
            is_consult = row["consulting_only"]
            
            features = {
                "semantic_score":      row["semantic_similarity"],
                "retrieval_score":     get_retrieval_score(cand),
                "ranking_score":       get_ranking_score(cand),
                "production_score":    row["production_score"],
                "behavior_score":      (0.40 * (1.0 if cand.get("redrob_signals", {}).get("notice_period_days", 90) <= 30 else 
                                                (0.7 if cand.get("redrob_signals", {}).get("notice_period_days", 90) <= 60 else 
                                                 (0.4 if cand.get("redrob_signals", {}).get("notice_period_days", 90) <= 90 else 0.2))) +
                                        0.40 * row["response_probability"] +
                                        0.10 * row["activity_score"] +
                                        0.10 * ((row["engagement_score"] + row["openness_score"]) / 2.0)),
                "startup_score":       row["startup_score"],
                "career_growth_score": row["career_growth_score"],
                "evidence_score":      row["evidence_quality_score"],
                "archetype_score":     row["archetype_match_score"] - row["archetype_distance"],
            }
            
            penalties = {
                "keyword_stuffing":    row["keyword_stuffing_score"],
                "framework_collector": row["framework_collector_score"],
                "research_only":       row["research_only_score"],
                "inflated_seniority":  row["inflated_seniority_score"],
                "tutorial_project":    row["tutorial_project_score"],
            }
            
            raw_scores = {
                "production_score": row["production_score"],
                "startup_score":    row["startup_score"],
                "ownership_score":  row["ownership_score"],
            }

        cohort.append({
            "candidate_id": cid,
            "is_honeypot": is_hp,
            "consulting_only": is_consult,
            "features": features,
            "penalties": penalties,
            "raw_scores": raw_scores,
        })

    # ── Normalize features cohort-wide ────────────────────────────────────────
    print("Normalizing features...")
    cohort = normalize_features(cohort)

    # ── Final Ranking ─────────────────────────────────────────────────────────
    print("Scoring candidates...")
    scored = []
    for entry in cohort:
        cid = entry["candidate_id"]
        f = entry["features"]
        p = entry["penalties"]
        
        # Extract Deep Data signals
        cand = candidates_dict[cid]
        sig = cand.get("redrob_signals", {})
        
        # GitHub Boost
        gh_score = sig.get("github_activity_score", -1)
        github_bonus = 0.15 if gh_score > 50 else (0.05 if gh_score > 20 else 0.0)
        
        # Flake Penalty
        offer_acc = sig.get("offer_acceptance_rate", -1)
        int_comp = sig.get("interview_completion_rate", -1)
        flake_penalty = 0.0
        if offer_acc != -1 and offer_acc < 0.4: flake_penalty += 0.2
        if int_comp != -1 and int_comp < 0.5: flake_penalty += 0.2
        
        behavior_adj = max(f["behavior_score"] - flake_penalty, 0.0)

        # Composite score — weights aligned with JD priorities (retrieval+ranking heavy)
        composite = (
            0.20 * f["semantic_score"] +
            0.20 * f["retrieval_score"] +
            0.20 * f["ranking_score"] +
            0.10 * f["production_score"] +
            0.10 * behavior_adj +
            0.05 * f["startup_score"] +
            0.05 * f["career_growth_score"] +
            0.05 * f["evidence_score"] +
            0.05 * f["archetype_score"] +
            github_bonus
        )
        
        # Apply consulting-only penalty
        if entry["consulting_only"] == 1:
            composite *= 0.55
            
        # Apply honeypot penalties afterwards
        hp_penalty = sum(p.values())
        composite -= 0.15 * hp_penalty
        
        # Force hard honeypots to 0.0
        if entry["is_honeypot"] == 1:
            composite = 0.0

        scored.append({
            "candidate_id": cid,
            "composite": max(round(composite, 4), 0.0),
            "is_honeypot": entry["is_honeypot"],
            "raw_scores": entry["raw_scores"],
            "penalties": entry["penalties"],
            "consulting_only": entry["consulting_only"],
            "features": f,
        })

    # Sort by composite score descending; ties broken by candidate_id ascending
    scored.sort(key=lambda x: (-x["composite"], x["candidate_id"]))

    # Take top 100
    top100 = scored[:100]
    honeypot_count = sum(1 for s in top100 if s["is_honeypot"])
    print(f"Honeypots in top-100: {honeypot_count} (must be 0)")

    # ── Write output CSV ──────────────────────────────────────────────────────
    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, entry in enumerate(top100, start=1):
            cid = entry["candidate_id"]
            cand = candidates_dict[cid]
            entry["rank"] = rank
            reasoning = build_reasoning(cand, entry)
            score = entry["composite"]
            writer.writerow([cid, rank, f"{score:.4f}", reasoning])

    print(f"\nSubmission written to: {out_path}")
    print(f"Top-5 candidates:")
    for i, entry in enumerate(top100[:5], 1):
        cid = entry["candidate_id"]
        cand = candidates_dict[cid]
        title = cand["profile"].get("current_title", "?")
        yoe = cand["profile"].get("years_of_experience", 0)
        print(f"  #{i}: {cid} | {title} | {yoe}y | score={entry['composite']:.4f}")

if __name__ == "__main__":
    main()
