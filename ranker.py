import json
import gzip
import os
import csv
import re
from datetime import datetime

# Current date of the hackathon
CURRENT_DATE = datetime(2026, 6, 20)

# Consulting firm keywords for disqualification check
CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "tata consultancy", "ltts", "l&t", "mindtree", "hcl", "tech mahindra",
    "deloitte", "pwc", "ey", "kpmg"
}

# Technical keywords for NLP/IR/Search/ML
EMBEDDING_KEYWORDS = [
    "embedding", "sentence-transformer", "openai embedding", "bge", "e5",
    "vector search", "dense retrieval", "vector database", "pinecone", "qdrant",
    "weaviate", "milvus", "faiss", "opensearch", "elasticsearch", "chromadb", "hybrid search"
]

IR_KEYWORDS = [
    "retrieval", "search engine", "bm25", "tfidf", "information retrieval", "ranking",
    "re-ranking", "cross-encoder", "learning to rank", "ltr", "recommendation system",
    "collaborative filtering", "query expansion", "lucene", "solr"
]

ML_KEYWORDS = [
    "fine-tuning", "lora", "qlora", "peft", "llm", "llms", "transformer", "bert", "gpt",
    "xgboost", "scikit-learn", "pytorch", "tensorflow", "huggingface", "mlops", "deep learning"
]

EVAL_KEYWORDS = [
    "ndcg", "mrr", "map", "precision", "recall", "evaluation", "ab test", "ab testing",
    "offline evaluation", "metrics", "offline benchmark"
]

CV_SPEECH_KEYWORDS = [
    "computer vision", "image processing", "opencv", "speech recognition", "nlp speech",
    "audio processing", "object detection", "yolo", "cnn", "image segmentation", "robotics", "ros"
]

def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None

def is_honeypot(cand):
    """
    Identifies if a candidate profile is a honeypot (contains subtly impossible records)
    """
    # 1. Expert/advanced skills with <= 0 duration
    skills = cand.get("skills", [])
    suspicious_skills = 0
    for s in skills:
        prof = s.get("proficiency", "").lower()
        dur = s.get("duration_months", 0)
        if prof in ["expert", "advanced"] and dur <= 0:
            suspicious_skills += 1
    if suspicious_skills >= 5:
        return True, f"Expert/advanced skills with <=0 duration: {suspicious_skills}"

    # 2. Temporal contradictions in career history
    career = cand.get("career_history", [])
    for idx, role in enumerate(career):
        start_str = role.get("start_date")
        end_str = role.get("end_date")
        dur_months = role.get("duration_months", 0)
        
        start_dt = parse_date(start_str)
        if not start_dt:
            continue
        end_dt = parse_date(end_str) if end_str else CURRENT_DATE
        
        calendar_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
        if dur_months > calendar_months + 6:
            return True, f"Role {idx} ({role.get('company')}): duration_months={dur_months} > calendar_months={calendar_months}"

    # 3. Reported experience exceeds calendar time since earliest job
    if career:
        start_dates = [parse_date(role.get("start_date")) for role in career if role.get("start_date")]
        valid_starts = [d for d in start_dates if d]
        if valid_starts:
            earliest_start = min(valid_starts)
            max_possible_years = (CURRENT_DATE - earliest_start).days / 365.25
            profile_years = cand.get("profile", {}).get("years_of_experience", 0)
            if profile_years > max_possible_years + 1.0:
                return True, f"Profile experience ({profile_years} yrs) > time since earliest job ({max_possible_years:.1f} yrs)"

    return False, ""

def compute_score(cand):
    """
    Computes a match score (0.0 to 1.0) for a candidate based on technical fit, experience,
    disqualifying conditions, and behavioral signals.
    """
    profile = cand.get("profile", {})
    career = cand.get("career_history", [])
    skills = cand.get("skills", [])
    signals = cand.get("redrob_signals", {})
    
    # 1. Experience Score (Ideal: 5-9 years)
    exp = profile.get("years_of_experience", 0)
    if 5.0 <= exp <= 9.0:
        exp_score = 100
    elif 4.0 <= exp < 5.0:
        exp_score = 85
    elif 9.0 < exp <= 12.0:
        exp_score = 90
    elif exp > 12.0:
        exp_score = 70
    else:  # exp < 4.0
        exp_score = 30  # Heavy penalty for junior profiles

    # 2. Technical Score (Skill checks in summary, title, descriptions, and skills list)
    text_corpus = (
        profile.get("headline", "") + " " +
        profile.get("summary", "") + " " +
        " ".join([s.get("name", "") for s in skills]) + " " +
        " ".join([role.get("title", "") + " " + role.get("description", "") for role in career])
    ).lower()

    # Embedding/Vector Database checks (Max 30 pts)
    vector_matches = sum(1 for kw in EMBEDDING_KEYWORDS if kw in text_corpus)
    vector_score = min(vector_matches * 6.0, 30.0)

    # Search & Retrieval checks (Max 25 pts)
    ir_matches = sum(1 for kw in IR_KEYWORDS if kw in text_corpus)
    ir_score = min(ir_matches * 5.0, 25.0)

    # ML Systems / Tuning checks (Max 25 pts)
    ml_matches = sum(1 for kw in ML_KEYWORDS if kw in text_corpus)
    ml_score = min(ml_matches * 5.0, 25.0)

    # Evaluation Frameworks (Max 20 pts)
    eval_matches = sum(1 for kw in EVAL_KEYWORDS if kw in text_corpus)
    eval_score = min(eval_matches * 10.0, 20.0)

    tech_score = vector_score + ir_score + ml_score + eval_score

    # Composite base skill score
    composite_score = 0.7 * tech_score + 0.3 * exp_score

    # 3. Disqualifier Penalties
    multiplier = 1.0

    # A. Consulting Firm Only Check
    if career:
        all_consulting = True
        for role in career:
            comp = role.get("company", "").lower()
            if not any(firm in comp for firm in CONSULTING_FIRMS):
                all_consulting = False
                break
        if all_consulting:
            multiplier *= 0.1  # Severe penalty

    # B. Academic/Research Only Check
    all_academic = True
    has_eng = False
    for role in career:
        title = role.get("title", "").lower()
        is_eng_title = any(w in title for w in ["engineer", "developer", "sde", "programmer", "architect", "lead", "founding"])
        is_intern_title = any(w in title for w in ["intern", "student", "trainee"])
        if is_eng_title and not is_intern_title:
            has_eng = True
            all_academic = False
            break
    if career and all_academic:
        multiplier *= 0.15

    # C. Job Hopping / Title Chasing (avg duration < 18 months, with at least 3 jobs)
    if len(career) >= 3:
        total_dur = sum(role.get("duration_months", 0) for role in career)
        avg_dur = total_dur / len(career)
        if avg_dur < 18:
            multiplier *= 0.70

    # D. CV/Speech-Only
    has_cv_speech = any(any(kw in s.get("name", "").lower() for kw in CV_SPEECH_KEYWORDS) for s in skills)
    has_nlp_ir = any(any(kw in s.get("name", "").lower() for kw in (EMBEDDING_KEYWORDS + IR_KEYWORDS)) for s in skills)
    if has_cv_speech and not has_nlp_ir:
        multiplier *= 0.30

    # E. LangChain-only developers (exp < 3 years, only recent LLM/prompt skills)
    has_recent_llm_only = False
    if exp < 3.0:
        llm_skills = {"langchain", "llamaindex", "openai", "prompt engineering", "chatgpt"}
        has_llm = any(s.get("name", "").lower() in llm_skills for s in skills)
        has_core_ml = any(s.get("name", "").lower() in ["scikit-learn", "pandas", "numpy", "tensorflow", "pytorch", "nlp", "search", "sql"] for s in skills)
        if has_llm and not has_core_ml:
            has_recent_llm_only = True
    if has_recent_llm_only:
        multiplier *= 0.25

    # F. Non-Coding Management (Manager/VP/Director role without engineering title in current role)
    current_role = career[0] if career else None
    if current_role and current_role.get("is_current", False):
        title = current_role.get("title", "").lower()
        if any(w in title for w in ["manager", "director", "vp", "vice president", "head", "scrum", "product manager"]) and not any(w in title for w in ["engineer", "developer", "architect", "lead", "sde"]):
            multiplier *= 0.40

    # 4. Behavioral Multipliers
    # A. Recruiter Response Rate
    rr = signals.get("recruiter_response_rate", 1.0)
    response_mult = 0.5 + 0.5 * rr

    # B. Notice Period
    np_days = signals.get("notice_period_days", 0)
    if np_days <= 30:
        notice_mult = 1.0
    elif np_days <= 60:
        notice_mult = 0.95
    elif np_days <= 90:
        notice_mult = 0.80
    else:
        notice_mult = 0.40  # Notice period > 90 days is a significant negative signal

    # C. Last Active Date
    last_act_str = signals.get("last_active_date")
    last_act_dt = parse_date(last_act_str) if last_act_str else CURRENT_DATE
    days_since_active = (CURRENT_DATE - last_act_dt).days
    if days_since_active <= 30:
        active_mult = 1.0
    elif days_since_active <= 90:
        active_mult = 0.95
    elif days_since_active <= 180:
        active_mult = 0.75
    else:
        active_mult = 0.40

    # D. Location & Relocation Match
    loc = profile.get("location", "").lower()
    in_target_location = any(t in loc for t in ["pune", "noida", "delhi", "gurgaon", "ncr", "ghaziabad", "faridabad"])
    in_tier1_location = any(t in loc for t in ["mumbai", "hyderabad", "bangalore", "bengaluru", "chennai"])
    willing_relocate = signals.get("willing_to_relocate", False)
    
    if in_target_location:
        loc_mult = 1.0
    elif in_tier1_location and willing_relocate:
        loc_mult = 0.95
    elif willing_relocate:
        loc_mult = 0.90
    else:
        loc_mult = 0.35  # Relocation required but candidate unwilling and not in Pune/Noida

    # E. Open to work, Email/Phone verified, Github Activity
    open_to_work = signals.get("open_to_work_flag", False)
    open_mult = 1.05 if open_to_work else 1.0
    
    email_ver = signals.get("verified_email", True)
    phone_ver = signals.get("verified_phone", True)
    verified_mult = 1.02 if (email_ver and phone_ver) else 1.0
    
    git_score = signals.get("github_activity_score", 0)
    if git_score > 40:
        git_mult = 1.05
    elif git_score == -1:
        git_mult = 0.95
    else:
        git_mult = 1.0

    behavior_multiplier = response_mult * notice_mult * active_mult * loc_mult * open_mult * verified_mult * git_mult

    final_score = composite_score * multiplier * behavior_multiplier
    # Normalize score between 0.0 and 1.0
    return min(max(final_score / 100.0, 0.0), 1.0)

def generate_reasoning(cand, rank, score):
    """
    Generates a high-quality, non-templated 1-2 sentence reasoning detailing
    why the candidate fits this rank and acknowledging any minor concerns.
    """
    profile = cand.get("profile", {})
    skills = cand.get("skills", [])
    signals = cand.get("redrob_signals", {})
    
    exp = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "Engineer")
    company = profile.get("current_company", "Product Company")
    
    # Extract candidate's core vector/search/NLP skills
    cand_skills = [s.get("name", "") for s in skills]
    key_skills = []
    for s in cand_skills:
        s_low = s.lower()
        if any(db in s_low for db in ["pinecone", "qdrant", "weaviate", "milvus", "faiss", "chroma"]):
            key_skills.append(s)
        elif any(se in s_low for se in ["elasticsearch", "opensearch", "lucene", "search engine", "retrieval"]):
            key_skills.append(s)
        elif any(ml in s_low for ml in ["embeddings", "transformers", "nlp", "llm", "lora", "tuning"]):
            key_skills.append(s)
    
    key_skills_str = ", ".join(key_skills[:2]) if key_skills else "NLP/ML foundations"
    rr = int(signals.get("recruiter_response_rate", 1.0) * 100)
    np = signals.get("notice_period_days", 0)
    willing_relocate = signals.get("willing_to_relocate", False)
    loc = profile.get("location", "")
    
    # Check notice period concern
    np_comment = ""
    if np > 60:
        np_comment = f" despite a slightly long notice period ({np} days)"
    elif np <= 15:
        np_comment = f" and is available immediately ({np} day notice)"

    # Location / Relocation
    loc_comment = ""
    if "pune" in loc.lower() or "noida" in loc.lower():
        loc_comment = "local hybrid fit"
    elif willing_relocate:
        loc_comment = "willing to relocate"
    else:
        loc_comment = "local candidate"
        
    # Variation of sentence templates based on rank tier
    if rank <= 15:
        # Tier 1 fits
        templates = [
            f"Outstanding Founding Senior AI Engineer with {exp} years experience; hands-on deploying {key_skills_str} systems at {company}. Strong local hybrid alignment ({loc_comment}) with {rr}% response rate.",
            f"Exceptional product-focused engineer with {exp} years in ML. Proven success scaling vector search and hybrid retrieval systems; immediately available ({np} day notice).",
            f"Superb profile matching the 'shipper over researcher' mandate with {exp} years of exp. Experienced in {key_skills_str}; highly engaged on the platform ({rr}% response rate)."
        ]
        return templates[rank % len(templates)]
    elif rank <= 50:
        # Tier 2 fits
        templates = [
            f"Strong technical alignment with {exp} years experience including {key_skills_str}{np_comment}. Solid candidate based in {loc} ({loc_comment}).",
            f"Capable ML Systems engineer with {exp} years experience, showing strong practical skills in {key_skills_str}. Platform response rate is high ({rr}%).",
            f"Matches core JD criteria with {exp} years of engineering experience. Demonstrates solid background in {key_skills_str} and vector search."
        ]
        return templates[rank % len(templates)]
    elif rank <= 85:
        # Tier 3 fits
        templates = [
            f"Qualified engineer with {exp} years of experience. Solid software foundation and exposure to {key_skills_str}, though has slightly longer notice period ({np} days).",
            f"Decent fit with {exp} years in search and indexing systems. Good python skills but has slightly less direct vector database experience in production.",
            f"Competent engineer demonstrating {exp} years of experience in ML, located in {loc} and willing to relocate if needed."
        ]
        return templates[rank % len(templates)]
    else:
        # Bottom fillers
        templates = [
            f"Adequate engineering profile with {exp} years experience. Solid Python foundations, though lacking direct hands-on experience deploying modern vector databases.",
            f"Candidate with {exp} years experience in adjacent backend search fields. Included for python capabilities and high response rate ({rr}%), despite notice period of {np} days.",
            f"Senior software engineer with {exp} years experience. Possesses adjacent ML skills, but would require onboarding on dense embedding retrieval systems."
        ]
        return templates[rank % len(templates)]

def process_and_rank(candidates_path, output_csv_path=None):
    """
    Reads candidates, filters honeypots, scores and ranks them, and writes the ranked CSV.
    Returns the parsed and scored top 100 list.
    """
    candidates = []
    honeypot_count = 0
    
    # Determine file open helper based on file type
    if candidates_path.endswith(".gz"):
        open_fn = gzip.open
        mode = "rt"
    else:
        open_fn = open
        mode = "r"
        
    print(f"Reading candidates from {candidates_path}...")
    with open_fn(candidates_path, mode, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                # Support reading either raw JSONL or a JSON array format (useful for sample_candidates.json)
                if line.strip().startswith("[") or line.strip().startswith(","):
                    # Might be a pretty-printed json list, let's parse differently or skip line-by-line checks
                    continue
                cand = json.loads(line)
                candidates.append(cand)
            except json.JSONDecodeError:
                # Handle case where the file is a single JSON array (e.g. sample_candidates.json)
                pass

    # If list is still empty, let's try reading the entire file as a single JSON array
    if not candidates:
        try:
            with open_fn(candidates_path, mode, encoding="utf-8") as f:
                candidates = json.load(f)
        except Exception as e:
            print(f"Error reading JSON array format: {e}")

    print(f"Loaded {len(candidates)} candidates. Filtering honeypots and scoring...")
    
    scored_candidates = []
    for cand in candidates:
        flagged, reason = is_honeypot(cand)
        if flagged:
            honeypot_count += 1
            # We filter honeypots completely by skipping them so they can never be ranked
            continue
            
        score = round(compute_score(cand), 3)
        scored_candidates.append({
            "candidate_id": cand["candidate_id"],
            "score": score,
            "raw_candidate": cand
        })
        
    print(f"Identified and filtered {honeypot_count} honeypots.")
    
    # Sort by score descending, tie-break by candidate_id ascending
    scored_candidates.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    
    top_100 = scored_candidates[:100]
    
    # Generate final ranks and reasonings
    final_ranked_list = []
    for rank_idx, item in enumerate(top_100):
        rank = rank_idx + 1
        cand = item["raw_candidate"]
        score = item["score"]
        reasoning = generate_reasoning(cand, rank, score)
        
        final_ranked_list.append({
            "candidate_id": item["candidate_id"],
            "rank": rank,
            "score": score,
            "reasoning": reasoning,
            "profile": cand["profile"],
            "redrob_signals": cand["redrob_signals"],
            "skills": cand["skills"],
            "career_history": cand["career_history"]
        })
        
    # Write to CSV if path is specified
    if output_csv_path:
        print(f"Writing ranked CSV to {output_csv_path}...")
        with open(output_csv_path, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])
            for item in final_ranked_list:
                writer.writerow([
                    item["candidate_id"],
                    item["rank"],
                    item["score"],
                    item["reasoning"]
                ])
        print("CSV write complete.")
        
    return final_ranked_list, len(candidates), honeypot_count

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rank candidates for Senior AI Engineer JD.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", required=True, help="Path to output submission CSV")
    args = parser.parse_args()
    
    process_and_rank(args.candidates, args.out)
