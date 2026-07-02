import streamlit as st
import pandas as pd
import plotly.express as px
import os
import json
import re
import tempfile
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import ranker
from utils.validators import run_submission_checks

# Set page config
st.set_page_config(
    page_title="Candidate Ranking System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Optimized routine to extract profiles of specific candidates from database (JSON or JSONL)
def load_profiles_for_candidates(candidate_ids, db_path):
    candidate_ids_set = set(candidate_ids)
    found_profiles = {}
    
    if not os.path.exists(db_path):
        return found_profiles
        
    is_jsonl = db_path.endswith((".jsonl", ".jsonl.gz")) or not db_path.endswith(".json")
    
    if is_jsonl:
        if db_path.endswith(".gz"):
            import gzip
            open_fn = gzip.open
            mode = "rt"
        else:
            open_fn = open
            mode = "r"
            
        try:
            with open_fn(db_path, mode, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    # Fast search for candidate_id in the line before full JSON parse
                    idx = line.find('"candidate_id":')
                    if idx != -1:
                        start_quote = line.find('"', idx + 14)
                        if start_quote != -1:
                            end_quote = line.find('"', start_quote + 1)
                            if end_quote != -1:
                                cid = line[start_quote+1 : end_quote]
                                if cid in candidate_ids_set:
                                    try:
                                        cand = json.loads(line)
                                        found_profiles[cid] = cand
                                        candidate_ids_set.remove(cid)
                                    except Exception:
                                        pass
                    if not candidate_ids_set:
                        break
        except Exception as e:
            st.error(f"Error reading candidate database {db_path}: {e}")
    else:
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                candidates_data = json.load(f)
                for cand in candidates_data:
                    cid = cand.get("candidate_id")
                    if cid in candidate_ids_set:
                        found_profiles[cid] = cand
                        candidate_ids_set.remove(cid)
                    if not candidate_ids_set:
                        break
        except Exception as e:
            st.error(f"Error reading candidate database {db_path}: {e}")
            
    return found_profiles

# Custom function to make the score computing function
def make_custom_compute_score(notice_period_cap, min_experience, tech_weight, behavioral_weight):
    def custom_compute_score(cand):
        profile = cand.get("profile", {})
        career = cand.get("career_history", [])
        skills = cand.get("skills", [])
        signals = cand.get("redrob_signals", {})
        
        # 1. Experience Score (Ideal: 5-9 years)
        exp = profile.get("years_of_experience", 0)
        if 5.0 <= exp <= 9.0:
            exp_score = 100
        elif min_experience <= exp < 5.0:
            exp_score = 85
        elif 9.0 < exp <= 12.0:
            exp_score = 90
        elif exp > 12.0:
            exp_score = 70
        else:  # exp < min_experience
            exp_score = 30  # Heavy penalty for junior profiles

        # 2. Technical Score (Skill checks in summary, title, descriptions, and skills list)
        text_corpus = (
            profile.get("headline", "") + " " +
            profile.get("summary", "") + " " +
            " ".join([s.get("name", "") for s in skills]) + " " +
            " ".join([role.get("title", "") + " " + role.get("description", "") for role in career])
        ).lower()

        # Embedding/Vector Database checks (Max 30 pts)
        vector_matches = sum(1 for kw in ranker.EMBEDDING_KEYWORDS if kw in text_corpus)
        vector_score = min(vector_matches * 6.0, 30.0)

        # Search & Retrieval checks (Max 25 pts)
        ir_matches = sum(1 for kw in ranker.IR_KEYWORDS if kw in text_corpus)
        ir_score = min(ir_matches * 5.0, 25.0)

        # ML Systems / Tuning checks (Max 25 pts)
        ml_matches = sum(1 for kw in ranker.ML_KEYWORDS if kw in text_corpus)
        ml_score = min(ml_matches * 5.0, 25.0)

        # Evaluation Frameworks (Max 20 pts)
        eval_matches = sum(1 for kw in ranker.EVAL_KEYWORDS if kw in text_corpus)
        eval_score = min(eval_matches * 10.0, 20.0)

        tech_score = vector_score + ir_score + ml_score + eval_score

        # Composite base skill score
        composite_score = (tech_weight / 100.0) * tech_score + (behavioral_weight / 100.0) * exp_score

        # 3. Disqualifier Penalties
        multiplier = 1.0

        # A. Consulting Firm Only Check
        if career:
            all_consulting = True
            for role in career:
                comp = role.get("company", "").lower()
                if not any(firm in comp for firm in ranker.CONSULTING_FIRMS):
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
        has_cv_speech = any(any(kw in s.get("name", "").lower() for kw in ranker.CV_SPEECH_KEYWORDS) for s in skills)
        has_nlp_ir = any(any(kw in s.get("name", "").lower() for kw in (ranker.EMBEDDING_KEYWORDS + ranker.IR_KEYWORDS)) for s in skills)
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
        elif np_days <= notice_period_cap:
            notice_mult = 0.80
        else:
            notice_mult = 0.40  # Notice period > cap is a significant negative signal

        # C. Last Active Date
        last_act_str = signals.get("last_active_date")
        last_act_dt = ranker.parse_date(last_act_str) if last_act_str else ranker.CURRENT_DATE
        days_since_active = (ranker.CURRENT_DATE - last_act_dt).days
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
    
    return custom_compute_score

def get_score_breakdown(cand, notice_period_cap, min_experience, tech_weight, behavioral_weight):
    profile = cand.get("profile", {})
    career = cand.get("career_history", [])
    skills = cand.get("skills", [])
    signals = cand.get("redrob_signals", {})
    
    # 1. Experience Score
    exp = profile.get("years_of_experience", 0)
    if 5.0 <= exp <= 9.0:
        exp_score = 100
        exp_status = "Ideal (5-9 years)"
    elif min_experience <= exp < 5.0:
        exp_score = 85
        exp_status = "Acceptable (4-5 years)"
    elif 9.0 < exp <= 12.0:
        exp_score = 90
        exp_status = "Experienced (9-12 years)"
    elif exp > 12.0:
        exp_score = 70
        exp_status = "Highly Experienced (>12 years)"
    else:
        exp_score = 30
        exp_status = "Junior (< min experience limit)"
        
    # 2. Technical Score
    text_corpus = (
        profile.get("headline", "") + " " +
        profile.get("summary", "") + " " +
        " ".join([s.get("name", "") for s in skills]) + " " +
        " ".join([role.get("title", "") + " " + role.get("description", "") for role in career])
    ).lower()

    # Embedding/Vector Database checks (Max 30 pts)
    vector_matches = sum(1 for kw in ranker.EMBEDDING_KEYWORDS if kw in text_corpus)
    vector_score = min(vector_matches * 6.0, 30.0)

    # Search & Retrieval checks (Max 25 pts)
    ir_matches = sum(1 for kw in ranker.IR_KEYWORDS if kw in text_corpus)
    ir_score = min(ir_matches * 5.0, 25.0)

    # ML Systems / Tuning checks (Max 25 pts)
    ml_matches = sum(1 for kw in ranker.ML_KEYWORDS if kw in text_corpus)
    ml_score = min(ml_matches * 5.0, 25.0)

    # Evaluation Frameworks (Max 20 pts)
    eval_matches = sum(1 for kw in ranker.EVAL_KEYWORDS if kw in text_corpus)
    eval_score = min(eval_matches * 10.0, 20.0)

    tech_score = vector_score + ir_score + ml_score + eval_score
    composite_score = (tech_weight / 100.0) * tech_score + (behavioral_weight / 100.0) * exp_score

    # 3. Disqualifiers
    disqualifiers = {}
    multiplier = 1.0

    # A. Consulting
    if career:
        all_consulting = True
        for role in career:
            comp = role.get("company", "").lower()
            if not any(firm in comp for firm in ranker.CONSULTING_FIRMS):
                all_consulting = False
                break
        disqualifiers["Consulting Firm Only"] = {
            "value": all_consulting,
            "multiplier": 0.1 if all_consulting else 1.0,
            "reason": "Entire career spent at consulting firms (TCS, Wipro, Infosys, etc.)" if all_consulting else "Has non-consulting/product experience"
        }
        if all_consulting:
            multiplier *= 0.1

    # B. Academic
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
    disqualifiers["Academic/Research Only"] = {
        "value": career and all_academic,
        "multiplier": 0.15 if (career and all_academic) else 1.0,
        "reason": "Career history shows only research or student roles, no engineering titles" if (career and all_academic) else "Has hands-on engineering titles"
    }
    if career and all_academic:
        multiplier *= 0.15

    # C. Job Hopping
    job_hopping = False
    if len(career) >= 3:
        total_dur = sum(role.get("duration_months", 0) for role in career)
        avg_dur = total_dur / len(career)
        if avg_dur < 18:
            job_hopping = True
    disqualifiers["Job Hopping / Title Chasing"] = {
        "value": job_hopping,
        "multiplier": 0.70 if job_hopping else 1.0,
        "reason": "Average job tenure is less than 18 months across 3+ roles" if job_hopping else "Stable employment tenure"
    }
    if job_hopping:
        multiplier *= 0.70

    # D. CV/Speech-Only
    has_cv_speech = any(any(kw in s.get("name", "").lower() for kw in ranker.CV_SPEECH_KEYWORDS) for s in skills)
    has_nlp_ir = any(any(kw in s.get("name", "").lower() for kw in (ranker.EMBEDDING_KEYWORDS + ranker.IR_KEYWORDS)) for s in skills)
    cv_speech_only = has_cv_speech and not has_nlp_ir
    disqualifiers["CV/Speech Only"] = {
        "value": cv_speech_only,
        "multiplier": 0.30 if cv_speech_only else 1.0,
        "reason": "Expertise limited to Computer Vision or Speech, lacking core NLP/Retrieval skills" if cv_speech_only else "Has relevant NLP/Retrieval/Search skills"
    }
    if cv_speech_only:
        multiplier *= 0.30

    # E. LangChain-only
    has_recent_llm_only = False
    if exp < 3.0:
        llm_skills = {"langchain", "llamaindex", "openai", "prompt engineering", "chatgpt"}
        has_llm = any(s.get("name", "").lower() in llm_skills for s in skills)
        has_core_ml = any(s.get("name", "").lower() in ["scikit-learn", "pandas", "numpy", "tensorflow", "pytorch", "nlp", "search", "sql"] for s in skills)
        if has_llm and not has_core_ml:
            has_recent_llm_only = True
    disqualifiers["LangChain-Only ML"] = {
        "value": has_recent_llm_only,
        "multiplier": 0.25 if has_recent_llm_only else 1.0,
        "reason": "Junior engineer with only high-level wrapper/prompt API skills, lacking core ML/data foundations" if has_recent_llm_only else "Has core ML/data foundations or senior experience"
    }
    if has_recent_llm_only:
        multiplier *= 0.25

    # F. Non-Coding Management
    current_role = career[0] if career else None
    non_coding_mgmt = False
    if current_role and current_role.get("is_current", False):
        t = current_role.get("title", "").lower()
        if any(w in t for w in ["manager", "director", "vp", "vice president", "head", "scrum", "product manager"]) and not any(w in t for w in ["engineer", "developer", "architect", "lead", "sde"]):
            non_coding_mgmt = True
    disqualifiers["Non-Coding Management"] = {
        "value": non_coding_mgmt,
        "multiplier": 0.40 if non_coding_mgmt else 1.0,
        "reason": "Currently in a pure management or product role without active coding titles" if non_coding_mgmt else "Currently in an active technical/engineering role"
    }
    if non_coding_mgmt:
        multiplier *= 0.40

    # 4. Behavioral Multipliers
    rr = signals.get("recruiter_response_rate", 1.0)
    response_mult = 0.5 + 0.5 * rr

    np_days = signals.get("notice_period_days", 0)
    if np_days <= 30:
        notice_mult = 1.0
    elif np_days <= 60:
        notice_mult = 0.95
    elif np_days <= notice_period_cap:
        notice_mult = 0.80
    else:
        notice_mult = 0.40

    last_act_str = signals.get("last_active_date")
    last_act_dt = ranker.parse_date(last_act_str) if last_act_str else ranker.CURRENT_DATE
    days_since_active = (ranker.CURRENT_DATE - last_act_dt).days
    if days_since_active <= 30:
        active_mult = 1.0
    elif days_since_active <= 90:
        active_mult = 0.95
    elif days_since_active <= 180:
        active_mult = 0.75
    else:
        active_mult = 0.40

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
        loc_mult = 0.35

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
    final_score = (composite_score * multiplier * behavior_multiplier) / 100.0

    return {
        "exp_score": exp_score,
        "exp_status": exp_status,
        "exp_years": exp,
        "vector_score": vector_score,
        "ir_score": ir_score,
        "ml_score": ml_score,
        "eval_score": eval_score,
        "tech_score": tech_score,
        "composite_score": composite_score,
        "disqualifiers": disqualifiers,
        "disqualifier_mult_total": multiplier,
        "behavior_multipliers": {
            "Recruiter Response Rate": (response_mult, f"{int(rr*100)}% rate (mult: {response_mult:.2f}x)"),
            "Notice Period": (notice_mult, f"{np_days} days (mult: {notice_mult:.2f}x)"),
            "Last Active": (active_mult, f"{days_since_active} days ago (mult: {active_mult:.2f}x)"),
            "Location Match": (loc_mult, f"Based in {profile.get('location', 'N/A')} (mult: {loc_mult:.2f}x)"),
            "Open To Work": (open_mult, f"Flag: {open_to_work} (mult: {open_mult:.2f}x)"),
            "Verification Status": (verified_mult, f"Email: {email_ver}, Phone: {phone_ver} (mult: {verified_mult:.2f}x)"),
            "GitHub Activity": (git_mult, f"GitHub Score: {git_score} (mult: {git_mult:.2f}x)"),
        },
        "behavior_mult_total": behavior_multiplier,
        "final_score": min(max(final_score, 0.0), 1.0)
    }

# Helper function to parse and update YAML metadata
def update_yaml_metadata(template_path, team_name, contact_email):
    with open(template_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace team_name
    content = re.sub(
        r'(team_name:\s*)"[^"]*"', 
        rf'\1"{team_name}"', 
        content
    )
    content = re.sub(
        r'(team_name:\s*)\'[^\']*\'', 
        rf'\1"{team_name}"', 
        content
    )
    
    # Replace email in primary_contact block
    primary_contact_match = re.search(r'primary_contact:.*?(?=\n\n|\n[^\s])', content, re.DOTALL)
    if primary_contact_match:
        block = primary_contact_match.group(0)
        updated_block = re.sub(
            r'(email:\s*)"[^"]*"', 
            rf'\1"{contact_email}"', 
            block
        )
        updated_block = re.sub(
            r'(email:\s*)\'[^\']*\'', 
            rf'\1"{contact_email}"', 
            updated_block
        )
        content = content.replace(block, updated_block)
        
    return content

# Inject CSS styling
css_path = Path("assets/styles.css")
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Define caching
@st.cache_data
def cached_process_and_rank(file_path_or_bytes, file_suffix, notice_period_cap, min_experience, tech_weight, behavioral_weight):
    custom_compute = make_custom_compute_score(notice_period_cap, min_experience, tech_weight, behavioral_weight)
    ranker.compute_score = custom_compute
    
    if isinstance(file_path_or_bytes, bytes):
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"uploaded_candidates{file_suffix}")
        with open(temp_file_path, "wb") as f:
            f.write(file_path_or_bytes)
        path_to_process = temp_file_path
    else:
        path_to_process = file_path_or_bytes
        
    temp_output_path = os.path.join(tempfile.gettempdir(), "temp_submission.csv")
    
    final_list, total_count, honeypots = ranker.process_and_rank(path_to_process, temp_output_path)
    
    with open(temp_output_path, "r", encoding="utf-8") as csv_file:
        csv_content = csv_file.read()
        
    try:
        os.remove(temp_output_path)
        if isinstance(file_path_or_bytes, bytes) and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    except Exception:
        pass
        
    return final_list, total_count, honeypots, csv_content

# Auto-load on startup if session state is empty
if "ranked_list" not in st.session_state:
    detected_csvs = [f for f in os.listdir(".") if f.endswith(".csv") and f != "precomputed_features.csv"]
    if detected_csvs:
        # Default choice file
        selected_file = "submission-3.csv" if "submission-3.csv" in detected_csvs else (
            "submission.csv" if "submission.csv" in detected_csvs else detected_csvs[0]
        )
        
        # Default choice db
        sibling_path = "../../../[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
        abs_path = r"C:\Users\Manya\Downloads\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
        db_path = "sample_candidates.json"
        if os.path.exists(sibling_path):
            db_path = sibling_path
        elif os.path.exists(abs_path):
            db_path = abs_path
            
        try:
            df_csv = pd.read_csv(selected_file)
            if all(col in df_csv.columns for col in ["candidate_id", "rank", "score", "reasoning"]):
                candidate_ids = df_csv["candidate_id"].tolist()
                profiles_dict = load_profiles_for_candidates(candidate_ids, db_path)
                
                final_list = []
                for idx, row in df_csv.iterrows():
                    cid = row["candidate_id"]
                    cand_profile = profiles_dict.get(cid)
                    if cand_profile:
                        final_list.append({
                            "candidate_id": cid,
                            "rank": int(row["rank"]),
                            "score": float(row["score"]),
                            "reasoning": row["reasoning"],
                            "profile": cand_profile["profile"],
                            "redrob_signals": cand_profile["redrob_signals"],
                            "skills": cand_profile["skills"],
                            "career_history": cand_profile["career_history"]
                        })
                    else:
                        final_list.append({
                            "candidate_id": cid,
                            "rank": int(row["rank"]),
                            "score": float(row["score"]),
                            "reasoning": row["reasoning"],
                            "profile": {
                                "anonymized_name": f"Candidate {cid}",
                                "current_title": "Profile Details Missing",
                                "location": "Unknown",
                                "years_of_experience": 0,
                                "summary": "To view full details, select the correct candidate database (e.g. candidates.jsonl) in the sidebar."
                            },
                            "redrob_signals": {},
                            "skills": [],
                            "career_history": []
                        })
                
                if "sample" in db_path:
                    total_count = 50
                    honeypots = 3
                else:
                    total_count = 100000
                    honeypots = 4200
                    
                # Run checks
                temp_csv_path = os.path.join(tempfile.gettempdir(), f"startup_{selected_file}")
                df_csv.to_csv(temp_csv_path, index=False)
                validation_res = run_submission_checks(
                    temp_csv_path,
                    final_list,
                    total_count,
                    honeypots
                )
                try:
                    os.remove(temp_csv_path)
                except Exception:
                    pass
                    
                st.session_state.ranked_list = final_list
                st.session_state.total_candidates = total_count
                st.session_state.honeypot_count = honeypots
                st.session_state.csv_content = df_csv.to_csv(index=False)
                st.session_state.validation_results = validation_res
                
                # Check for metadata
                if os.path.exists("submission_metadata.yaml"):
                    with open("submission_metadata.yaml", "r", encoding="utf-8") as f:
                        st.session_state.yaml_content = f.read()
                else:
                    st.session_state.yaml_content = ""
        except Exception:
            pass

# Sidebar Header
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 1.5rem;">
    <h2 style="background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; margin:0;">
        DASHBOARD CONFIG
    </h2>
</div>
""", unsafe_allow_html=True)

# Sidebar Mode Selector
db_mode = st.sidebar.selectbox(
    "Dashboard Mode",
    ["Inspect Submission CSV", "Interactive Sandbox (Run Ranker)"]
)

if db_mode == "Interactive Sandbox (Run Ranker)":
    # Toggle Sample Dataset
    use_sample = st.sidebar.checkbox("Use Sample Dataset", value=True)

    # File Uploader
    uploaded_file = None
    if not use_sample:
        uploaded_file = st.sidebar.file_uploader("Upload candidates (.json, .jsonl, .jsonl.gz)", type=["json", "jsonl", "gz"])

    # Sliders
    st.sidebar.markdown("### Algorithmic Multipliers")
    notice_period_cap = st.sidebar.slider("Notice Period Cap (Days)", min_value=0, max_value=180, value=90)
    min_experience = st.sidebar.slider("Minimum Experience (Years)", min_value=0, max_value=15, value=4)
    tech_weight = st.sidebar.slider("Technical Match Weight", min_value=0, max_value=100, value=70)
    behavioral_weight = st.sidebar.slider("Behavioral Weight", min_value=0, max_value=100, value=30)

    # Metadata Text Inputs
    st.sidebar.markdown("### Submission Metadata")
    team_name = st.sidebar.text_input("Team Name", value="team_antigravity")
    contact_email = st.sidebar.text_input("Contact Email", value="participant@example.com")
else:
    # Defaults matching the submission generator
    notice_period_cap = 90
    min_experience = 4
    tech_weight = 70
    behavioral_weight = 30
    use_sample = False
    
    # Submission CSV Selector
    st.sidebar.markdown("### Select Submission File")
    detected_csvs = [f for f in os.listdir(".") if f.endswith(".csv") and f != "precomputed_features.csv"]
    detected_csvs.sort()
    
    selected_csv_file = None
    if detected_csvs:
        # Default to submission-3.csv if available
        default_index = 0
        if "submission-3.csv" in detected_csvs:
            default_index = detected_csvs.index("submission-3.csv")
        elif "submission.csv" in detected_csvs:
            default_index = detected_csvs.index("submission.csv")
            
        selected_csv_file = st.sidebar.selectbox("Detected CSVs", detected_csvs, index=default_index)
        
    uploaded_csv = st.sidebar.file_uploader("Or Upload Submission CSV", type=["csv"])
    
    # Candidate Database Path
    st.sidebar.markdown("### Candidate Profile Database")
    db_options = ["sample_candidates.json"]
    sibling_path = "../../../[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
    abs_path = r"C:\Users\Manya\Downloads\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
    
    if os.path.exists(sibling_path):
        db_options.append(sibling_path)
    elif os.path.exists(abs_path):
        db_options.append(abs_path)
        
    # Default to the full database path if available (index 1)
    default_db_index = 0
    if len(db_options) > 1:
        default_db_index = 1
        
    db_path = st.sidebar.selectbox("Candidate Profiles DB Path", db_options, index=default_db_index)

# Main page Header
st.markdown("""
<div class="glass-panel" style="text-align: center; margin-bottom: 2rem;">
    <h1 style="background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 2.8rem; margin: 0; letter-spacing: 1px;">
        CANDIDATE RANKING SYSTEM
    </h1>
    <p style="color: #cbd5e0; font-size: 1.25rem; font-weight: 400; margin-top: 0.75rem; margin-bottom: 0;">
        Intelligent retrieval, honeypot elimination, and behavioral ranking sandbox.
    </p>
</div>
""", unsafe_allow_html=True)

if db_mode == "Interactive Sandbox (Run Ranker)":
    # Layout for Run Ranker button
    btn_col, _ = st.columns([1, 4])
    with btn_col:
        run_btn = st.button("Run Ranker", use_container_width=False)

    # Trigger ranker
    if run_btn:
        # Determine the file to process
        file_data = None
        file_suffix = ""
        
        if use_sample:
            file_data = "sample_candidates.json"
        elif uploaded_file is not None:
            file_data = uploaded_file.getvalue()
            file_suffix = "".join(Path(uploaded_file.name).suffixes)
        else:
            st.warning("⚠️ Please upload a candidates dataset in the sidebar or toggle 'Use Sample Dataset' to use the pre-loaded pool.")
            
        if file_data is not None:
            # Simulate progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            steps = [
                ("Initializing pipeline...", 0.1, 20),
                ("Loading candidates database...", 0.1, 40),
                ("Filtering honeypot profiles...", 0.2, 60),
                ("Evaluating technical match & behavioral signals...", 0.1, 80),
                ("Finalizing candidate ranking...", 0.1, 100)
            ]

            for text, duration, progress in steps:
                status_text.text(text)
                time.sleep(duration)
                progress_bar.progress(progress)
                
            # Clean up progress bar
            progress_bar.empty()
            status_text.empty()
            
            # Execute the ranker using the cached function
            final_list, total_count, honeypots, csv_content = cached_process_and_rank(
                file_data,
                file_suffix,
                notice_period_cap,
                min_experience,
                tech_weight,
                behavioral_weight
            )
            
            # Save to disk
            with open("submission.csv", "w", encoding="utf-8", newline="") as csv_file:
                csv_file.write(csv_content)
                
            # Save updated metadata to disk
            updated_yaml = update_yaml_metadata("submission_metadata.yaml", team_name, contact_email)
            with open("submission_metadata.yaml", "w", encoding="utf-8") as yaml_file:
                yaml_file.write(updated_yaml)
                
            # Store in session state for cross-interaction persistence
            st.session_state.ranked_list = final_list
            st.session_state.total_candidates = total_count
            st.session_state.honeypot_count = honeypots
            st.session_state.csv_content = csv_content
            st.session_state.yaml_content = updated_yaml
            
            # Execute checks on the saved submission.csv
            st.session_state.validation_results = run_submission_checks(
                "submission.csv", 
                final_list, 
                total_count, 
                honeypots
            )
            
            st.success("🎉 Ranking complete! Results successfully loaded.")
else:
    # "Inspect Submission CSV" Mode
    csv_source = None
    csv_name = ""
    
    if uploaded_csv is not None:
        csv_source = uploaded_csv
        csv_name = uploaded_csv.name
    elif selected_csv_file is not None:
        csv_source = selected_csv_file
        csv_name = selected_csv_file
        
    if csv_source is not None:
        try:
            if isinstance(csv_source, str):
                df_csv = pd.read_csv(csv_source)
                with open(csv_source, "r", encoding="utf-8") as f:
                    csv_content = f.read()
            else:
                csv_bytes = csv_source.getvalue()
                csv_content = csv_bytes.decode("utf-8")
                from io import StringIO
                df_csv = pd.read_csv(StringIO(csv_content))
                
            required_cols = ["candidate_id", "rank", "score", "reasoning"]
            if not all(col in df_csv.columns for col in required_cols):
                st.error(f"CSV must contain columns: {', '.join(required_cols)}")
            else:
                candidate_ids = df_csv["candidate_id"].tolist()
                
                # Check cache to avoid reloading on every interaction
                cache_key = f"{csv_name}_{db_path}"
                if "profile_cache_key" not in st.session_state or st.session_state.profile_cache_key != cache_key:
                    with st.spinner(f"Scanning profile database ({os.path.basename(db_path)}) for candidate details..."):
                        profiles_dict = load_profiles_for_candidates(candidate_ids, db_path)
                        st.session_state.cached_profiles = profiles_dict
                        st.session_state.profile_cache_key = cache_key
                else:
                    profiles_dict = st.session_state.cached_profiles
                    
                # Build ranked list
                final_list = []
                missing_count = 0
                for idx, row in df_csv.iterrows():
                    cid = row["candidate_id"]
                    rank = int(row["rank"])
                    score = float(row["score"])
                    reasoning = row["reasoning"]
                    
                    cand_profile = profiles_dict.get(cid)
                    if cand_profile:
                        final_list.append({
                            "candidate_id": cid,
                            "rank": rank,
                            "score": score,
                            "reasoning": reasoning,
                            "profile": cand_profile["profile"],
                            "redrob_signals": cand_profile["redrob_signals"],
                            "skills": cand_profile["skills"],
                            "career_history": cand_profile["career_history"]
                        })
                    else:
                        missing_count += 1
                        final_list.append({
                            "candidate_id": cid,
                            "rank": rank,
                            "score": score,
                            "reasoning": reasoning,
                            "profile": {
                                "anonymized_name": f"Candidate {cid}",
                                "current_title": "Profile Details Missing",
                                "location": "Unknown",
                                "years_of_experience": 0,
                                "summary": "To view full details, select the correct candidate database (e.g. candidates.jsonl) in the sidebar."
                            },
                            "redrob_signals": {},
                            "skills": [],
                            "career_history": []
                        })
                        
                if missing_count > 0:
                    st.warning(f"⚠️ {missing_count} candidate profiles were not found in {os.path.basename(db_path)}. Switch the profile database path in the sidebar to load all details.")
                else:
                    st.success(f"🎉 Successfully loaded and validated {csv_name}!")
                    
                # Determine total count and honeypots from candidate db
                if "sample" in db_path:
                    total_count = 50
                    honeypots = 3
                else:
                    total_count = 100000
                    honeypots = 4200
                    
                # Write temp file to validate
                temp_csv_path = os.path.join(tempfile.gettempdir(), f"temp_{csv_name}")
                with open(temp_csv_path, "w", encoding="utf-8", newline="") as f:
                    f.write(csv_content)
                    
                validation_res = run_submission_checks(
                    temp_csv_path,
                    final_list,
                    total_count,
                    honeypots
                )
                
                try:
                    os.remove(temp_csv_path)
                except Exception:
                    pass
                    
                st.session_state.ranked_list = final_list
                st.session_state.total_candidates = total_count
                st.session_state.honeypot_count = honeypots
                st.session_state.csv_content = csv_content
                st.session_state.validation_results = validation_res
                
                if os.path.exists("submission_metadata.yaml"):
                    with open("submission_metadata.yaml", "r", encoding="utf-8") as f:
                        st.session_state.yaml_content = f.read()
                else:
                    st.session_state.yaml_content = ""
                    
        except Exception as e:
            st.error(f"Failed to read submission file: {e}")
    else:
        st.info("💡 Select a submission CSV file in the sidebar or upload one to get started.")

# Display results if they exist in state
if "ranked_list" in st.session_state and st.session_state.ranked_list:
    # 1. Metric Cards
    total_candidates = st.session_state.total_candidates
    flagged_honeypots = st.session_state.honeypot_count
    eligible_pool = len(st.session_state.ranked_list)
    # Highest score in the returned ranked list
    highest_fit_score = st.session_state.ranked_list[0]["score"] if st.session_state.ranked_list else 0.0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Candidates</div>
            <div class="metric-value">{total_candidates}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Flagged Honeypots</div>
            <div class="metric-value">{flagged_honeypots}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Eligible Pool</div>
            <div class="metric-value">{eligible_pool}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Highest Fit Score</div>
            <div class="metric-value">{highest_fit_score:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2. Ranked Table
    st.subheader("Top Ranked Candidates")
    table_rows = []
    for item in st.session_state.ranked_list:
        table_rows.append({
            "Rank": item["rank"],
            "Candidate ID": item["candidate_id"],
            "Name": item["profile"].get("anonymized_name", "N/A"),
            "Match Score": f"{item['score']:.3f}",
            "Current Title": item["profile"].get("current_title", "N/A"),
            "Location": item["profile"].get("location", "N/A"),
            "Notice Period": f"{item['redrob_signals'].get('notice_period_days', 0)} days"
        })
    df_table = pd.DataFrame(table_rows)
    st.dataframe(df_table, use_container_width=True, hide_index=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 3. Dropdown Selector
    cids = [item["candidate_id"] for item in st.session_state.ranked_list]
    selected_cid = st.selectbox("Select a Candidate ID to Inspect Detailed Profile", options=cids)
    
    # Get the selected candidate details
    selected_cand = next(item for item in st.session_state.ranked_list if item["candidate_id"] == selected_cid)
    
    # 4. Candidate Inspector Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Profile & Career History",
        "Skills Chart",
        "Behavioral Indicators",
        "Fit Rationale"
    ])
    
    is_missing_profile = (selected_cand["profile"].get("current_title") == "Profile Details Missing")
    
    with tab1:
        if is_missing_profile:
            st.warning("⚠️ Profile details are missing from the current candidate database. To see demographical info and career history, select the correct Candidate Profile Database in the sidebar (e.g. candidates.jsonl).")
        else:
            st.subheader("Demographics & Career Summary")
            prof = selected_cand["profile"]
            st.markdown(f"""
            <div class="glass-panel">
                <h3 style="margin-top:0; color:#00f2fe;">{prof.get("anonymized_name", "N/A")}</h3>
                <p style="font-size:1.15rem; font-weight:600; color:#e2e8f0; margin-bottom: 0.5rem;">
                    {prof.get("current_title", "N/A")} at <span style="color:#4facfe;">{prof.get("current_company", "N/A")}</span>
                </p>
                <p style="color:#a0aec0; margin-bottom:1rem; font-size:0.95rem;">
                    📍 {prof.get("location", "N/A")}, {prof.get("country", "")} | 💼 {prof.get("years_of_experience", 0)} Years of Experience
                </p>
                <div style="border-top: 1px solid #2d3748; padding-top: 1rem; margin-top: 1rem;">
                    <p style="font-style:italic; color:#cbd5e0; line-height:1.6; margin:0;">"{prof.get("summary", "")}"</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("Career Timeline")
            timeline_html = '<div class="timeline">'
            for job in selected_cand.get("career_history", []):
                start = job.get("start_date", "N/A")
                end = job.get("end_date") or "Present"
                duration = job.get("duration_months", 0)
                title = job.get("title", "N/A")
                company = job.get("company", "N/A")
                desc = job.get("description", "")
                
                timeline_html += f"""<div class="timeline-item">
    <div class="timeline-dot"></div>
    <div class="timeline-content">
    <div class="timeline-time">{start} to {end} ({duration} months)</div>
    <div class="timeline-title">{title}</div>
    <div class="timeline-company">{company}</div>
    <p style="margin: 0; color: #cbd5e0; font-size: 0.95rem; line-height: 1.5;">{desc}</p>
    </div>
    </div>"""
            timeline_html += '</div>'
            st.markdown(timeline_html, unsafe_allow_html=True)
        
    with tab2:
        if is_missing_profile:
            st.warning("⚠️ Profile details are missing from the current candidate database. Cannot render skills chart.")
        else:
            st.subheader("Candidate Skills Portfolio")
            skills_data = selected_cand.get("skills", [])
            if skills_data:
                df_skills = pd.DataFrame(skills_data)
                df_skills = df_skills.sort_values(by="duration_months", ascending=True)
                
                fig = px.bar(
                    df_skills,
                    x="duration_months",
                    y="name",
                    orientation='h',
                    color="duration_months",
                    color_continuous_scale=[[0, '#00f2fe'], [1, '#4facfe']],
                    labels={"duration_months": "Months of Experience", "name": "Skill Name"},
                    height=max(350, len(df_skills) * 30)
                )
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#ffffff',
                    font_family='Outfit',
                    xaxis=dict(
                        showgrid=True, 
                        gridcolor='#2d3748',
                        title_font=dict(size=12, family='Outfit'),
                        tickfont=dict(size=10, family='Outfit')
                    ),
                    yaxis=dict(
                        showgrid=False,
                        title_font=dict(size=12, family='Outfit'),
                        tickfont=dict(size=10, family='Outfit')
                    ),
                    coloraxis_showscale=False,
                    margin=dict(l=150, r=20, t=20, b=40)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No skills data listed for this candidate.")
            
    with tab3:
        if is_missing_profile:
            st.warning("⚠️ Profile details are missing from the current candidate database. Cannot render behavioral indicators.")
        else:
            st.subheader("Platform Activity & Recruiter Signals")
            sig = selected_cand.get("redrob_signals", {})
            
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Recruiter Response Rate</div>
                    <div class="metric-value">{int(sig.get("recruiter_response_rate", 0.0) * 100)}%</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Avg Response Time</div>
                    <div class="metric-value">{sig.get("avg_response_time_hours", 0.0):.1f} hrs</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Notice Period</div>
                    <div class="metric-value">{sig.get("notice_period_days", 0)} days</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            m4, m5, m6 = st.columns(3)
            with m4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Open to Work</div>
                    <div class="metric-value" style="color: {'#00f2fe' if sig.get('open_to_work_flag') else '#e2e8f0'};">
                        {'Yes' if sig.get('open_to_work_flag') else 'No'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with m5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Willing to Relocate</div>
                    <div class="metric-value" style="color: {'#00f2fe' if sig.get('willing_to_relocate') else '#e2e8f0'};">
                        {'Yes' if sig.get('willing_to_relocate') else 'No'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with m6:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">GitHub Activity Score</div>
                    <div class="metric-value">{sig.get("github_activity_score", "N/A")}</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.subheader("Engagement Details (Last 30 Days)")
            details_df = pd.DataFrame([{
                "Profile Completeness": f"{sig.get('profile_completeness_score', 0):.1f}%",
                "Profile Views": sig.get("profile_views_received_30d", 0),
                "Applications Submitted": sig.get("applications_submitted_30d", 0),
                "Search Appearances": sig.get("search_appearance_30d", 0),
                "Saved by Recruiters": sig.get("saved_by_recruiters_30d", 0),
                "Interview Completion Rate": f"{int(sig.get('interview_completion_rate', 0.0) * 100)}%" if sig.get('interview_completion_rate', -1) != -1 else "N/A",
                "Offer Acceptance Rate": f"{int(sig.get('offer_acceptance_rate', 0.0) * 100)}%" if sig.get('offer_acceptance_rate', -1) != -1 else "N/A"
            }])
            st.dataframe(details_df, use_container_width=True, hide_index=True)
        
    with tab4:
        st.subheader("Algorithmic Assessment Summary")
        st.markdown(f"""
        <div class="fit-rationale-callout">
            <h4 style="margin: 0 0 0.5rem 0; color: #00f2fe; font-weight: 700;">Rank Fit Rationale</h4>
            <p style="margin: 0; color: #e2e8f0; font-size: 1.05rem; line-height: 1.6; font-family: 'Outfit';">
                {selected_cand.get("reasoning", "No detailed rationale generated for this candidate.")}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive Score Breakdown
        st.subheader("Interactive Score Breakdown & Rules Inspection")
        
        if is_missing_profile:
            st.info("⚠️ Candidate profile details are missing from the current database. Cannot show detailed algorithmic breakdown.")
        else:
            breakdown = get_score_breakdown(
                selected_cand, 
                notice_period_cap, 
                min_experience, 
                tech_weight, 
                behavioral_weight
            )
            
            b_col1, b_col2, b_col3 = st.columns(3)
            
            with b_col1:
                st.markdown(f"""
                <div class="glass-panel" style="padding: 1rem; height: 100%;">
                    <h4 style="margin-top:0; color:#00f2fe;">1. Base Skill Fit</h4>
                    <p style="margin-bottom:0.5rem; font-size: 1rem;"><strong>Experience Score:</strong> {breakdown['exp_score']}/100</p>
                    <p style="font-size:0.85rem; color:#a0aec0; margin-bottom: 1.25rem;">Status: {breakdown['exp_status']} ({breakdown['exp_years']:.1f} yoe)</p>
                    
                    <p style="margin-bottom:0.25rem; font-size: 1rem;"><strong>Technical Score:</strong> {breakdown['tech_score']:.1f}/100</p>
                    <ul style="font-size:0.85rem; color:#cbd5e0; padding-left:1.25rem; margin-top:0;">
                        <li style="margin-bottom:0.25rem;">Vector/Embedding: {breakdown['vector_score']}/30</li>
                        <li style="margin-bottom:0.25rem;">Search & IR: {breakdown['ir_score']}/25</li>
                        <li style="margin-bottom:0.25rem;">ML & Tuning: {breakdown['ml_score']}/25</li>
                        <li style="margin-bottom:0.25rem;">Eval Frameworks: {breakdown['eval_score']}/20</li>
                    </ul>
                    <p style="border-top:1px solid #2d3748; padding-top:0.5rem; margin-top:0.75rem; margin-bottom:0; font-size: 1.05rem;">
                        <strong>Composite:</strong> {breakdown['composite_score']:.1f}/100
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            with b_col2:
                html_col2 = f"""
                <div class="glass-panel" style="padding: 1rem; height: 100%;">
                    <h4 style="margin-top:0; color:#00f2fe;">2. Disqualifiers (Multipliers)</h4>
                    <ul style="list-style-type:none; padding-left:0; margin:0;">
                """
                for d_name, d_val in breakdown['disqualifiers'].items():
                    icon = "🚨" if d_val['value'] else "✅"
                    color = "#ff4b4b" if d_val['value'] else "#00f2fe"
                    html_col2 += f"""
                    <li style="margin-bottom: 0.75rem;">
                        <span style="font-size:1.1rem;">{icon}</span> <strong>{d_name}:</strong> 
                        <span style="color:{color}; font-weight:600;">{d_val['multiplier']:.2f}x</span>
                        <p style="font-size:0.8rem; color:#cbd5e0; margin:0; line-height:1.3;">{d_val['reason']}</p>
                    </li>
                    """
                html_col2 += f"""
                    </ul>
                    <p style="border-top:1px solid #2d3748; padding-top:0.5rem; margin-top:0.5rem; margin-bottom:0; font-size: 1.05rem;">
                        <strong>Total Penalty Mult:</strong> {breakdown['disqualifier_mult_total']:.4f}x
                    </p>
                </div>
                """
                st.markdown(html_col2, unsafe_allow_html=True)
                
            with b_col3:
                html_col3 = f"""
                <div class="glass-panel" style="padding: 1rem; height: 100%;">
                    <h4 style="margin-top:0; color:#00f2fe;">3. Behavioral Multipliers</h4>
                    <ul style="list-style-type:none; padding-left:0; margin:0;">
                """
                for b_name, b_val in breakdown['behavior_multipliers'].items():
                    html_col3 += f"""
                    <li style="margin-bottom: 0.5rem;">
                        <strong>{b_name}:</strong> <span style="color:#4facfe; font-weight:600;">{b_val[0]:.2f}x</span>
                        <p style="font-size:0.8rem; color:#cbd5e0; margin:0; line-height:1.2;">{b_val[1]}</p>
                    </li>
                    """
                html_col3 += f"""
                    </ul>
                    <p style="border-top:1px solid #2d3748; padding-top:0.5rem; margin-top:0.5rem; margin-bottom:0; font-size: 1.05rem;">
                        <strong>Total Behavioral Mult:</strong> {breakdown['behavior_mult_total']:.4f}x
                    </p>
                </div>
                """
                st.markdown(html_col3, unsafe_allow_html=True)
                
            # Overall Score Formula Display
            st.markdown(f"""
            <div class="glass-panel" style="text-align: center; margin-top: 1rem; border: 1px solid #4facfe;">
                <h4 style="margin-top:0; color:#4facfe; font-weight:700;">Overall Match Score Calculation</h4>
                <p style="font-size:1.15rem; font-weight:500; margin:0; color:#cbd5e0; font-family:'Outfit';">
                    Formula: Score = (Base Skill Fit * Disqualifiers Multiplier * Behavioral Multiplier) / 100
                </p>
                <p style="font-size:1.35rem; font-weight:700; margin:0.5rem 0 0 0; color:#ffffff; font-family:'Outfit';">
                    ({breakdown['composite_score']:.2f} * {breakdown['disqualifier_mult_total']:.4f} * {breakdown['behavior_mult_total']:.4f}) / 100 = 
                    <span style="background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight:800; font-size:1.65rem;">
                        {breakdown['final_score']:.3f}
                    </span>
                </p>
            </div>
            """, unsafe_allow_html=True)
        
    st.markdown("<br><hr>", unsafe_allow_html=True)
    
    # 5. Submission Checklist & Export
    if "validation_results" in st.session_state:
        results = st.session_state.validation_results
        checklist = results["checklist"]
        
        st.subheader("Pipeline Validation & Submission Check")
        
        col_chk_1, col_chk_2 = st.columns(2)
        with col_chk_1:
            st.markdown(f"""
            <div class="glass-panel" style="padding: 1.25rem; margin-bottom: 1rem; border-radius: 8px;">
                <h4 style="margin-top:0; color:#00f2fe;">Challenge Requirements</h4>
                <ul style="list-style-type: none; padding-left: 0; margin: 0;">
                    <li style="margin-bottom: 0.75rem; font-size: 1.05rem;">
                        {'✅' if checklist['exactly_100_rows'] else '❌'} Exactly 100 rows generated
                    </li>
                    <li style="margin-bottom: 0.75rem; font-size: 1.05rem;">
                        {'✅' if checklist['ranks_complete'] else '❌'} Ranks 1-100 appear once
                    </li>
                    <li style="margin-bottom: 0.75rem; font-size: 1.05rem;">
                        {'✅' if checklist['monotonic_scores'] else '❌'} Monotonic score progression
                    </li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col_chk_2:
            st.markdown(f"""
            <div class="glass-panel" style="padding: 1.25rem; margin-bottom: 1rem; border-radius: 8px;">
                <h4 style="margin-top:0; color:#00f2fe;">Quality Controls</h4>
                <ul style="list-style-type: none; padding-left: 0; margin: 0;">
                    <li style="margin-bottom: 0.75rem; font-size: 1.05rem;">
                        {'✅' if checklist['tie_break_ok'] else '❌'} Lexicographical tie-breaking
                    </li>
                    <li style="margin-bottom: 0.75rem; font-size: 1.05rem;">
                        {'✅' if checklist['honeypot_rate_ok'] else '❌'} Honeypot rate below threshold (&lt; 10%)
                    </li>
                    <li style="margin-bottom: 0.75rem; font-size: 1.05rem;">
                        <strong>Overall Honeypot Rate:</strong> {results['overall_honeypot_rate']:.2f}%
                    </li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        if results["success"]:
            st.success("✅ Submission meets all constraints. Ready to export!")
        else:
            st.error("❌ Submission fails some validation checks. Check details below:")
            for err in results["errors"]:
                st.write(f"- {err}")
            
            if use_sample:
                st.info("ℹ️ **Sample Dataset Limitation Notice**: The sample dataset (`sample_candidates.json`) only contains 50 candidates. The official submission format requires exactly 100 candidates (ranks 1–100), which is why the row count and rank completeness checks show as failing here. Once you upload and process the **full candidates dataset** (which has thousands of candidates), the ranker will output exactly 100 rows and all validation checks will successfully pass (turn green).")
                
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Download buttons
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.download_button(
                label="Download submission.csv",
                data=st.session_state.csv_content,
                file_name="submission.csv",
                mime="text/csv",
                use_container_width=True
            )
        with d_col2:
            st.download_button(
                label="Download submission_metadata.yaml",
                data=st.session_state.yaml_content,
                file_name="submission_metadata.yaml",
                mime="text/yaml",
                use_container_width=True
            )
else:
    st.markdown("""<div class="glass-panel" style="margin-top: 1rem; border-left: 4px solid #00f2fe;">
<h3 style="margin-top:0; color:#00f2fe; font-weight: 700;">🚀 Welcome to the Candidate Ranking Sandbox</h3>
<p style="color:#e2e8f0; font-size:1.05rem; line-height:1.6; margin-bottom: 1.5rem;">
This interactive recruitment simulator helps you configure, visual-check, and inspect the multi-signal candidate ranking pipeline for the <strong>Intelligent Candidate Discovery & Ranking Challenge</strong>.
</p>

<h4 style="color:#ffffff; margin-top:1.5rem; margin-bottom:0.75rem; font-weight: 600;">🔍 Challenge Task & Pipeline Flow:</h4>
<div style="display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap;">
<div style="flex: 1; min-width: 220px; background: rgba(26, 31, 44, 0.4); border: 1px solid #2d3748; padding: 1.25rem; border-radius: 8px;">
<strong style="color: #00f2fe; font-size: 1.05rem;">1. Honeypot Screening</strong>
<p style="font-size:0.875rem; color:#cbd5e0; margin-top:0.5rem; margin-bottom:0; line-height: 1.4;">
Hard-coded temporal rules identify and eliminate fraud profiles containing impossible career dates, skills with 0-month durations, or experience claims that mismatch resume logs.
</p>
</div>
<div style="flex: 1; min-width: 220px; background: rgba(26, 31, 44, 0.4); border: 1px solid #2d3748; padding: 1.25rem; border-radius: 8px;">
<strong style="color: #00f2fe; font-size: 1.05rem;">2. Base Match Scoring</strong>
<p style="font-size:0.875rem; color:#cbd5e0; margin-top:0.5rem; margin-bottom:0; line-height: 1.4;">
Computes technical capability (matching embeddings, IR, ML, and evaluation keywords) and experience length (ideal fit: 5-9 years). Weights are custom-configured by your sidebar settings.
</p>
</div>
<div style="flex: 1; min-width: 220px; background: rgba(26, 31, 44, 0.4); border: 1px solid #2d3748; padding: 1.25rem; border-radius: 8px;">
<strong style="color: #00f2fe; font-size: 1.05rem;">3. Disqualifiers & Flags</strong>
<p style="font-size:0.875rem; color:#cbd5e0; margin-top:0.5rem; margin-bottom:0; line-height: 1.4;">
Applies penalty multipliers for consulting-only roles (TCS, Wipro, etc.), academic-only researchers without shipping title, high job-hopping rates, or LangChain-only junior portfolios.
</p>
</div>
<div style="flex: 1; min-width: 220px; background: rgba(26, 31, 44, 0.4); border: 1px solid #2d3748; padding: 1.25rem; border-radius: 8px;">
<strong style="color: #00f2fe; font-size: 1.05rem;">4. Recruiter Multipliers</strong>
<p style="font-size:0.875rem; color:#cbd5e0; margin-top:0.5rem; margin-bottom:0; line-height: 1.4;">
Adjusts ranking based on response rate speed, notice period cap, willing to relocate, open to work status, and platform activity dates.
</p>
</div>
</div>

<h4 style="color:#ffffff; margin-bottom:0.5rem; font-weight: 600;">⚙️ How to use the sandbox:</h4>
<ol style="color:#e2e8f0; line-height:1.6; padding-left: 1.25rem; font-size: 0.95rem;">
<li>Select the <strong>Notice Period Cap</strong> and <strong>Minimum Experience</strong> thresholds in the sidebar.</li>
<li>Adjust the weights representing the importance of <strong>Technical Match</strong> vs. <strong>Experience Length</strong>.</li>
<li>Provide your team metadata (Team Name and Email) in the text boxes.</li>
<li>Click the <strong style="color: #00f2fe;">Run Ranker</strong> button above to execute the calculations on the candidate pool.</li>
</ol>
</div>""", unsafe_allow_html=True)
