import json
from datetime import datetime
from collections import Counter
from pathlib import Path

def analyze_candidates():
    candidates_file = Path(r"d:\redrob\[PUB] India_runs_data_and_ai_challenge\candidates.jsonl")
    
    count = 0
    date_anomalies = 0
    duration_anomalies = 0
    skill_anomalies = 0
    consulting_counts = Counter()
    honeypot_candidates = []
    
    current_date = datetime(2026, 6, 15) # From ADDITIONAL_METADATA
    
    with open(candidates_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            count += 1
            cid = cand["candidate_id"]
            profile = cand.get("profile", {})
            career_history = cand.get("career_history", [])
            skills = cand.get("skills", [])
            
            # Check 1: Date anomalies in career history
            has_date_anomaly = False
            for job in career_history:
                start_str = job.get("start_date")
                end_str = job.get("end_date")
                is_current = job.get("is_current", False)
                duration_m = job.get("duration_months", 0)
                
                try:
                    start_dt = datetime.strptime(start_str, "%Y-%m-%d")
                    end_dt = datetime.strptime(end_str, "%Y-%m-%d") if end_str else current_date
                    
                    # If start date is after end date
                    if start_dt > end_dt:
                        has_date_anomaly = True
                    
                    # If duration in months is wildly inconsistent with start and end dates
                    calc_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
                    if abs(duration_m - calc_months) > 3: # allow some leeway
                        has_date_anomaly = True
                except Exception:
                    has_date_anomaly = True
            
            if has_date_anomaly:
                date_anomalies += 1
                
            # Check 2: Skill duration anomalies
            # expert or advanced skills with 0 duration_months
            has_skill_anomaly = False
            zero_dur_expert_count = 0
            for s in skills:
                name = s.get("name", "")
                prof = s.get("proficiency", "")
                dur = s.get("duration_months", 0)
                if prof in ["expert", "advanced"] and dur == 0:
                    zero_dur_expert_count += 1
            
            if zero_dur_expert_count >= 3:
                has_skill_anomaly = True
                skill_anomalies += 1
                
            # Check 3: General profile anomaly
            # Years of experience mismatch
            total_history_duration = sum(job.get("duration_months", 0) for job in career_history)
            years_from_history = total_history_duration / 12.0
            years_claimed = profile.get("years_of_experience", 0)
            
            # Claimed experience vs history duration mismatch
            has_experience_mismatch = abs(years_claimed - years_from_history) > 3.0
            if has_experience_mismatch:
                duration_anomalies += 1
            
            is_honeypot = has_date_anomaly or has_skill_anomaly or has_experience_mismatch
            if is_honeypot:
                honeypot_candidates.append((cid, has_date_anomaly, has_skill_anomaly, has_experience_mismatch))
                
            # Count current/past companies for consulting
            for job in career_history:
                comp = job.get("company", "").lower()
                for c_firm in ["tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant", "capgemini"]:
                    if c_firm in comp:
                        consulting_counts[c_firm] += 1
            
            if count % 10000 == 0:
                print(f"Processed {count} candidates...")
                
    print(f"\nAnalysis Summary (Total Candidates: {count}):")
    print(f"Date anomalies: {date_anomalies}")
    print(f"Skill anomalies (expert/advanced with 0 duration): {skill_anomalies}")
    print(f"Experience duration mismatches: {duration_anomalies}")
    print(f"Total potential honeypots: {len(honeypot_candidates)}")
    print(f"Sample honeypots (first 10): {honeypot_candidates[:10]}")
    print(f"Consulting counts: {dict(consulting_counts)}")

if __name__ == "__main__":
    analyze_candidates()
