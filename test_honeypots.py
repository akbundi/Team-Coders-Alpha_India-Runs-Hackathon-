import json
from datetime import datetime
from pathlib import Path

def test_honeypots():
    candidates_file = Path(r"d:\redrob\[PUB] India_runs_data_and_ai_challenge\candidates.jsonl")
    current_date = datetime(2026, 6, 15)
    
    anomalies = []
    
    with open(candidates_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            cid = cand["candidate_id"]
            profile = cand.get("profile", {})
            career_history = cand.get("career_history", [])
            skills = cand.get("skills", [])
            education = cand.get("education", [])
            
            reasons = []
            
            # Check 1: Job Date Anomalies
            for idx, job in enumerate(career_history):
                start_str = job.get("start_date")
                end_str = job.get("end_date")
                duration_m = job.get("duration_months", 0)
                
                try:
                    start_dt = datetime.strptime(start_str, "%Y-%m-%d")
                    end_dt = datetime.strptime(end_str, "%Y-%m-%d") if end_str else current_date
                    
                    if start_dt > end_dt:
                        reasons.append(f"Job {idx} start_date {start_str} > end_date {end_str or 'current'}")
                    
                    calc_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
                    if abs(duration_m - calc_months) > 3:
                        reasons.append(f"Job {idx} duration_months {duration_m} != calculated {calc_months}")
                except Exception as e:
                    reasons.append(f"Job {idx} date parsing error: {e}")
            
            # Check 2: Skill Duration Anomalies
            zero_dur_expert_count = 0
            for s in skills:
                prof = s.get("proficiency", "")
                dur = s.get("duration_months", 0)
                if prof in ["expert", "advanced"] and dur == 0:
                    zero_dur_expert_count += 1
            if zero_dur_expert_count >= 3:
                reasons.append(f"Skill anomaly: {zero_dur_expert_count} expert/advanced skills with 0 duration")
                
            # Check 3: Claimed vs actual experience mismatch
            total_history_duration = sum(job.get("duration_months", 0) for job in career_history)
            years_from_history = total_history_duration / 12.0
            years_claimed = profile.get("years_of_experience", 0)
            
            if abs(years_claimed - years_from_history) > 3.0:
                reasons.append(f"Exp mismatch: claimed {years_claimed} yrs, history has {years_from_history:.1f} yrs")
                
            # Check 4: Education vs Experience anomaly
            if education:
                max_edu_end = max(e.get("end_year") for e in education if e.get("end_year") is not None)
                if max_edu_end:
                    # Current year is 2026.
                    max_possible_exp = 2026 - max_edu_end + 1
                    if years_claimed > max_possible_exp + 3.0: # Allow 3 years overlap
                        reasons.append(f"Edu vs Exp anomaly: graduated {max_edu_end}, max possible exp {max_possible_exp}, claimed {years_claimed}")
            
            # Check 5: Impossible duration (e.g. 8 years at a company founded 3 years ago)
            # How to check if a company was founded recently?
            # Wait, in the candidate summary or company name, is there a hint? E.g., Series A startup founded 3 years ago?
            # Wait, let's look at the company name and history duration. 
            # If a company was founded recently, maybe we can search for descriptions that mention it?
            # E.g. "at a company founded 3 years ago" - does the description say "founded 3 years ago"?
            # Let's check if any company description in career history contains words like "founded" and "ago".
            for job in career_history:
                desc = job.get("description", "").lower()
                duration = job.get("duration_months", 0)
                if "founded" in desc:
                    # Let's extract foundation year or age from desc if possible
                    # E.g., "company founded 3 years ago" (or 2 years ago, etc.)
                    # Let's scan for descriptions containing "founded" and check if duration is larger than the age of the company!
                    import re
                    match = re.search(r'founded\s+(\d+)\s+years?\s+ago', desc)
                    if match:
                        years_ago = int(match.group(1))
                        if duration > (years_ago * 12 + 6): # Allow 6 months leeway
                            reasons.append(f"Company founded {years_ago} years ago but worked for {duration/12:.1f} years")
                    
                    match_yr = re.search(r'founded\s+in\s+(\d{4})', desc)
                    if match_yr:
                        found_year = int(match_yr.group(1))
                        start_str = job.get("start_date")
                        if start_str:
                            try:
                                start_yr = datetime.strptime(start_str, "%Y-%m-%d").year
                                if start_yr < found_year:
                                    reasons.append(f"Started working in {start_yr} at company founded in {found_year}")
                            except Exception:
                                pass
            
            if reasons:
                anomalies.append((cid, reasons))
                
    print(f"Total anomalies found: {len(anomalies)}")
    print("First 15 anomalies:")
    for cid, r in anomalies[:15]:
        print(f"  {cid}: {r}")

if __name__ == "__main__":
    test_honeypots()
