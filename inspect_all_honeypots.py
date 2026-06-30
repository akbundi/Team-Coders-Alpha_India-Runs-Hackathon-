import json
from datetime import datetime
from pathlib import Path

def inspect_all():
    candidates_file = Path(r"d:\redrob\[PUB] India_runs_data_and_ai_challenge\candidates.jsonl")
    current_date = datetime(2026, 6, 15)
    
    flagged_skills = set()
    flagged_date_order = set()
    flagged_date_calc = set()
    flagged_exp_mismatch = set()
    
    with open(candidates_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            cid = cand["candidate_id"]
            profile = cand.get("profile", {})
            career_history = cand.get("career_history", [])
            skills = cand.get("skills", [])
            
            # 1. Skill duration anomaly
            zero_dur_expert_count = sum(1 for s in skills if s.get("proficiency") in ["expert", "advanced"] and s.get("duration_months") == 0)
            if zero_dur_expert_count >= 3:
                flagged_skills.add(cid)
                
            # 2. Date order
            has_date_order_anomaly = False
            has_date_calc_anomaly = False
            for job in career_history:
                start_str = job.get("start_date")
                end_str = job.get("end_date")
                duration_m = job.get("duration_months", 0)
                try:
                    start_dt = datetime.strptime(start_str, "%Y-%m-%d")
                    end_dt = datetime.strptime(end_str, "%Y-%m-%d") if end_str else current_date
                    if start_dt > end_dt:
                        has_date_order_anomaly = True
                    calc_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
                    if abs(duration_m - calc_months) > 3:
                        has_date_calc_anomaly = True
                except Exception:
                    has_date_order_anomaly = True
                    
            if has_date_order_anomaly:
                flagged_date_order.add(cid)
            if has_date_calc_anomaly:
                flagged_date_calc.add(cid)
                
            # 3. Exp mismatch
            total_history_duration = sum(job.get("duration_months", 0) for job in career_history)
            years_from_history = total_history_duration / 12.0
            years_claimed = profile.get("years_of_experience", 0)
            
            # Let's count mismatch
            # If history duration is e.g. 11 months (0.9 yrs) but claimed is 12.8 yrs, mismatch is 11.9
            if abs(years_claimed - years_from_history) > 3.0:
                flagged_exp_mismatch.add(cid)
                
    print(f"Total candidates: 100000")
    print(f"Flagged by skill duration: {len(flagged_skills)}")
    print(f"Flagged by date order: {len(flagged_date_order)}")
    print(f"Flagged by date calculation: {len(flagged_date_calc)}")
    print(f"Flagged by experience mismatch: {len(flagged_exp_mismatch)}")
    
    union_all = flagged_skills | flagged_date_order | flagged_date_calc | flagged_exp_mismatch
    print(f"Union of all flags: {len(union_all)}")
    
    # Wait, let's see how many have BOTH date_calc/date_order and exp_mismatch, or skills
    # Let's see: is it possible that experience mismatch > 3.0 has many false positives?
    # Let's count how many have exp_mismatch AND (skills or date_calc or date_order)
    intersection_exp_and_other = flagged_exp_mismatch & (flagged_skills | flagged_date_order | flagged_date_calc)
    print(f"Intersection of experience mismatch with other flags: {len(intersection_exp_and_other)}")
    
    # Let's check: what if we require exp_mismatch to be very large (e.g. > 5.0 years) or what if we combine them?
    # Let's see if we print some of the flagged_exp_mismatch candidates that are NOT flagged by others, to see if they are false positives.
    only_exp_mismatch = flagged_exp_mismatch - (flagged_skills | flagged_date_order | flagged_date_calc)
    print(f"Only experience mismatch: {len(only_exp_mismatch)}")
    
    # Let's print some IDs from only_exp_mismatch
    print("Sample only_exp_mismatch:", list(only_exp_mismatch)[:10])

if __name__ == "__main__":
    inspect_all()
