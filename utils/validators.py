import csv
import sys
from pathlib import Path

# Add the parent directory of this file to the path
sys.path.append(str(Path(__file__).parent.parent))

from validate_submission import validate_submission
from ranker import is_honeypot

def run_submission_checks(csv_path, final_list, total_candidates, honeypots_filtered):
    """
    Runs official and additional dashboard validation checks on the generated submission.
    Returns a dictionary of check outcomes and details.
    """
    errors = validate_submission(csv_path)
    
    # 1. Exactly 100 data rows generated
    row_count = 0
    ranks = []
    scores = []
    cids_by_rank = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_count += 1
                if 'rank' in row:
                    ranks.append(int(row['rank']))
                if 'score' in row:
                    scores.append(float(row['score']))
                if 'candidate_id' in row:
                    cids_by_rank.append(row['candidate_id'])
    except Exception as e:
        return {
            "success": False,
            "checklist": {
                "exactly_100_rows": False,
                "ranks_complete": False,
                "monotonic_scores": False,
                "tie_break_ok": False,
                "honeypot_rate_ok": False,
            },
            "honeypot_rate": 100.0,
            "errors": [f"Failed to read written CSV: {e}"]
        }
        
    exactly_100_rows = (row_count == 100)
    
    # 2. Ranks 1-100 appear once
    ranks_complete = (len(set(ranks)) == 100 and min(ranks) == 1 and max(ranks) == 100)
    
    # 3. Scores decrease monotonically by rank
    monotonic_scores = True
    for i in range(len(scores) - 1):
        if scores[i] < scores[i+1]:
            monotonic_scores = False
            break
            
    # 4. Tie-break rule verified (candidate_id ascending when scores are equal)
    tie_break_ok = True
    for i in range(len(scores) - 1):
        if scores[i] == scores[i+1]:
            if cids_by_rank[i] > cids_by_rank[i+1]:
                tie_break_ok = False
                break
                
    # 5. Honeypot rate below threshold (< 10%) in the final list
    ranked_honeypots = 0
    for cand in final_list:
        cand_dict = {
            "skills": cand.get("skills", []),
            "career_history": cand.get("career_history", []),
            "profile": cand.get("profile", {}),
            "redrob_signals": cand.get("redrob_signals", {})
        }
        flagged, _ = is_honeypot(cand_dict)
        if flagged:
            ranked_honeypots += 1
            
    # Calculate honeypot rates
    honeypot_rate = (ranked_honeypots / 100.0) * 100.0
    overall_honeypot_rate = (honeypots_filtered / max(total_candidates, 1)) * 100.0
    
    # Under 10% is acceptable
    honeypot_rate_ok = (overall_honeypot_rate < 10.0)

    checklist = {
        "exactly_100_rows": exactly_100_rows,
        "ranks_complete": ranks_complete,
        "monotonic_scores": monotonic_scores,
        "tie_break_ok": tie_break_ok,
        "honeypot_rate_ok": honeypot_rate_ok,
    }
    
    success = all(checklist.values()) and len(errors) == 0
    
    return {
        "success": success,
        "checklist": checklist,
        "overall_honeypot_rate": overall_honeypot_rate,
        "honeypot_rate": honeypot_rate,
        "errors": errors
    }
