import json
from pathlib import Path

def inspect_honeypots():
    candidates_file = Path(r"d:\redrob\[PUB] India_runs_data_and_ai_challenge\candidates.jsonl")
    target_ids = ["CAND_0003430", "CAND_0003582", "CAND_0005291", "CAND_0007353", "CAND_0007413", "CAND_0008960", "CAND_0010294", "CAND_0010770", "CAND_0013536", "CAND_0016000"]
    
    found = 0
    with open(candidates_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            if cand["candidate_id"] in target_ids:
                print(f"=== Candidate: {cand['candidate_id']} ===")
                print(json.dumps(cand, indent=2))
                print("\n" + "="*40 + "\n")
                found += 1
                if found == len(target_ids):
                    break

if __name__ == "__main__":
    inspect_honeypots()
