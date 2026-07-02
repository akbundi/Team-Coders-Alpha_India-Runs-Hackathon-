# Redrob Candidate Ranking System

[![Live Website](https://img.shields.io/badge/Streamlit-Live%20Website-FF4B4B?style=for-the-badge&logo=Streamlit)](https://o4xtetvbrmrlmqcpbhz9mv.streamlit.app/)

👉 **[Try it Live](https://o4xtetvbrmrlmqcpbhz9mv.streamlit.app/)**

A top-tier candidate ranking solution for the **Intelligent Candidate Discovery & Ranking Challenge**.

## Architecture

Multi-signal ranking pipeline combining:
- **Semantic similarity** (30%) — SentenceTransformer cosine similarity vs JD
- **Skill ontology match** (20%) — weighted against JD core/secondary skill sets
- **Title relevance** (15%) — tiered title categories: Core AI → Adjacent → Disqualified
- **AI/ML experience years** (12%) — derived from career history titles/descriptions
- **Evidence scoring** (10%) — measurable impact metrics extracted from descriptions
- **Behavioral signals** (8%) — activity, response rate, open-to-work, notice period
- **Experience fit** (5%) — optimal range 5–9 years per JD

### Honeypot Detection (Hard Rules)
1. Expert/advanced skill with `duration_months == 0`
2. Career start date after end date, or calculated duration mismatch > 3 months
3. Claimed years of experience vs career history mismatch > 2.0 years

### Additional Penalties
- **Consulting-only** (score × 0.55) — entire career at TCS/Wipro/Infosys/Accenture/Cognizant/Capgemini
- **Keyword stuffing** — many AI buzzwords, zero measurable achievements

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Pre-compute features (run once — already done, CSV committed)
```bash
python precompute.py
```

### Step 2: Rank candidates (< 5 min CPU, no network)
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

### Step 3: Validate
```bash
python validate_submission.py submission.csv
```

## Files

| File | Purpose |
|------|---------|
| `precompute.py` | Offline: encodes all candidates with SentenceTransformer, saves features |
| `rank.py` | Runtime: loads precomputed features, scores and ranks top-100 candidates |
| `precomputed_features.csv` | Pre-computed semantic similarity + candidate flags (committed to git) |
| `requirements.txt` | Python dependencies |
| `submission_metadata.yaml` | Submission metadata |

## Compute Constraints Compliance

| Constraint | Limit | Our System |
|------------|-------|-----------|
| Runtime | ≤ 5 min | < 5 seconds (CSV lookup + sklearn TF-IDF fallback) |
| Memory | ≤ 16 GB | < 500 MB |
| Compute | CPU only | ✅ No GPU during ranking |
| Network | Off | ✅ No external API calls |
