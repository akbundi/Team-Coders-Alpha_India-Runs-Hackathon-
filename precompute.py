#!/usr/bin/env python3
"""
precompute.py  –  Offline feature extraction (imports extraction logic from rank.py).
Produces precomputed_features.csv containing all the columns that rank.py expects.
"""

import json
import csv
from datetime import datetime
from pathlib import Path
import pandas as pd

from rank import (
    extract_online_features,
    build_tfidf_index,
)

def main():
    base_dir        = Path(r"d:\redrob\[PUB] India_runs_data_and_ai_challenge")
    candidates_file = base_dir / "candidates.jsonl"
    out_file        = base_dir / "precomputed_features.csv"

    current_date = datetime(2026, 6, 15)

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

    # Load candidates
    print("Loading candidates...")
    candidates = []
    with open(candidates_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip():
                candidates.append(json.loads(line))
    print(f"Loaded {len(candidates)} candidates.")

    # Build TF-IDF index for fallback similarities used inside extract_online_features
    print("Building TF-IDF indices...")
    build_tfidf_index(candidates)

    # Extract features for all candidates
    print("Extracting features offline...")
    rows = []
    for idx, cand in enumerate(candidates):
        row = extract_online_features(cand, current_date, jd_text, ideal_text)
        rows.append(row)
        if (idx + 1) % 10_000 == 0:
            print(f"  {idx + 1:,} processed...")

    # Write to CSV
    print(f"Saving to {out_file}...")
    df = pd.DataFrame(rows)
    df.to_csv(out_file, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
