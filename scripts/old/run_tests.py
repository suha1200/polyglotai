"""
run_tests.py
-------------
Run all queries from tests/prompts_trilingual.jsonl against the FAISS index.

Outputs:
- test_results.csv : table with query, lang, top-k results (file, lang, score, snippet)

Usage:
    python scripts/run_tests.py --k 3
"""

import os, json, argparse, csv
import faiss
import numpy as np
from typing import List, Tuple

# Import helpers
from embed_utils import load_model, encode_queries
from text_norm import normalize_text

# -------------------
# Paths
# -------------------
INDEX_DIR = "index"
INDEX_PATH = os.path.join(INDEX_DIR, "docs.index")
META_PATH = os.path.join(INDEX_DIR, "metadata.json")
TEST_FILE = "tests/prompts_trilingual.jsonl"


# -------------------
# Load FAISS + metadata
# -------------------
def load_faiss_and_meta() -> Tuple[faiss.Index, List[dict]]:
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


# -------------------
# Load test queries
# -------------------
def load_queries() -> List[dict]:
    queries = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


# -------------------
# Search helper
# -------------------
def search(index: faiss.Index, query_vec: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)
    scores, indices = index.search(query_vec.astype(np.float32), top_k)
    return scores[0], indices[0]


# -------------------
# Run tests
# -------------------
def run_tests(k: int = 3, out_file="test_results.csv"):
    model = load_model()
    index, metadata = load_faiss_and_meta()
    queries = load_queries()

    rows = []
    for q in queries:
        lang = q["lang"]
        text = q["q"]

        # Normalize + embed query
        norm_text = normalize_text(text, lang=lang)
        q_vec = encode_queries(model, [norm_text])[0]

        # Search FAISS
        scores, indices = search(index, q_vec, top_k=k)

        # Record results
        for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
            if idx == -1:
                continue
            meta = metadata[idx]
            rows.append({
                "query_lang": lang,
                "query": text,
                "rank": rank,
                "score": round(float(score), 4),
                "result_lang": meta.get("lang"),
                "result_file": meta.get("file"),
                "chunk_id": meta.get("chunk_id"),
                "snippet": meta.get("text", "")[:200]
            })

    # Save to CSV
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Test results saved to {out_file} (UTF-8)")

    # 👉 Call evaluation here!
    evaluate_results(rows, k=k)


def evaluate_results(rows: List[dict], k: int = 3):
    """
    Compute simple recall@1 and recall@k based on query_lang vs result_lang.
    Assumption: A correct retrieval means the retrieved doc has the same language as the query.
    """
    from collections import defaultdict

    total = len({r["query"] for r in rows})  # number of unique queries
    correct_at_1 = 0
    correct_at_k = 0

    # Group rows by query
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["query"]].append(r)

    for query, results in grouped.items():
        # sort results by rank
        results = sorted(results, key=lambda x: x["rank"])

        # Check top-1
        if results and results[0]["result_lang"] == results[0]["query_lang"]:
            correct_at_1 += 1

        # Check top-k
        if any(r["result_lang"] == r["query_lang"] for r in results[:k]):
            correct_at_k += 1

    recall1 = correct_at_1 / total if total else 0
    recallk = correct_at_k / total if total else 0

    print(f"\n📊 Evaluation:")
    print(f"Queries evaluated: {total}")
    print(f"Recall@1: {recall1:.2f}")
    print(f"Recall@{k}: {recallk:.2f}")

# -------------------
# CLI
# -------------------
def main():
    """
    CLI entry point.
    Example:
        python scripts/run_tests.py --k 3
    """
    parser = argparse.ArgumentParser(description="Run trilingual test queries and evaluate retrieval.")
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of top results per query (used for recall@k)."
    )
    args = parser.parse_args()

    # Run tests + evaluation
    run_tests(k=args.k)


if __name__ == "__main__":
    main()
