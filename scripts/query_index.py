"""
query_index.py
---------------
Search the FAISS index with a user query and print top-k matches.

Features:
1. Loads FAISS index + metadata (with text snippets) from index/ folder.
2. Encodes the query with the same embedding model used during indexing.
3. Searches FAISS (cosine similarity).
4. Prints results to terminal (with filename, lang, score, and snippet).
5. Saves results to a UTF-8 file (search_results.txt) to avoid broken Arabic rendering.

Usage examples:
  python scripts/query_index.py --q "Phase 2 goals" --k 5
  python scripts/query_index.py --q "لخّص أهداف المرحلة الثانية" --lang ar --k 3
  python scripts/query_index.py --q "Résume les objectifs de la phase 2" --lang fr --k 5
"""

import os
import json
import argparse
from typing import List, Tuple

import faiss
import numpy as np

# Helpers from our utility modules
from embed_utils import load_model, encode_queries
from text_norm import normalize_text

# -------------------------
# Paths to index + metadata
# -------------------------
INDEX_DIR = "index"
INDEX_PATH = os.path.join(INDEX_DIR, "docs.index")
META_PATH = os.path.join(INDEX_DIR, "metadata.json")


# -------------------------
# Load FAISS + metadata
# -------------------------
def load_faiss_and_meta() -> Tuple[faiss.Index, List[dict]]:
    """
    Load the FAISS index and its associated metadata.
    The metadata list must be aligned with the FAISS vectors.
    """
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("FAISS index not found. Did you run build_index.py?")
    if not os.path.exists(META_PATH):
        raise FileNotFoundError("Metadata not found. Did you run build_index.py?")

    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if index.ntotal != len(metadata):
        raise ValueError(
            f"Index vectors ({index.ntotal}) != metadata rows ({len(metadata)}). "
            "Rebuild the index to fix mismatch."
        )
    return index, metadata


# -------------------------
# Search function
# -------------------------
def search(index: faiss.Index, query_vec: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search FAISS for the top_k most similar vectors.
    We used normalized embeddings with IndexFlatIP,
    so inner product = cosine similarity.
    """
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)  # ensure shape (1, dim)
    scores, indices = index.search(query_vec.astype(np.float32), top_k)
    return scores[0], indices[0]


# -------------------------
# Pretty-print results to terminal
# -------------------------
def pretty_print_results(scores: np.ndarray, indices: np.ndarray, metadata: List[dict], show: int = 5):
    """
    Print results as a simple table in the terminal, plus a snippet preview.
    """
    print("\nTop matches:")
    print("-" * 70)
    print(f"{'rank':<5} {'score':<8} {'lang':<4} {'file':<30} {'chunk_id':<8}")
    print("-" * 70)
    for rank, (score, idx) in enumerate(zip(scores[:show], indices[:show]), start=1):
        if idx == -1:
            continue  # FAISS uses -1 for "no result"
        meta = metadata[idx]
        print(f"{rank:<5} {score:<8.4f} {str(meta.get('lang')):<4} {meta.get('file', ''):<30} {str(meta.get('chunk_id')):<8}")
        # print a snippet (first 120 chars)
        snippet = meta.get("text", "")
        if snippet:
            print(f"    snippet: {snippet[:120]}...")
    print("-" * 70)


# -------------------------
# Save results to file (UTF-8 safe)
# -------------------------
def save_results_to_file(user_query: str, scores, indices, metadata, filepath="search_results.txt", lang_hint=None, show=5):
    """
    Save results to a file (UTF-8) so Arabic/French text renders properly.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Query: {user_query}\n")
        if lang_hint:
            f.write(f"(normalized with lang hint: {lang_hint})\n")
        f.write("\nTop matches:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'rank':<5} {'score':<8} {'lang':<4} {'file':<30} {'chunk_id':<8}\n")
        f.write("-" * 70 + "\n")
        for rank, (score, idx) in enumerate(zip(scores[:show], indices[:show]), start=1):
            if idx == -1:
                continue
            meta = metadata[idx]
            f.write(f"{rank:<5} {score:<8.4f} {str(meta.get('lang')):<4} {meta.get('file', ''):<30} {str(meta.get('chunk_id')):<8}\n")
            snippet = meta.get("text", "")
            if snippet:
                f.write(f"    snippet: {snippet[:200]}...\n")
        f.write("-" * 70 + "\n")

    print(f"\n✅ Results also saved to {filepath} (UTF-8)")


# -------------------------
# Main program
# -------------------------
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Search the FAISS index with a query.")
    parser.add_argument("--q", required=True, help="Your query (in EN, AR, or FR).")
    parser.add_argument("--k", type=int, default=5, help="Number of results to return.")
    parser.add_argument("--lang", default=None, help="Optional language hint (ar, fr, en). Helps with normalization.")
    args = parser.parse_args()

    user_query = args.q
    k = args.k
    lang_hint = args.lang

    # Load model + FAISS index
    model = load_model()
    index, metadata = load_faiss_and_meta()

    # Normalize + encode query
    norm_query = normalize_text(user_query, lang=lang_hint)
    query_vec = encode_queries(model, [norm_query])[0]

    # Search FAISS
    scores, indices = search(index, query_vec, top_k=k)

    # Print results in terminal
    print(f"\nQuery: {user_query}")
    if lang_hint:
        print(f"(normalized with lang hint: {lang_hint})")
    pretty_print_results(scores, indices, metadata, show=k)

    # Save results to UTF-8 file
    save_results_to_file(user_query, scores, indices, metadata, lang_hint=lang_hint, show=k)


if __name__ == "__main__":
    main()
