# search_v5.py
import os, json
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

INDEX_NAME = "polyglotai-v1"
MODEL_NAME = "BAAI/bge-m3"
DOC_PREFIX = "Represent this passage for retrieval: "
QUERY_PREFIX = "Represent this question for retrieving relevant passages: "

def embed_texts(model: SentenceTransformer, texts: List[str], is_query: bool) -> List[List[float]]:
    def prep(t: str) -> str:
        t = " ".join((t or "").split())
        prefix = QUERY_PREFIX if is_query else DOC_PREFIX
        return prefix + t
    return model.encode([prep(t) for t in texts], normalize_embeddings=True).tolist()

def main():
    # --- user input (edit these for quick tests) ---
    namespace = "fr"   # "en" | "fr" | "ar" | "all"
    query = "Quelles sont les implications éthiques de l’IA dans la santé ?"
    top_k = 5
    # ----------------------------------------------

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing PINECONE_API_KEY env var.")

    print(f"Loading model on CPU: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    print(f"Index: {INDEX_NAME} | Namespace: {namespace} | top_k={top_k}")
    qvec = embed_texts(model, [query], is_query=True)[0]

    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)

    res = index.query(
        vector=qvec,
        top_k=top_k,
        namespace=namespace,
        include_values=False,
        include_metadata=True,
    )

    print("\nQuery:", query)
    print("\nTop results:")
    for i, match in enumerate(res["matches"], 1):
        mid = match.get("id")
        score = match.get("score")
        md: Dict = match.get("metadata") or {}
        pack = md.get("pack_id", "")
        page = md.get("page", "")
        sect = md.get("section_title", "")
        print(f"{i}. id={mid} | score={score:.4f} | pack={pack} | page={page}")
        if sect:
            print(f"   section: {sect}")
    print("\nTip: set namespace='fr' or 'ar' to test French/Arabic queries.")

if __name__ == "__main__":
    main()
