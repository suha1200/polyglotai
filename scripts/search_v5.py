'''

# Run English only
python scripts/search_v5.py --lang en

# Run Arabic only, show 5 after rerank
python scripts/search_v5.py --lang ar --show_n 5

# Run ALL languages, retrieve 20 then rerank to top 5
python scripts/search_v5.py --lang all --top_k 20 --show_n 5

# Disable reranking (just raw Pinecone order)
python scripts/search_v5.py --lang fr --no_rerank


'''

import os
import argparse
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone
from pathlib import Path
import json

CONTENT_MAP = None  # lazy-initialized
load_dotenv()

INDEX_NAME = "polyglotai-v6"
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-large"  # multilingual cross-encoder
DOC_PREFIX = "Represent this passage for retrieval: "
QUERY_PREFIX = "Represent this question for retrieving relevant passages: "

# --- Your fixed test queries ---
LANG_QUERIES = {
    "en": [
        "What is the purpose of computer ethics?",
        "Which challenges do software developers face in ensuring reliability?",
        "What is the role of privacy in digital communication?",
    ],
    "fr": [
        "Qu’est-ce que l’éthique informatique cherche à protéger ?",
        "Quels sont les risques liés à la vie privée en ligne ?",
        "Quelle est la définition de la pensée computationnelle ?",
    ],
    "ar": [
        "ما هو الهدف من أخلاقيات الحوسبة؟",
        "ما هي التحديات التي يواجهها مطورو البرمجيات لضمان الموثوقية؟",
        "ما هو دور الخصوصية في وسائل الاتصال الرقمية؟",
    ],
}

def load_content_map(jsonl_path: str) -> dict:
    global CONTENT_MAP
    if CONTENT_MAP is not None:
        return CONTENT_MAP
    CONTENT_MAP = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            cid = row.get("chunk_id")
            txt = row.get("content") or ""
            if cid and txt:
                CONTENT_MAP[cid] = txt
    return CONTENT_MAP

def get_text_for_id(cid: str) -> str:
    if CONTENT_MAP is None:
        return ""
    return CONTENT_MAP.get(cid, "")

def reformulate_query(q: str, lang: str) -> str:
    q2 = q.strip()
    lo = q2.lower()
    if lang == "en":
        if "reliab" in lo:
            q2 += " (software quality, dependability, verification, testing, fault tolerance, defects)"
        if "ethic" in lo:
            q2 += " (privacy, fairness, accountability, responsibility)"
    elif lang == "fr":
        if "éthique" in lo or "ethique" in lo:
            q2 += " (vie privée, équité, responsabilité)"
        if "fiabilité" in lo:
            q2 += " (qualité logicielle, dépendabilité, vérification, tests, tolérance aux pannes, défauts)"
    elif lang == "ar":
        if "أخلاق" in q2 or "الأخلاقي" in q2:
            q2 += " (الخصوصية، العدالة، المسؤولية، المساءلة)"
        if "موثوق" in q2 or "موثوقية" in q2:
            q2 += " (جودة البرمجيات، الاعتمادية، التحقق، الاختبار، تحمل الأعطال، العيوب)"
    return q2

def load_models(rerank: bool) -> Tuple[SentenceTransformer, CrossEncoder | None]:
    print(f"Loading embedder on CPU: {EMBED_MODEL}")
    emb = SentenceTransformer(EMBED_MODEL, device="cpu")
    rer = None
    if rerank:
        print(f"Loading reranker on CPU: {RERANK_MODEL}")
        rer = CrossEncoder(RERANK_MODEL, device="cpu")
    return emb, rer

def embed_query(model: SentenceTransformer, text: str) -> List[float]:
    t = " ".join((text or "").split())
    t = QUERY_PREFIX + t
    return model.encode([t], normalize_embeddings=True).tolist()[0]

def pinecone_query(pc: Pinecone, index_name: str, namespace: str, qvec: List[float], top_k: int):
    index = pc.Index(index_name)
    return index.query(
        vector=qvec,
        top_k=top_k,
        namespace=namespace,
        include_values=False,
        include_metadata=True,
        filter={"language": {"$eq": namespace}},  # belt & suspenders
    )

def rerank(rer, query: str, matches: List[Dict]) -> List[Tuple[float, Dict]]:
    pairs = []
    for m in matches:
        cid = m.get("id", "")
        passage = get_text_for_id(cid)
        if not passage:  # fallback if not found
            md = m.get("metadata") or {}
            passage = md.get("section_title", "")
        # (optional) trim very long passages for speed
        if len(passage) > 1200:
            passage = passage[:1200]
        pairs.append((query, passage))
    scores = rer.predict(pairs)
    scored = list(zip(scores, matches))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def run_one_query(pc: Pinecone, emb: SentenceTransformer, rer: CrossEncoder | None,
                  namespace: str, query: str, top_k: int, show_n: int):
    q_ref = reformulate_query(query, namespace)
    qvec = embed_query(emb, q_ref)
    res = pinecone_query(pc, INDEX_NAME, namespace, qvec, top_k)
    matches = res.get("matches") or []

    print(f"\n[{namespace.upper()}] Query: {query}")
    if q_ref != query:
        print(f"  Reformulated: {q_ref}")

    if not matches:
        print("  No results.")
        return

    if rer is not None:
        ranked = rerank(rer, q_ref, matches)
        print(f"  Top {min(show_n, len(ranked))} after rerank:")
        for i, (score, m) in enumerate(ranked[:show_n], 1):
            md = m.get("metadata") or {}
            print(f"   {i}. r={score:.4f} | id={m.get('id')} | pack={md.get('pack_id','')} | page={md.get('page','')}")
            if md.get("section_title"):
                print(f"       section: {md['section_title']}")
        # Also show original top-5 Pinecone scores for context
        print("  (Original Pinecone top-5):")
        for i, m in enumerate(matches[:min(5, len(matches))], 1):
            md = m.get("metadata") or {}
            print(f"   {i}. s={m.get('score'):.4f} | id={m.get('id')} | sect={md.get('section_title','')}")
    else:
        print(f"  Top {min(show_n, len(matches))} (raw Pinecone):")
        for i, m in enumerate(matches[:show_n], 1):
            md = m.get("metadata") or {}
            print(f"   {i}. s={m.get('score'):.4f} | id={m.get('id')} | pack={md.get('pack_id','')} | page={md.get('page','')}")
            if md.get("section_title"):
                print(f"       section: {md['section_title']}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "fr", "ar", "all"], default="en")
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--show_n", type=int, default=5)
    ap.add_argument("--no_rerank", action="store_true")
    # NEW: path to your chunks file with content
    ap.add_argument("--content_file", default="data/processed/chunks_all_clean.jsonl",
                    help="JSONL with chunk_id + content for reranker")
    args = ap.parse_args()

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing PINECONE_API_KEY")

    emb, rer = load_models(rerank=not args.no_rerank)

    # ✅ load the content map here (only if reranking enabled)
    if rer is not None:
        path = args.content_file
        if not Path(path).exists():
            raise FileNotFoundError(f"--content_file not found: {path}")
        print(f"Loading content map from: {path}")
        load_content_map(path)

    pc = Pinecone(api_key=api_key)

    langs = ["en", "fr", "ar"] if args.lang == "all" else [args.lang]
    for ns in langs:
        for q in LANG_QUERIES[ns]:
            run_one_query(pc, emb, rer, ns, q, args.top_k, args.show_n)


if __name__ == "__main__":
    main()
