# Minimal Pinecone smoke test: embed first N chunks and upsert to a serverless index.
from dotenv import load_dotenv
load_dotenv()
# embed_v4_full.py
import os, json, time
from pathlib import Path
from typing import List, Dict
import torch
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import re

MODEL_NAME = "BAAI/bge-m3"
INDEX_NAME = "polyglotai-v1"
DIM = 1024
METRIC = "cosine"
CLOUD = os.getenv("PINECONE_CLOUD", "aws")
REGION = os.getenv("PINECONE_REGION", "us-east-1")
DOC_PREFIX = "Represent this passage for retrieval: "

INFILE = Path("data/processed/chunks_all.jsonl")

def infer_language(pack_id: str | None, chunk_id: str | None) -> str:
    """
    Infer language from pack_id/chunk_id naming patterns.
    Examples seen:
      - english_pack_6_0           -> en
      - french_pack_90_0           -> fr
      - AR-COMP-THINK-2022_84_0    -> ar
    """
    p = (pack_id or "")
    c = (chunk_id or "")
    s_lower = f"{p} {c}".lower()
    s_upper = f"{p} {c}".upper()

    # English
    if "english_pack" in s_lower or re.search(r"\ben[-_]", s_lower):
        return "en"

    # French
    if "french_pack" in s_lower or re.search(r"\bfr[-_]", s_lower):
        return "fr"

    # Arabic (AR-... prefix style)
    if s_upper.startswith("AR-") or " AR-" in s_upper or s_upper.startswith("AR_") or " AR_" in s_upper:
        return "ar"

    # Fallback: unknown
    return "unknown"

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def batched(iterable, n: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

def prep_text(t: str, max_char: int = 8000) -> str:
    t = " ".join((t or "").split())
    return DOC_PREFIX + (t[:max_char] if len(t) > max_char else t)

def ensure_index(pc: Pinecone, name: str, dim: int, metric: str, cloud: str, region: str):
    existing = {ix["name"] for ix in pc.list_indexes()}
    if name in existing:
        return
    print(f"Creating Pinecone index: {name}")
    pc.create_index(name=name, dimension=dim, metric=metric, spec=ServerlessSpec(cloud=cloud, region=region))
    while True:
        if pc.describe_index(name).status["ready"]:
            break
        time.sleep(1)

def main():
    batch_size = 64

    print(f"Loading model on CPU: {MODEL_NAME}")
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing PINECONE_API_KEY environment variable")
    pc = Pinecone(api_key=api_key)

    ensure_index(pc, INDEX_NAME, DIM, METRIC, CLOUD, REGION)
    index = pc.Index(INDEX_NAME)

    total_rows, total_upserted = 0, 0

    for batch in batched(read_jsonl(INFILE), batch_size):
        ids, texts, metas, nss = [], [], [], []
        for row in batch:
            cid = row.get("chunk_id")
            text = row.get("content") or ""
            if not cid or not text:
                continue
            ids.append(cid)
            texts.append(prep_text(text))
            lang = infer_language(row.get("pack_id"), row.get("chunk_id"))
            nss.append(lang if lang in {"en", "fr", "ar"} else "all")

            metas.append({
                "pack_id": str(row.get("pack_id") or ""),
                "language": lang,                              # <-- now a safe string
                "page": str(row.get("page") or ""),
                "section_title": str(row.get("section_title") or ""),
            })
            lang = (row.get("language") or "").lower()
            nss.append(lang if lang in {"en", "fr", "ar"} else "all")

        if not texts:
            continue

        vecs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True).tolist()

        # group by namespace
        by_ns: Dict[str, List[Dict]] = {}
        for cid, v, meta, ns in zip(ids, vecs, metas, nss):
            by_ns.setdefault(ns, []).append({"id": cid, "values": v, "metadata": meta})

        for ns, vectors in by_ns.items():
            index.upsert(vectors=vectors, namespace=ns)

        total_rows += len(batch)
        total_upserted += len(ids)
        print(f"Upserted: {total_upserted}", end="\r")

    print(f"\nDone! Read rows: {total_rows}, Upserted: {total_upserted}")

if __name__ == "__main__":
    main()

