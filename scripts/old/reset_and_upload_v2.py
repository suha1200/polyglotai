from dotenv import load_dotenv
load_dotenv()
import os, json, time, re
from pathlib import Path
from typing import Dict, List, Iterable

import torch
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# ==== Config (edit if needed) ====
INFILE = Path("data/processed/chunks_all.jsonl")
INDEX_NAME = "polyglotai-v3"
MODEL_NAME = "BAAI/bge-m3"
DIM = 1024
METRIC = "cosine"
BATCH_SIZE = 64
MAX_CHAR = 8000  # safety truncate
DOC_PREFIX = "Represent this passage for retrieval: "
CLOUD = os.getenv("PINECONE_CLOUD", "aws")
REGION = os.getenv("PINECONE_REGION", "us-east-1")
# =================================

def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)

def batched(it, n: int):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf; buf = []
    if buf: yield buf

def prep_doc(text: str) -> str:
    t = " ".join((text or "").split())
    if len(t) > MAX_CHAR: t = t[:MAX_CHAR]
    return DOC_PREFIX + t

# STRICT language inference from IDs
def infer_language(pack_id: str | None, chunk_id: str | None) -> str:
    p = (pack_id or "")
    c = (chunk_id or "")
    c_lower = c.lower(); p_lower = p.lower()
    # Arabic first: AR-... / AR_... prefix
    if c.startswith("AR-") or c.startswith("AR_") or p.startswith("AR-") or p.startswith("AR_"):
        return "ar"
    # French: strict prefix french_pack_
    if c_lower.startswith("french_pack_") or p_lower.startswith("french_pack_"):
        return "fr"
    # English: strict prefix english_pack_
    if c_lower.startswith("english_pack_") or p_lower.startswith("english_pack_"):
        return "en"
    return "unknown"

def ensure_index(pc: Pinecone, name: str, dim: int, metric: str, cloud: str, region: str):
    existing = {ix["name"] for ix in pc.list_indexes()}
    if name not in existing:
        print(f"Creating index '{name}' (dim={dim}, metric={metric}, {cloud}/{region}) ...")
        pc.create_index(name=name, dimension=dim, metric=metric, spec=ServerlessSpec(cloud=cloud, region=region))
        while True:
            if pc.describe_index(name).status["ready"]:
                break
            time.sleep(1)

def main():
    # --- model (CPU)
    print(f"Loading model on CPU: {MODEL_NAME}")
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    # --- pinecone
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing PINECONE_API_KEY env var.")
    pc = Pinecone(api_key=api_key)
    ensure_index(pc, INDEX_NAME, DIM, METRIC, CLOUD, REGION)
    index = pc.Index(INDEX_NAME)

    total_read = total_sent = 0
    per_ns_counts = {"en":0, "fr":0, "ar":0, "all":0}

    for batch in batched(read_jsonl(INFILE), BATCH_SIZE):
        ids: List[str] = []
        texts: List[str] = []
        metas: List[Dict] = []
        nss:  List[str] = []

        for row in batch:
            cid = row.get("chunk_id")
            text = row.get("content") or ""
            if not cid or not text:
                continue

            lang = infer_language(row.get("pack_id"), cid)
            ns = lang if lang in {"en","fr","ar"} else "all"

            ids.append(cid)
            texts.append(prep_doc(text))
            metas.append({
                "pack_id": str(row.get("pack_id") or ""),
                "language": lang,                         # safe string
                "page": str(row.get("page") or ""),
                "section_title": str(row.get("section_title") or ""),
            })
            nss.append(ns)

        if not ids:
            continue

        vecs = model.encode(texts, batch_size=BATCH_SIZE, normalize_embeddings=True).tolist()

        # group by namespace
        by_ns: Dict[str, List[Dict]] = {}
        for _id, v, md, ns in zip(ids, vecs, metas, nss):
            by_ns.setdefault(ns, []).append({"id": _id, "values": v, "metadata": md})

        # upsert per namespace
        for ns, vectors in by_ns.items():
            if not vectors: continue
            index.upsert(vectors=vectors, namespace=ns)
            per_ns_counts[ns] = per_ns_counts.get(ns, 0) + len(vectors)

        total_read += len(batch)
        total_sent += len(ids)
        print(f"Upserted: {total_sent}", end="\r")

    print(f"\nDone. Read rows: {total_read}, Upserted: {total_sent}")
    print("Per-namespace upserts:", per_ns_counts)

if __name__ == "__main__":
    main()
