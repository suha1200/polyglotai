from dotenv import load_dotenv
load_dotenv()
import os, json, time
from pathlib import Path
from typing import Dict, List, Iterable, Optional

import torch
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# ==== Config (v4) ====
INFILE = Path("data/processed/chunks_all.v4.jsonl")
INDEX_NAME = "polyglotai-v4"
MODEL_NAME = "BAAI/bge-m3"
DIM = 1024
METRIC = "cosine"
BATCH_SIZE = 64
MAX_CHAR = 8000  # safety truncate (characters)
DOC_PREFIX = "Represent this passage for retrieval: "
CLOUD = os.getenv("PINECONE_CLOUD", "aws")
REGION = os.getenv("PINECONE_REGION", "us-east-1")
LANGS = {"en", "fr", "ar"}  # valid namespaces only
PROGRESS_EVERY = 10  # batches
# =====================

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
    if len(t) > MAX_CHAR:
        t = t[:MAX_CHAR]
    return DOC_PREFIX + t

# Fallback language inference if 'language' is missing
def infer_language_from_ids(pack_id: Optional[str], chunk_id: Optional[str]) -> str:
    p = (pack_id or "")
    c = (chunk_id or "")
    pl = p.lower(); cl = c.lower()
    if c.startswith("AR-") or c.startswith("AR_") or p.startswith("AR-") or p.startswith("AR_"):
        return "ar"
    if cl.startswith("french_pack_") or pl.startswith("french_pack_"):
        return "fr"
    if cl.startswith("english_pack_") or pl.startswith("english_pack_"):
        return "en"
    return "unknown"

def ensure_index(pc: Pinecone, name: str, dim: int, metric: str, cloud: str, region: str):
    # Works across pinecone client versions
    existing = {getattr(ix, "name", ix.get("name")) for ix in pc.list_indexes()}
    if name not in existing:
        print(f"Creating index '{name}' (dim={dim}, metric={metric}, {cloud}/{region}) ...")
        pc.create_index(
            name=name,
            dimension=dim,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        # Wait for readiness
        while True:
            desc = pc.describe_index(name)
            status = getattr(desc, "status", None) or desc.get("status")
            if status and status.get("ready"):
                break
            time.sleep(1)

def upsert_with_retry(index, vectors, namespace: str, retries: int = 3, backoff: float = 1.0):
    for attempt in range(retries):
        try:
            index.upsert(vectors=vectors, namespace=namespace)
            return
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(backoff * (2 ** attempt))

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

    total_rows = total_vecs = 0
    per_ns_counts = {k: 0 for k in LANGS}

    for bidx, batch in enumerate(batched(read_jsonl(INFILE), BATCH_SIZE), start=1):
        ids: List[str] = []
        texts: List[str] = []
        metas: List[Dict] = []
        nss:  List[str] = []

        for row in batch:
            cid = row.get("chunk_id")
            text = row.get("content") or ""
            if not cid or not text:
                continue

            # Prefer explicit language field (chunkers added it)
            lang = (row.get("language") or "").lower()
            if lang not in LANGS:
                # Fallback to ID-based inference; still skip if unknown
                lang = infer_language_from_ids(row.get("pack_id"), cid)
            if lang not in LANGS:
                # Skip unknowns entirely (no 'all' namespace in v4)
                continue

            ids.append(cid)
            texts.append(prep_doc(text))
            metas.append({
                "pack_id": str(row.get("pack_id") or ""),
                "language": lang,
                "page": str(row.get("page") or ""),
                "section_title": str(row.get("section_title") or ""),
            })
            nss.append(lang)

        if not ids:
            continue

        vecs = model.encode(texts, batch_size=BATCH_SIZE, normalize_embeddings=True).tolist()

        # group by namespace
        by_ns: Dict[str, List[Dict]] = {}
        for _id, v, md, ns in zip(ids, vecs, metas, nss):
            by_ns.setdefault(ns, []).append({"id": _id, "values": v, "metadata": md})

        # upsert per namespace
        for ns, vectors in by_ns.items():
            if not vectors:
                continue
            upsert_with_retry(index, vectors, namespace=ns)
            per_ns_counts[ns] += len(vectors)

        total_rows += len(batch)
        total_vecs += len(ids)

        if bidx % PROGRESS_EVERY == 0:
            print(f"Progress: batches={bidx}, rows_read~{total_rows}, upserted={total_vecs}, per_ns={per_ns_counts}")

    print(f"\nDone. Rows read (approx): {total_rows}, Upserted vectors: {total_vecs}")
    print("Per-namespace counts:", per_ns_counts)

if __name__ == "__main__":
    main()
