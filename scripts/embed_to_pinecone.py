# embed_v2.py
import json
from pathlib import Path
from typing import Iterable, Dict, List
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"
DOC_PREFIX = "Represent this passage for retrieval: "

def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
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
    if len(t) > max_char:
        t = t[:max_char]
    return DOC_PREFIX + t

def main():
    infile = Path("data/processed/chunks_all.jsonl")
    batch_size = 64
    sample_out = Path("data/processed/embeds_sample.jsonl")  # small preview only
    sample_limit = 10  # only write first 10 vectors so file stays tiny

    print(f"Loading model on CPU: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    total_rows = 0
    total_embedded = 0
    sample_written = 0

    # clean sample file if exists
    if sample_out.exists():
        sample_out.unlink()

    for batch in batched(read_jsonl(infile), batch_size):
        # pull content and ids
        texts: List[str] = []
        ids: List[str] = []
        langs: List[str] = []
        metas: List[Dict] = []

        for row in batch:
            cid = row.get("chunk_id")
            text = row.get("content") or ""
            if not cid or not text:
                continue
            ids.append(cid)
            langs.append((row.get("language") or "").lower())
            metas.append({
                "pack_id": row.get("pack_id"),
                "page": row.get("page"),
                "section_title": row.get("section_title"),
            })
            texts.append(prep_text(text))

        if not texts:
            continue

        # embed this batch (normalized for cosine)
        vecs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
        total_rows += len(batch)
        total_embedded += len(ids)

        # print lightweight progress
        print(f"Embedded so far: {total_embedded}", end="\r")

        # write a tiny sample so you can inspect structure (first 10 only)
        if sample_written < sample_limit:
            to_write = min(sample_limit - sample_written, len(ids))
            with sample_out.open("a", encoding="utf-8") as fout:
                for i in range(to_write):
                    item = {
                        "id": ids[i],
                        "language": langs[i],
                        "metadata": metas[i],
                        "vector_dim": len(vecs[i]),
                        # store a short preview of the vector (first 8 numbers) to keep file tiny
                        "vector_head": [float(x) for x in vecs[i][:8]],
                    }
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            sample_written += to_write

    print()  # newline after the \r progress
    print(f"Done. Batches processed. Read rows: {total_rows}, Embedded: {total_embedded}")
    print(f"Wrote a tiny preview to: {sample_out} (first {sample_limit} vectors only)")

if __name__ == "__main__":
    main()
