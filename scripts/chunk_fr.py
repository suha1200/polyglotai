import json
import argparse
from pathlib import Path

def chunk_text(text, chunk_size=350, overlap=60):
    words = text.split()
    chunks, start = [], 0
    step = max(1, chunk_size - overlap)
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step
    return chunks

def main():
    p = argparse.ArgumentParser(description="Chunk English sections from merged corpus.")
    p.add_argument("--infile", required=True)
    p.add_argument("--outfile", required=True)
    p.add_argument("--chunk_size", type=int, default=350)
    p.add_argument("--overlap", type=int, default=60)
    p.add_argument("--min_length", type=int, default=30, help="Minimum chars per chunk")
    args = p.parse_args()

    infile, outfile = Path(args.infile), Path(args.outfile)
    count_rows = count_chunks = 0

    with infile.open("r", encoding="utf-8") as fin, outfile.open("w", encoding="utf-8") as fout:
        for line in fin:
            count_rows += 1
            row = json.loads(line)
            if row.get("language") != "fr":
                continue

            text = (row.get("content") or "").strip()
            if not text:
                continue

            for i, chunk in enumerate(chunk_text(text, args.chunk_size, args.overlap)):
                if len(chunk) < args.min_length:
                    continue
                out_row = {
                    "pack_id": row.get("pack_id"),
                    "section_title": row.get("section_title"),
                    "page": row.get("page"),
                    "chunk_id": f"{row.get('pack_id')}_{row.get('page')}_{i}",
                    "content": chunk,
                }
                fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                count_chunks += 1

    print(f"Done. Read rows: {count_rows}, wrote chunks: {count_chunks}")

if __name__ == "__main__":
    main()
