import json
import argparse
from pathlib import Path
import hashlib
import unicodedata
import re
from typing import Tuple

# --------------------
# Utilities
# --------------------
def short_sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:8]

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

# --------------------
# Hygiene filters (French)
# --------------------
FR_TITLE_BOILERPLATE = re.compile(
    r"(couverture|à propos|a propos|remerciements|sommaire|table des matières|bibliograph|droits|licen[cs]e)",
    re.I
)

def char_ratios(txt: str) -> Tuple[float, float, float]:
    if not txt:
        return 0.0, 0.0, 1.0
    total = len(txt)
    letters = sum(1 for ch in txt if ch.isalpha())
    digits = sum(1 for ch in txt if ch.isdigit())
    letters_ratio = letters / total if total else 0.0
    digits_ratio = digits / total if total else 0.0
    non_letters_ratio = 1.0 - letters_ratio
    return letters_ratio, digits_ratio, non_letters_ratio

def too_short(content: str, min_chars: int, min_words: int) -> bool:
    if not content:
        return True
    if len(content) < min_chars:
        return True
    if len(content.split()) < min_words:
        return True
    return False

def is_boilerplate_title(section_title: str) -> bool:
    return bool(section_title and FR_TITLE_BOILERPLATE.search(section_title))

def should_drop_chunk(section_title: str,
                      content: str,
                      min_chars: int,
                      min_words: int,
                      digits_punct_drop: float) -> tuple[bool, str]:
    if too_short(content, min_chars, min_words):
        return True, "too_short"
    if is_boilerplate_title(section_title):
        return True, "boilerplate_title"
    _, _, non_letters_ratio = char_ratios(content)
    if non_letters_ratio > digits_punct_drop:
        return True, "digits_punct_heavy"
    return False, ""

def content_fingerprint(content: str) -> str:
    txt = unicodedata.normalize("NFKC", content).strip()
    return hashlib.sha1(txt.encode("utf-8", errors="ignore")).hexdigest()

# --------------------
# Main
# --------------------
def main():
    p = argparse.ArgumentParser(description="Chunk French sections from merged corpus with hygiene filters.")
    p.add_argument("--infile", required=True)
    p.add_argument("--outfile", required=True)
    p.add_argument("--chunk_size", type=int, default=350)
    p.add_argument("--overlap", type=int, default=60)

    # Hygiene knobs
    p.add_argument("--min_chars", type=int, default=50, help="Minimum chars per chunk (FR ≥50 recommended)")
    p.add_argument("--min_words", type=int, default=6, help="Minimum words per chunk")
    p.add_argument("--digits_punct_drop", type=float, default=0.60, help="Drop if non-letters ratio > this (tables/headers/code)")
    p.add_argument("--prepend_title", action="store_true", help="Prepend [section_title] to content")
    p.add_argument("--dedupe", action="store_true", help="Enable exact-content dedupe")
    args = p.parse_args()

    infile, outfile = Path(args.infile), Path(args.outfile)
    count_rows = 0
    kept, dropped = 0, 0
    stats_by_reason: dict[str, int] = {}
    seen_hashes = set()

    with infile.open("r", encoding="utf-8") as fin, outfile.open("w", encoding="utf-8") as fout:
        for line in fin:
            count_rows += 1
            row = json.loads(line)

            if (row.get("language") or "").lower() != "fr":
                continue

            text = (row.get("content") or "").strip()
            if not text:
                continue

            section_title = row.get("section_title") or ""
            pack_id = row.get("pack_id")
            page = row.get("page")

            for i, chunk in enumerate(chunk_text(text, args.chunk_size, args.overlap)):
                drop, reason = should_drop_chunk(
                    section_title=section_title,
                    content=chunk,
                    min_chars=args.min_chars,
                    min_words=args.min_words,
                    digits_punct_drop=args.digits_punct_drop,
                )
                if drop:
                    dropped += 1
                    stats_by_reason[reason] = stats_by_reason.get(reason, 0) + 1
                    continue

                # Dedup
                if args.dedupe:
                    h = content_fingerprint(chunk)
                    if h in seen_hashes:
                        dropped += 1
                        stats_by_reason["dedupe_exact"] = stats_by_reason.get("dedupe_exact", 0) + 1
                        continue
                    seen_hashes.add(h)

                out_content = chunk
                if args.prepend_title and section_title and not out_content.strip().startswith("["):
                    out_content = f"[{section_title}] " + out_content

                suffix = short_sha(out_content)
                out_row = {
                    "language": "fr",
                    "pack_id": pack_id,
                    "section_title": section_title,
                    "page": page,
                    "chunk_id": f"{pack_id}_{page}_{i}_{suffix}",
                    "content": out_content,
                }
                fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                kept += 1

    print(f"[fr] Done. Read rows: {count_rows}, wrote chunks: {kept}, dropped: {dropped}")
    print(f"[fr] Drop breakdown: {stats_by_reason}")

if __name__ == "__main__":
    main()
