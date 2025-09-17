import json, argparse, re, unicodedata, hashlib
from pathlib import Path
from typing import Tuple

# --------------------
# Arabic normalization
# --------------------
DIACRITICS_PATTERN = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")

def short_sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:8]

def normalize_arabic(s: str) -> str:
    if not s:
        return s
    s = unicodedata.normalize("NFKC", s)
    s = (s.replace("أ","ا").replace("إ","ا").replace("آ","ا")
           .replace("ى","ي").replace("ئ","ي").replace("ؤ","و")
           .replace("ة","ه"))
    s = s.replace("ـ","")
    s = DIACRITICS_PATTERN.sub("", s)
    return s

# --------------------
# Hygiene rules (Phase 1)
# --------------------
# Boilerplate/Front-matter titles (Arabic)
AR_TITLE_BOILERPLATE = re.compile(r"(?:^|[\s:])(?:الفهرس|شكر(?:\s*وتقدير)?|المحتويات|الخاتمة|المراجع|حقوق|ترخيص)(?:$|[\s:])")

# Arabic movie/subtitle noise (Terminator/تي ٢ etc.)
AR_MOVIE_TERMS = re.compile(r"(ترميناتور|تي\s*٢|تي2|T2|ترميناتور\s*2)", re.I)

def char_ratios(txt: str) -> Tuple[float, float, float]:
    """
    Returns (letters_ratio, digits_ratio, non_letters_ratio).
    Uses Python's str methods to be robust with Arabic unicode.
    """
    if not txt:
        return 0.0, 0.0, 1.0
    total = len(txt)
    letters = sum(1 for ch in txt if ch.isalpha())
    digits  = sum(1 for ch in txt if ch.isdigit())
    # punctuation/others are implicitly counted in non_letters_ratio
    letters_ratio = letters / total if total else 0.0
    digits_ratio  = digits / total if total else 0.0
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
    return bool(section_title and AR_TITLE_BOILERPLATE.search(section_title))

def looks_like_ar_movie_subs(content: str, movie_digits_ratio_drop: float) -> bool:
    if not content:
        return False
    _, digits_ratio, _ = char_ratios(content)
    return bool(AR_MOVIE_TERMS.search(content)) and (digits_ratio > movie_digits_ratio_drop)

def content_fingerprint(content: str, do_normalize: bool) -> str:
    txt = normalize_arabic(content) if do_normalize else content
    txt = unicodedata.normalize("NFKC", txt).strip()
    return hashlib.sha1(txt.encode("utf-8", errors="ignore")).hexdigest()

def should_drop_chunk(section_title: str,
                      content: str,
                      min_chars: int,
                      min_words: int,
                      digits_punct_drop: float,
                      movie_digits_ratio_drop: float) -> tuple[bool, str]:
    if too_short(content, min_chars, min_words):
        return True, "too_short"
    if is_boilerplate_title(section_title):
        return True, "boilerplate_title"
    _, _, non_letters_ratio = char_ratios(content)
    if non_letters_ratio > digits_punct_drop:
        return True, "digits_punct_heavy"
    if looks_like_ar_movie_subs(content, movie_digits_ratio_drop):
        return True, "ar_movie_subs"
    return False, ""

# --------------------
# Chunking
# --------------------
def chunk_text(text, chunk_size=350, overlap=60):
    words = text.split()
    chunks, start = [], 0
    step = max(1, chunk_size - overlap)
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words): break
        start += step
    return chunks

# --------------------
# Main
# --------------------
def main():
    p = argparse.ArgumentParser(description="Chunk Arabic sections from merged corpus with hygiene filters.")
    p.add_argument("--infile", required=True)
    p.add_argument("--outfile", required=True)
    p.add_argument("--chunk_size", type=int, default=350)
    p.add_argument("--overlap", type=int, default=60)

    # Hygiene knobs (Phase 1)
    p.add_argument("--min_chars", type=int, default=60, help="Minimum chars per chunk (AR recommended ≥60)")
    p.add_argument("--min_words", type=int, default=6, help="Minimum words per chunk")
    p.add_argument("--digits_punct_drop", type=float, default=0.60, help="Drop if non-letters ratio > this (tables/headers/code)")
    p.add_argument("--ar_movie_digits_drop", type=float, default=0.30, help="If movie terms present AND digits ratio > this, drop")

    # Phase 2 nicety
    p.add_argument("--prepend_title", action="store_true", help="Prepend [section_title] to content")

    # Normalization & dedupe
    p.add_argument("--normalize", action="store_true", help="Normalize Arabic for output content")
    p.add_argument("--dedupe", action="store_true", help="Enable exact-content dedupe (post-normalization for hashing)")

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

            # Language gate
            if (row.get("language") or "").lower() != "ar":
                continue

            raw_text = (row.get("content") or "").strip()
            if not raw_text:
                continue

            # NOTE: normalize flag is applied to final output content (below),
            # but we chunk based on the raw_text to avoid altering token positions too much.
            text_for_chunking = normalize_arabic(raw_text) if args.normalize else raw_text

            # Generate chunks
            chunks = chunk_text(text_for_chunking, args.chunk_size, args.overlap)
            section_title = row.get("section_title") or ""
            pack_id = row.get("pack_id")
            page = row.get("page")

            for i, chunk in enumerate(chunks):
                # Hygiene decisions (on the chunk as-is)
                drop, reason = should_drop_chunk(
                    section_title=section_title,
                    content=chunk,
                    min_chars=args.min_chars,
                    min_words=args.min_words,
                    digits_punct_drop=args.digits_punct_drop,
                    movie_digits_ratio_drop=args.ar_movie_digits_drop,
                )
                if drop:
                    dropped += 1
                    stats_by_reason[reason] = stats_by_reason.get(reason, 0) + 1
                    continue

                # Deduping by normalized content fingerprint (stronger)
                h = content_fingerprint(chunk, do_normalize=True) if args.dedupe else None
                if args.dedupe and h in seen_hashes:
                    dropped += 1
                    stats_by_reason["dedupe_exact"] = stats_by_reason.get("dedupe_exact", 0) + 1
                    continue
                if args.dedupe and h is not None:
                    seen_hashes.add(h)

                # Output content (optionally normalized) + prepend title (Phase 2 nicety)
                out_content = normalize_arabic(chunk) if args.normalize else chunk
                if args.prepend_title and section_title and not out_content.strip().startswith("["):
                    out_content = f"[{section_title}] " + out_content

                suffix = short_sha(out_content)
                out_row = {
                    "language": "ar",
                    "pack_id": pack_id,
                    "section_title": section_title,
                    "page": page,
                    "chunk_id": f"{pack_id}_{page}_{i}_{suffix}",
                    "content": out_content,
                }
                fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                kept += 1

    print(f"[ar] Done. Read rows: {count_rows}, wrote chunks: {kept}, dropped: {dropped}")
    print(f"[ar] Drop breakdown: {stats_by_reason}")

if __name__ == "__main__":
    main()
