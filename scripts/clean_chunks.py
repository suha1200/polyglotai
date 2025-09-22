# scripts/clean_chunks.py
import argparse, json, re, hashlib
from pathlib import Path
from typing import Dict, Iterable

# --- Config thresholds (tune if needed)
MIN_LEN = {"en": 50, "fr": 50, "ar": 60}  # characters
MAX_PUNCT_RATIO = 0.6
MAX_DIGIT_RATIO = 0.35

# Boilerplate section title keywords per language (lowercased containment)
BAD_TITLES = {
    "en": ["cover", "table of contents", "contents", "toc", "index", "about this text",
           "acknowledg", "preface", "bibliograph", "copyright", "license"],
    "fr": ["couverture", "table des matières", "sommaire", "contenu", "index", "à propos",
           "remerciements", "préface", "bibliograph", "droits", "licence"],
    "ar": ["الفهرس", "المحتويات", "فهرس", "شكر", "شكر وتقدير", "الخاتمة", "المراجع", "حقوق", "ترخيص"]
}

def infer_language(pack_id: str|None, chunk_id: str|None) -> str:
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

def ratios(txt: str):
    if not txt: return 1.0, 1.0
    total = max(len(txt), 1)
    punct = sum(1 for ch in txt if not ch.isalnum() and not ch.isspace())
    digits = sum(1 for ch in txt if ch.isdigit())
    return punct/total, digits/total

def looks_like_movie_subtitle_ar(text: str) -> bool:
    # Heuristic to drop obvious subtitle-like noise
    if not text: return False
    t = text.strip()
    if "تي ٢" in t or "ترميناتور" in t:  # extend if you see more patterns
        return True
    # drop if digits-heavy and very short
    pr, dr = ratios(t)
    return (dr > 0.3 and len(t) < 120)

def is_bad_title(title: str, lang: str) -> bool:
    t = (title or "").lower().strip()
    for kw in BAD_TITLES.get(lang, []):
        if kw in t:
            return True
    return False

def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            yield json.loads(s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, type=Path)
    ap.add_argument("--outfile", required=True, type=Path)
    ap.add_argument("--discarded_outfile", required=True, type=Path)
    args = ap.parse_args()

    kept, discarded = 0, 0
    seen_hashes = set()

    with args.outfile.open("w", encoding="utf-8") as fout, \
         args.discarded_outfile.open("w", encoding="utf-8") as fdisc:
        for row in read_jsonl(args.infile):
            cid = row.get("chunk_id")
            text = row.get("content") or ""
            pack_id = row.get("pack_id")
            lang = (row.get("language") or "").lower()
            if lang not in {"en","fr","ar"}:
                lang = infer_language(pack_id, cid)

            # skip unknown language
            if lang not in {"en","fr","ar"}:
                discarded += 1; fdisc.write(json.dumps(row, ensure_ascii=False)+"\n"); continue

            # min length
            if len(text.strip()) < MIN_LEN[lang]:
                discarded += 1; fdisc.write(json.dumps(row, ensure_ascii=False)+"\n"); continue

            # boilerplate title
            title = row.get("section_title") or ""
            if is_bad_title(title, lang):
                discarded += 1; fdisc.write(json.dumps(row, ensure_ascii=False)+"\n"); continue

            # noise ratios
            pr, dr = ratios(text)
            if pr > MAX_PUNCT_RATIO or dr > MAX_DIGIT_RATIO:
                discarded += 1; fdisc.write(json.dumps(row, ensure_ascii=False)+"\n"); continue

            # arabic subtitle-like
            if lang == "ar" and looks_like_movie_subtitle_ar(text):
                discarded += 1; fdisc.write(json.dumps(row, ensure_ascii=False)+"\n"); continue

            # content dedupe
            h = hashlib.sha1(text.strip().encode("utf-8")).hexdigest()
            if h in seen_hashes:
                discarded += 1; fdisc.write(json.dumps(row, ensure_ascii=False)+"\n"); continue
            seen_hashes.add(h)

            # ensure language in metadata
            row["language"] = lang
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Kept: {kept} | Discarded: {discarded}")

if __name__ == "__main__":
    main()
