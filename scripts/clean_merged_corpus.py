import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

# Common boilerplate or generic section titles to filter
GENERIC_TITLES = {
    "cover", "table of contents", "contents", "toc", "index",
    "preface", "introduction", "acknowledgements", "about the author",
    "copyright", "license", "back cover"
}

# Keywords per language for boilerplate detection
BOILERPLATE_KEYWORDS = {
    "en": GENERIC_TITLES,
    "fr": {
        "table des matières", "contenu", "index", "introduction",
        "remerciements", "préface", "droits d’auteur", "à propos de l’auteur"
    },
    "ar": {
        "جدول المحتويات", "المحتويات", "مقدمة", "فهرس", "حقوق النشر", "عن المؤلف"
    }
}

# Regex patterns
PUNCTUATION_RE = re.compile(r"[^\w\s]", re.UNICODE)
DIGIT_RE = re.compile(r"\d", re.UNICODE)

def is_boilerplate(section_title: str, lang: str) -> bool:
    if not section_title:
        return True
    section = section_title.lower().strip()
    return any(kw in section for kw in BOILERPLATE_KEYWORDS.get(lang, []))

def is_content_low_quality(content: str) -> bool:
    if not content or len(content.strip()) < 100:
        return True
    punct_ratio = len(PUNCTUATION_RE.findall(content)) / max(len(content), 1)
    digit_ratio = len(DIGIT_RE.findall(content)) / max(len(content), 1)
    return punct_ratio > 0.4 or digit_ratio > 0.3

def normalize_title(title: str) -> str:
    return title.lower().strip()

def clean_corpus(infile: Path, outfile: Path, discarded_outfile: Path):
    seen_titles = defaultdict(set)  # key: language, value: set of titles
    kept = []
    discarded = []

    with infile.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Cleaning {infile.name}"):
            row = json.loads(line)
            lang = row.get("language")
            content = row.get("content", "")
            title = row.get("section_title", "")
            norm_title = normalize_title(title)

            # Rules
            if is_boilerplate(norm_title, lang):
                discarded.append(row)
                continue
            if is_content_low_quality(content):
                discarded.append(row)
                continue
            if norm_title in seen_titles[lang]:
                # Re-label repeated generic title
                page = row.get("page", "")
                row["section_title"] = f"{title} - page {page}"
            else:
                seen_titles[lang].add(norm_title)

            kept.append(row)

    print(f"\n✅ Kept: {len(kept):,} | 🗑️ Discarded: {len(discarded):,}")

    with outfile.open("w", encoding="utf-8") as f_out:
        for row in kept:
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

    with discarded_outfile.open("w", encoding="utf-8") as f_disc:
        for row in discarded:
            f_disc.write(json.dumps(row, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean merged_corpus.jsonl across all languages.")
    parser.add_argument("--infile", required=True, type=Path, help="Input JSONL file (e.g., merged_corpus.jsonl)")
    parser.add_argument("--outfile", required=True, type=Path, help="Cleaned output JSONL file")
    parser.add_argument("--discarded_outfile", required=True, type=Path, help="File to store discarded rows")

    args = parser.parse_args()
    clean_corpus(args.infile, args.outfile, args.discarded_outfile)
