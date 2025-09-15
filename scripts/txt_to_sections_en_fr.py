# scripts/txt_to_sections.py
# -*- coding: utf-8 -*-
"""
TXT -> sectioned JSONL (chapters) for Arabic, English, and French books.

Features
- Detects chapter headings:
  * AR: "الفصل الأول" / "الفصل ٢" (Arabic-Indic supported)
  * EN: "Chapter 1" / "CHAPTER II" / "Chapter Ten"
  * FR: "Chapitre 1" / "CHAPITRE II" / "Chapitre premier"
- Stitches multi-line titles (e.g., "...العالم" + "بالذكاء الاصطناعي" → one line)
- Keeps common preface/appendix blocks (language-aware)
- Outputs JSONL records: pack_id, language, book, section_title, section_path, page, content
- Strong normalization for Arabic content; gentle normalization for EN/FR

Usage (PowerShell examples):
  python scripts\txt_to_sections.py "data\...\raw\txt\ar_ai_ethics_2023_clean.txt" ar "AR-ETHICS-2023" "data\...\sections" --book-stem "ar_ai_ethics_2023" --no-preclean --title-mode light
  python scripts\txt_to_sections.py "data\...\raw\txt\my_english_book.txt"       en "EN-BOOK-001"     "data\...\sections" --book-stem "my_english_book"
  python scripts\txt_to_sections.py "data\...\raw\txt\mon_livre_fr.txt"          fr "FR-LIVRE-001"    "data\...\sections" --title-mode light
"""

import re
import json
import pathlib
import unicodedata
import argparse

# -----------------------------
# Generic helpers / unicode
# -----------------------------

ARABIC_DIACRITICS = r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]"
ARABIC_INDIC      = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
BIDI_RE           = re.compile(r"[\u061C\u200E\u200F\u202A-\u202E\u2066-\u2069]")

def preclean_text(s: str) -> str:
    """
    Minimal pre-clean for detection across languages:
    - NFKC for consistent forms (e.g., ﻻ -> لا)
    - Remove BiDi/format controls (keep newlines/tabs)
    - Remove tatweel (ـ)
    - Convert page breaks (\f) to blank lines
    - Keep line structure; do NOT collapse across newlines
    """
    if not s:
        return s
    s = unicodedata.normalize("NFKC", s)
    s = "".join(ch for ch in s if ch in "\n\r\t" or unicodedata.category(ch) != "Cf")
    s = BIDI_RE.sub("", s)
    s = s.replace("\u0640", "")  # tatweel
    s = s.replace("\f", "\n\n")
    return s

def normalize_arabic(s: str, fix_order: bool = False) -> str:
    """
    Strong normalization for Arabic content paragraphs (stable retrieval).
    """
    if not s:
        return s
    s = unicodedata.normalize("NFKC", s)
    s = (s.replace("أ","ا").replace("إ","ا").replace("آ","ا")
           .replace("ى","ي").replace("ئ","ي").replace("ؤ","و"))
    s = s.replace("ـ","")
    s = re.sub(ARABIC_DIACRITICS, "", s)
    s = s.translate(ARABIC_INDIC)
    s = re.sub(r"\s+", " ", s).strip()
    if fix_order:
        def mostly_arabic(line: str) -> bool:
            return len(re.findall(r"[\u0600-\u06FF]", line)) >= max(1, len(line)//3)
        toks = s.split()
        if toks and mostly_arabic(s):
            s = " ".join(toks[::-1])
    return s

def normalize_generic(s: str) -> str:
    """
    Gentle normalization for EN/FR: NFKC + collapse whitespace (preserve content).
    """
    if not s:
        return s
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip()

def light_normalize(s: str) -> str:
    """
    Pretty titles/paths (all languages): NFKC + collapse spaces.
    """
    if not s:
        return s
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip()

def normalize_title_lines(lines):
    """Join wrapped title lines until a blank line or another chapter header."""
    title = " ".join(l.strip() for l in lines if l.strip())
    title = re.sub(r"\s+\)", ")", title)
    title = re.sub(r"\(\s+", "(", title)
    title = re.sub(r"\s{2,}", " ", title).strip()
    return title

# -----------------------------
# Language-aware headings
# -----------------------------

# Preface/appendix blocks per language (single-line matches)
KNOWN_BLOCKS_LNG = {
    "ar": [
        "المحتويات", "تمهيد السلسلة", "شكر وتقدير", "مسرد المصطلحات",
        "ملاحظات", "قراءات إضافية", "المراجع", "إلى أرنو",
    ],
    "en": [
        "Contents", "Preface", "Acknowledgments", "Acknowledgements",
        "Glossary", "Notes", "Further Reading", "References", "Index",
    ],
    "fr": [
        "Table des matières", "Préface", "Remerciements",
        "Glossaire", "Notes", "Lectures complémentaires", "Références", "Index",
    ],
}

# "Part" headers per language
PART_RE_LNG = {
    "ar": re.compile(r"(?m)^\s*الجزء\s+.+$"),
    "en": re.compile(r"(?im)^\s*part\s+(?:[IVXLCDM]+|\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*$"),
    "fr": re.compile(r"(?im)^\s*partie\s+(?:[IVXLCDM]+|\d+|première|deuxième|troisième|quatrième|cinquième)\s*$"),
}

# --- Arabic chapter regex (matches your earlier working logic)
CHAP_RE_AR = re.compile(
    r"^\s*الفصل\s+(?P<ord>(?:الحادي\s+عشر|الثاني\s+عشر|الأول|الثاني|الثالث|الرابع|الخـامس|الخامس|السادس|السابع|الثامن|التاسع|العاشر|[0-9\u0660-\u0669]+))\s*$",
    re.M
)

# --- English: Chapter 1 / CHAPTER II / Chapter Ten
EN_ORD_WORDS = r"(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
ROMANS       = r"(?:[IVXLCDM]+)"
CHAP_RE_EN   = re.compile(
    rf"(?im)^\s*chapter\s+(?:{ROMANS}|{EN_ORD_WORDS}|\d+)\s*$"
)

# --- French: Chapitre 1 / CHAPITRE II / Chapitre premier
FR_ORD_WORDS = r"(premier|deuxième|troisième|quatrième|cinquième|sixième|septième|huitième|neuvième|dixième|onzième|douzième)"
CHAP_RE_FR   = re.compile(
    rf"(?im)^\s*chapitre\s+(?:{ROMANS}|{FR_ORD_WORDS}|\d+)\s*$"
)

def get_chapter_regex_for_lang(lang: str):
    if lang == "ar": return CHAP_RE_AR
    if lang == "en": return CHAP_RE_EN
    if lang == "fr": return CHAP_RE_FR
    return CHAP_RE_EN  # default

def get_known_blocks_for_lang(lang: str):
    return KNOWN_BLOCKS_LNG.get(lang, KNOWN_BLOCKS_LNG["en"])

def get_part_regex_for_lang(lang: str):
    return PART_RE_LNG.get(lang, PART_RE_LNG["en"])

# -----------------------------
# Detection
# -----------------------------

def find_section_starts(text: str, lang: str):
    """
    Return sorted markers:
      {pos, kind: 'chapter'|'block'|'part', heading, title, skip_lines}
    - heading: matched line (e.g., "الفصل الأول", "Chapter 1", "Chapitre II")
    - title  : stitched lines after chapter heading (may fallback to heading)
    - skip_lines: how many initial lines to skip from section body
    """
    markers = []

    lines = text.splitlines()
    # char index at the start of each line
    idxs, pos_accum = [], 0
    for ln in lines:
        idxs.append(pos_accum)
        pos_accum += len(ln) + 1  # include newline

    # known one-line blocks
    for kb in get_known_blocks_for_lang(lang):
        # case-insensitive for EN/FR; exact match for AR
        flags = re.M | (re.I if lang in ("en", "fr") else 0)
        for m in re.finditer(rf"(?m)^\s*{re.escape(kb)}\s*$", text, flags):
            markers.append({
                "pos": m.start(), "kind": "block",
                "heading": kb, "title": kb, "skip_lines": 1
            })

    # parts
    PART_RE = get_part_regex_for_lang(lang)
    for m in PART_RE.finditer(text):
        line = text[m.start(): text.find("\n", m.start())].strip()
        markers.append({
            "pos": m.start(), "kind": "part",
            "heading": line, "title": line, "skip_lines": 1
        })

    # chapters
    chap_re = get_chapter_regex_for_lang(lang)
    i = 0
    while i < len(lines):
        m = chap_re.match(lines[i])
        if not m:
            i += 1
            continue

        heading = lines[i].strip()
        heading_pos = idxs[i]
        i += 1

        # collect following non-empty lines as the (wrapped) title
        title_lines = []
        while i < len(lines):
            cur = lines[i].strip()
            if not cur:
                if title_lines:
                    i += 1
                    break
                i += 1
                continue
            if chap_re.match(lines[i]):  # next chapter
                break
            if len(cur) > 120 and title_lines:
                break
            title_lines.append(lines[i])
            if len(title_lines) >= 3:
                i += 1
                break
            i += 1

        stitched = normalize_title_lines(title_lines) or heading
        skip = 1 + len(title_lines)
        markers.append({
            "pos": heading_pos, "kind": "chapter",
            "heading": heading, "title": stitched, "skip_lines": skip
        })

    # sort & dedup by pos
    markers.sort(key=lambda d: d["pos"])
    dedup, seen = [], set()
    for m in markers:
        if m["pos"] in seen:
            continue
        seen.add(m["pos"])
        dedup.append(m)
    return dedup

def slice_sections(text: str, title_mode: str, lang: str):
    """
    Build section tuples: (section_title, section_path, body_text)
    title_mode: 'raw' | 'light' | 'full'  (how titles/path are normalized)
    """
    markers = find_section_starts(text, lang)
    if not markers:
        return [("FULL_DOCUMENT", ["FULL_DOCUMENT"], text)]

    def norm_title(s: str) -> str:
        if title_mode == "raw":  return s.strip()
        if title_mode == "full":
            return normalize_arabic(s, fix_order=False) if lang == "ar" else normalize_generic(s)
        return light_normalize(s)  # default 'light'

    sections = []
    for i, m in enumerate(markers):
        start = m["pos"]
        end = markers[i+1]["pos"] if i + 1 < len(markers) else len(text)
        chunk = text[start:end]

        chunk_lines = chunk.splitlines()
        skip = min(m.get("skip_lines", 1), len(chunk_lines))
        body_lines = chunk_lines[skip:]
        body = "\n".join(body_lines).strip()

        # Prefer stitched title; fallback to heading; final fallback: "FULL_DOCUMENT"
        base_title = m["title"] or m["heading"] or "FULL_DOCUMENT"
        heading = m["heading"] or base_title

        title_clean   = norm_title(base_title)
        heading_clean = norm_title(heading)

        if m["kind"] == "chapter" and title_clean and heading_clean:
            section_title = title_clean
            section_path  = [heading_clean, title_clean]
        else:
            section_title = title_clean
            section_path  = [title_clean]

        sections.append((section_title, section_path, body))
    return sections

# -----------------------------
# Paragraph splitting
# -----------------------------

def split_paragraphs(body: str):
    """
    Split by blank lines; join wrapped lines within a paragraph.
    """
    paras = re.split(r"\n\s*\n+", body)
    out = []
    for p in paras:
        p = p.strip()
        if not p:
            continue
        p = re.sub(r"\s*\n\s*", " ", p)  # join wrapped lines
        out.append(p)
    return out

# -----------------------------
# Main conversion
# -----------------------------

def txt_to_sections(txt_path: str, lang: str, pack_id: str, out_dir: str,
                    book_stem: str = None, preclean: bool = True, title_mode: str = "light"):
    txt_path = pathlib.Path(txt_path)
    out_dir  = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    book = book_stem or txt_path.stem
    out_path = out_dir / f"{book}.sections.jsonl"

    raw = txt_path.read_text(encoding="utf-8", errors="replace")
    text = preclean_text(raw) if preclean else raw

    sections = slice_sections(text, title_mode=title_mode, lang=lang)

    with open(out_path, "w", encoding="utf-8") as f:
        page_counter = 1  # synthetic page counter
        for title, path, body in sections:
            # Titles/paths: pretty (light) unless 'raw'/'full' requested
            if title_mode == "raw":
                title_n = title.strip()
                path_n  = [p.strip() for p in path]
            elif title_mode == "full":
                if lang == "ar":
                    title_n = normalize_arabic(title, fix_order=False)
                    path_n  = [normalize_arabic(p, fix_order=False) for p in path]
                else:
                    title_n = normalize_generic(title)
                    path_n  = [normalize_generic(p) for p in path]
            else:
                title_n = light_normalize(title)
                path_n  = [light_normalize(p) for p in path]

            paras = split_paragraphs(body)
            for para in paras:
                content_n = normalize_arabic(para) if lang == "ar" else normalize_generic(para)
                if not content_n or len(content_n) < 5:
                    continue
                rec = {
                    "pack_id": pack_id,
                    "language": lang,
                    "book": book,
                    "section_title": title_n,
                    "section_path": path_n,
                    "page": page_counter,
                    "content": content_n,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                page_counter += 1
    return str(out_path)

# -----------------------------
# CLI
# -----------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("txt_path", help="Input TXT (raw from pdftotext or already cleaned)")
    ap.add_argument("lang", choices=["ar", "en", "fr"], help="Language of the book")
    ap.add_argument("pack_id")
    ap.add_argument("out_dir")
    ap.add_argument("--book-stem", default=None, help="Override output stem")
    ap.add_argument("--no-preclean", action="store_true",
                    help="Disable integrated pre-clean (use if your TXT is already cleaned)")
    ap.add_argument("--title-mode", choices=["raw", "light", "full"], default="light",
                    help="Title normalization: raw (none), light (NFKC+spaces), full (strong)")
    args = ap.parse_args()

    out = txt_to_sections(
        args.txt_path, args.lang, args.pack_id, args.out_dir,
        book_stem=args.book_stem,
        preclean=not args.no_preclean,
        title_mode=args.title_mode,
    )
    print(out)
