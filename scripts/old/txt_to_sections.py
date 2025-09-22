# scripts/txt_to_sections.py
# -*- coding: utf-8 -*-
"""
Turn an Arabic TXT into sectioned JSONL with clean chapter detection.

Features
- Detects chapter headings like "الفصل الأول" / "الفصل ٢" (Arabic-Indic digits supported)
- Stitches multi-line chapter titles (e.g., line wrap: "...العالم" + "بالذكاء الاصطناعي")
- Keeps common preface/appendix single-line blocks ("تمهيد السلسلة", "مسرد المصطلحات", ...)
- Outputs JSONL with: pack_id, language, book, section_title, section_path, page, content
- Titles/paths lightly normalized (pretty), content strongly normalized (stable retrieval)

Usage (PowerShell one-liners):
  python scripts\txt_to_sections.py "data\domain_packs\arabic_pack\raw\txt\ar_ai_ethics_2023_clean.txt" ar "AR-ETHICS-2023" "data\domain_packs\arabic_pack\sections" --book-stem "ar_ai_ethics_2023" --no-preclean --title-mode light

If your TXT is raw (direct from pdftotext), omit --no-preclean to enable integrated pre-clean.
"""

import re
import json
import pathlib
import unicodedata
import argparse

# -----------------------------
# Config / helpers
# -----------------------------

ARABIC_DIACRITICS = r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]"
ARABIC_INDIC      = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
BIDI_RE           = re.compile(r"[\u061C\u200E\u200F\u202A-\u202E\u2066-\u2069]")

# Known single-line blocks often present in Arabic books
KNOWN_BLOCKS = [
    "المحتويات", "تمهيد السلسلة", "شكر وتقدير", "مسرد المصطلحات",
    "ملاحظات", "قراءات إضافية", "المراجع", "إلى أرنو",
]

# Part headers
PART_RE = re.compile(r"(?m)^\s*الجزء\s+.+$")

def preclean_text(s: str) -> str:
    """
    Minimal pre-clean for detection:
    - NFKC to fold Arabic presentation forms (e.g., ﻻ -> لا)
    - Remove BiDi/format controls (keep newlines/tabs)
    - Remove tatweel
    - Convert page breaks (\f) to blank lines
    - Keep line structure intact (do NOT collapse across newlines)
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
    Strong normalization for content paragraphs (stable retrieval):
    - NFKC, hamza/ya/ta marbuta unification, remove tatweel/diacritics
    - Convert Arabic-Indic digits to Western
    - Collapse internal whitespace
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

def light_normalize(s: str) -> str:
    """Gentle normalization for titles: NFKC + collapse spaces only."""
    if not s:
        return s
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip()

# -----------------------------
# Chapter detection (matching your extract_chapters.py behavior)
# -----------------------------

ORD_MAP = {
    "الأول": 1, "الثاني": 2, "الثالث": 3, "الرابع": 4, "الخـامس": 5,
    "الخامس": 5, "السادس": 6, "السابع": 7, "الثامن": 8, "التاسع": 9, "العاشر": 10,
    "الحادي عشر": 11, "الثاني عشر": 12,
}

CHAP_RE = re.compile(
    r"^\s*الفصل\s+(?P<ord>(?:الحادي\s+عشر|الثاني\s+عشر|الأول|الثاني|الثالث|الرابع|الخـامس|الخامس|السادس|السابع|الثامن|التاسع|العاشر|[0-9\u0660-\u0669]+))\s*$",
    re.M
)

def ord_to_num(s: str) -> int | None:
    s = re.sub(r"\s+", " ", s.strip())
    if s in ORD_MAP: 
        return ORD_MAP[s]
    t = s.translate(ARABIC_INDIC)
    return int(t) if t.isdigit() else None

def normalize_title_lines(lines):
    """Join wrapped title lines until a blank line or another chapter header."""
    title = " ".join(l.strip() for l in lines if l.strip())
    title = re.sub(r"\s+\)", ")", title)
    title = re.sub(r"\(\s+", "(", title)
    title = re.sub(r"\s{2,}", " ", title).strip()
    return title

# -----------------------------
# Detection
# -----------------------------

def find_section_starts(text: str):
    """
    Return sorted markers:
      {pos, kind: 'chapter'|'block'|'part', heading, title, skip_lines}
    - heading: matched line (e.g., 'الفصل الأول')
    - title  : stitched lines after chapter heading (may be heading if empty)
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
    for kb in KNOWN_BLOCKS:
        for m in re.finditer(rf"(?m)^\s*{re.escape(kb)}\s*$", text):
            markers.append({
                "pos": m.start(), "kind": "block",
                "heading": kb, "title": kb, "skip_lines": 1
            })

    # parts
    for m in PART_RE.finditer(text):
        line = text[m.start(): text.find("\n", m.start())].strip()
        markers.append({
            "pos": m.start(), "kind": "part",
            "heading": line, "title": line, "skip_lines": 1
        })

    # chapters (mirror extract_chapters.py stitching)
    i = 0
    while i < len(lines):
        m = CHAP_RE.match(lines[i])
        if not m:
            i += 1
            continue

        heading = lines[i].strip()  # e.g., "الفصل الأول"
        heading_pos = idxs[i]       # precise start position of this line
        i += 1

        # collect following non-empty lines as title block (1–3 lines typical)
        title_lines = []
        while i < len(lines):
            cur = lines[i].strip()
            if not cur:  # blank line
                if title_lines:
                    i += 1
                    break
                i += 1
                continue
            if CHAP_RE.match(lines[i]):  # next chapter starts
                break
            # titles are usually short; stop if a long paragraph appears and we already have some
            if len(cur) > 120 and title_lines:
                break
            title_lines.append(lines[i])
            if len(title_lines) >= 3:
                i += 1
                break
            i += 1

        stitched = normalize_title_lines(title_lines) or heading
        skip = 1 + len(title_lines)  # heading + stitched title lines

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

def slice_sections(text: str, title_mode: str = "light"):
    """
    Build section tuples: (section_title, section_path, body_text)
    title_mode: 'raw' | 'light' | 'full'  (how titles/path are normalized)
    """
    markers = find_section_starts(text)
    if not markers:
        return [("FULL_DOCUMENT", ["FULL_DOCUMENT"], text)]

    def norm_title(s: str) -> str:
        if title_mode == "raw":  return s.strip()
        if title_mode == "full": return normalize_arabic(s, fix_order=False)
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
    Preserves paragraph boundaries for cleaner JSON chunks.
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

    sections = slice_sections(text, title_mode=title_mode)

    with open(out_path, "w", encoding="utf-8") as f:
        page_counter = 1  # synthetic page counter
        for title, path, body in sections:
            # Titles/paths: pretty (light) unless title_mode='raw'
            if title_mode == "raw":
                title_n = title.strip()
                path_n  = [p.strip() for p in path]
            elif title_mode == "full":
                title_n = normalize_arabic(title, fix_order=False)
                path_n  = [normalize_arabic(p, fix_order=False) for p in path]
            else:
                title_n = light_normalize(title)
                path_n  = [light_normalize(p) for p in path]

            paras = split_paragraphs(body)
            for para in paras:
                content_n = normalize_arabic(para)  # strong normalization for retrieval
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
    ap.add_argument("lang", choices=["ar", "en", "fr"])
    ap.add_argument("pack_id")
    ap.add_argument("out_dir")
    ap.add_argument("--book-stem", default=None, help="Override output stem")
    ap.add_argument("--no-preclean", action="store_true", help="Disable integrated pre-clean (use if your TXT is already cleaned)")
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
