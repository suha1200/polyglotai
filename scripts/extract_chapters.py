# extract_chapters.py
import re, json, pathlib, argparse

# Arabic ordinals we commonly see
ORD_MAP = {
    "الأول": 1, "الثاني": 2, "الثالث": 3, "الرابع": 4, "الخامس": 5,
    "السادس": 6, "السابع": 7, "الثامن": 8, "التاسع": 9, "العاشر": 10,
    "الحادي عشر": 11, "الثاني عشر": 12,
}

# arabic-indic digits → western
ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# match a standalone chapter heading line like "الفصل الأول" or "الفصل 2"
CHAP_RE = re.compile(
    r"^\s*الفصل\s+(?P<ord>(?:الحادي\s+عشر|الثاني\s+عشر|الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر|[0-9\u0660-\u0669]+))\s*$"
)

def ord_to_num(s: str) -> int | None:
    s = re.sub(r"\s+", " ", s.strip())
    if s in ORD_MAP: return ORD_MAP[s]
    # numeric form
    t = s.translate(ARABIC_INDIC)
    return int(t) if t.isdigit() else None

def normalize_title_lines(lines):
    """Join wrapped title lines until a blank line or another chapter header."""
    # join with a space, fix spaces around parentheses
    title = " ".join(l.strip() for l in lines if l.strip())
    title = re.sub(r"\s+\)", ")", title)
    title = re.sub(r"\(\s+", "(", title)
    title = re.sub(r"\s{2,}", " ", title).strip()
    return title

def extract_chapters(text: str):
    lines = text.splitlines()
    chapters = []
    i = 0
    while i < len(lines):
        m = CHAP_RE.match(lines[i])
        if not m:
            i += 1
            continue

        ord_text = m.group("ord")
        chap_no = ord_to_num(ord_text)
        chap_heading = lines[i].strip()  # e.g., "الفصل الأول"
        i += 1

        # collect following non-empty lines as title block (1–3 lines typical)
        title_lines = []
        while i < len(lines):
            # stop if blank, a page-break gap, or another chapter header
            if not lines[i].strip():  # blank line
                # allow one blank line if the title wrapped with an empty line in between
                # break on the first blank if we already have some title lines
                if title_lines:
                    i += 1
                    break
                else:
                    i += 1
                    continue
            if CHAP_RE.match(lines[i]):  # next chapter
                break
            # titles are usually short; stop when we hit a clearly long paragraph
            if len(lines[i].strip()) > 120 and title_lines:
                break
            title_lines.append(lines[i])
            # often titles wrap for 1–2 lines only; stop after 3 lines
            if len(title_lines) >= 3:
                i += 1
                break
            i += 1

        title = normalize_title_lines(title_lines)

        chapters.append({
            "chapter_number": chap_no,
            "chapter_heading": chap_heading,
            "chapter_title": title or None,
        })
    return chapters

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="clean .txt")
    ap.add_argument("--out", dest="outp", required=True, help="chapters.jsonl")
    args = ap.parse_args()

    p_in  = pathlib.Path(args.inp)
    p_out = pathlib.Path(args.outp)
    txt = p_in.read_text(encoding="utf-8", errors="replace")
    chs = extract_chapters(txt)

    with p_out.open("w", encoding="utf-8") as f:
        for c in chs:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"✅ wrote {len(chs)} chapters → {p_out}")
