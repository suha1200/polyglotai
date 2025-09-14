# scripts/txt_to_sections.py
import re, json, pathlib, unicodedata, argparse

ARABIC_DIACRITICS = r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]"
ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# ---- Normalizer (refined; token flip optional) ----
def normalize_arabic(s: str, fix_order: bool = False) -> str:
    if not s:
        return s
    s = unicodedata.normalize("NFKC", s)
    s = (s.replace("أ","ا").replace("إ","ا").replace("آ","ا")
           .replace("ى","ي").replace("ئ","ي").replace("ؤ","و"))
    s = s.replace("ـ","")                                  # tatweel
    s = re.sub(ARABIC_DIACRITICS, "", s)                  # remove diacritics
    s = s.translate(ARABIC_INDIC)                         # convert digits
    s = re.sub(r"\s+", " ", s).strip()                    # collapse spaces
    if fix_order:
        # conservative: reverse tokens only if mostly Arabic
        def mostly_arabic(line: str) -> bool:
            return len(re.findall(r"[\u0600-\u06FF]", line)) >= max(1, len(line)//3)
        toks = s.split()
        if toks and mostly_arabic(s):
            s = " ".join(toks[::-1])
    return s

# ---- Section detection ----
# Common preface/appendix sections found in your TXT
KNOWN_BLOCKS = [
    "المحتويات","تمهيد السلسلة","شكر وتقدير","مسرد المصطلحات",
    "ملاحظات","قراءات إضافية","المراجع","إلى أرنو"
]

# Matches lines like: "الفصل الأول", "الفصل الثاني", ... (with/without extra title)
# also Arabic-Indic numerals: "الفصل ١", etc.
ORDINALS = "الأول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر|الحادي عشر|الثاني عشر"
CHAPTER_RES = [
    re.compile(rf"^\s*الفصل\s+(?:{ORDINALS})\b.*$", re.M),
    re.compile(r"^\s*الفصل\s+([0-9٠-٩]+)\b.*$", re.M)
]

# Some books have part headers; keep it generic
PART_RE = re.compile(r"^\s*الجزء\s+.+$", re.M)

def find_section_starts(text: str):
    # gather all candidate starts: known blocks, parts, chapters
    starts = []
    # known blocks
    for kb in KNOWN_BLOCKS:
        for m in re.finditer(rf"(?m)^\s*{re.escape(kb)}\s*$", text):
            starts.append((m.start(), kb))

    # parts
    for m in PART_RE.finditer(text):
        title = text[m.start():text.find("\n", m.start())].strip()
        starts.append((m.start(), title))

    # chapters
    for cre in CHAPTER_RES:
        for m in cre.finditer(text):
            line = text[m.start():text.find("\n", m.start())]
            starts.append((m.start(), line.strip()))

    # de-dup and sort
    uniq = {}
    for pos, title in starts:
        if pos not in uniq:
            uniq[pos] = title
    return sorted([(pos, uniq[pos]) for pos in uniq.keys()], key=lambda x: x[0])

def slice_sections(text: str):
    markers = find_section_starts(text)
    if not markers:
        # fallback: everything as one section
        return [("FULL_DOCUMENT", text)]
    sections = []
    for i, (pos, title) in enumerate(markers):
        end = markers[i+1][0] if i+1 < len(markers) else len(text)
        chunk = text[pos:end]
        # Use the first line as title (clean it)
        first_line = chunk.splitlines()[0].strip()
        title_clean = normalize_arabic(first_line)
        # body without first line
        body = chunk[len(first_line):].strip()
        sections.append((title_clean, body))
    return sections

def split_paragraphs(body: str):
    # split on blank lines or long line breaks
    paras = re.split(r"\n\s*\n+", body)
    out = []
    for p in paras:
        p = p.strip()
        if not p:
            continue
        # join wrapped lines
        p = re.sub(r"\s*\n\s*", " ", p)
        out.append(p)
    return out

def txt_to_sections(txt_path: str, lang: str, pack_id: str, out_dir: str, book_stem: str = None):
    txt_path = pathlib.Path(txt_path)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    book = book_stem or txt_path.stem
    out_path = out_dir / f"{book}.sections.jsonl"

    raw = txt_path.read_text(encoding="utf-8", errors="ignore")
    # preserve a normalized copy for detection, but we’ll write normalized content too
    sections = slice_sections(raw)

    with open(out_path, "w", encoding="utf-8") as f:
        page_counter = 1  # TXT has no pages; we’ll simulate monotonic page ids
        for title, body in sections:
            # normalize title & paragraphs
            title_n = normalize_arabic(title)
            paras = split_paragraphs(body)
            for para in paras:
                content_n = normalize_arabic(para)
                if not content_n or len(content_n) < 5:
                    continue
                rec = {
                    "pack_id": pack_id,
                    "language": lang,
                    "book": book,
                    "section_title": title_n,
                    "section_path": [title_n],
                    "page": page_counter,     # synthetic page counter
                    "content": content_n
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                page_counter += 1
    return str(out_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("txt_path")
    ap.add_argument("lang", choices=["ar","en","fr"])
    ap.add_argument("pack_id")
    ap.add_argument("out_dir")
    ap.add_argument("--book-stem", default=None)
    args = ap.parse_args()
    print(txt_to_sections(args.txt_path, args.lang, args.pack_id, args.out_dir, book_stem=args.book_stem))
