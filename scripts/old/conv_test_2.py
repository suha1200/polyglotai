import argparse, pathlib, subprocess, sys, re, unicodedata, shutil

BIDI_RE = re.compile(r"[\u061C\u200E\u200F\u202A-\u202E\u2066-\u2069]")
DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")

def normalize_arabic(s: str, remove_diacritics=False) -> str:
    s = unicodedata.normalize("NFKC", s)  # fold presentation forms/ligatures
    # keep newlines/tabs; drop only format controls (Cf)
    s = "".join(ch for ch in s if ch in "\n\r\t" or unicodedata.category(ch) != "Cf")
    s = BIDI_RE.sub("", s)                # extra safety
    s = s.replace("\u0640", "")           # remove tatweel
    if remove_diacritics:
        s = DIACRITICS_RE.sub("", s)
    s = re.sub(r"[ \t]+", " ", s)         # tidy spaces, keep line breaks
    return s.strip()

def extract_with_pdftotext(pdf_path: pathlib.Path, first=None, last=None) -> str:
    exe = shutil.which("pdftotext")
    if not exe:
        sys.exit("pdftotext (Poppler) not found. Install Poppler and add it to PATH.")
    cmd = [exe, "-enc", "UTF-8", "-layout", "-nopgbrk"]
    if first: cmd += ["-f", str(first)]
    if last:  cmd += ["-l", str(last)]
    cmd += [str(pdf_path), "-"]  # output to stdout
    out = subprocess.check_output(cmd)
    return out.decode("utf-8", errors="replace")

def main():
    root = pathlib.Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--infile", required=False,
                   default=str(root / "data/domain_packs/arabic_pack/raw/ar_ai_ethics_2023.pdf"),
                   help="Path to .pdf or .txt")
    p.add_argument("--outfile", required=False,
                   default=str(root / "data/domain_packs/arabic_pack/processed/ar_ai_ethics_2023_fixed.txt"))
    p.add_argument("-f", "--first", type=int, help="first page (optional)")
    p.add_argument("-l", "--last",  type=int, help="last page (optional)")
    p.add_argument("--no-diacritics", action="store_true")
    args = p.parse_args()

    in_path = pathlib.Path(args.infile)

    if not in_path.exists():
        sys.exit(f"Input not found: {in_path}")

    if in_path.suffix.lower() == ".pdf":
        text = extract_with_pdftotext(in_path, first=args.first, last=args.last)
    else:
        text = in_path.read_text(encoding="utf-8", errors="replace")

    fixed = normalize_arabic(text, remove_diacritics=args.no_diacritics)

    out_path = pathlib.Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(fixed, encoding="utf-8")
    print(f"✅ Cleaned text saved to {out_path}")

if __name__ == "__main__":
    main()
