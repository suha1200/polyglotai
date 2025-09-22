# clean_arabic_pdf_text.py
import re, unicodedata, pathlib, argparse

BIDI = re.compile(r"[\u061C\u200E\u200F\u202A-\u202E\u2066-\u2069]")  # BiDi marks

def clean_text(s: str) -> str:
    # 1) fold presentation forms (e.g. ﻻ → لا, ﲆ → على)
    s = unicodedata.normalize("NFKC", s)
    # 2) remove BiDi/format controls but KEEP newlines/tabs
    s = "".join(ch for ch in s if ch in "\n\r\t" or unicodedata.category(ch) != "Cf")
    s = BIDI.sub("", s)
    # 3) remove tatweel
    s = s.replace("\u0640", "")
    # 4) unify whitespace (preserve line breaks)
    s = re.sub(r"[ \t]+", " ", s)
    # 5) turn page breaks \f into blank lines
    s = s.replace("\f", "\n\n")
    return s.strip()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="outp", required=True)
    args = ap.parse_args()

    p_in  = pathlib.Path(args.inp)
    p_out = pathlib.Path(args.outp)
    txt = p_in.read_text(encoding="utf-8", errors="replace")
    p_out.parent.mkdir(parents=True, exist_ok=True)
    p_out.write_text(clean_text(txt), encoding="utf-8")
    print("✅ cleaned →", p_out)
