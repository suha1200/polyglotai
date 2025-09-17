# hygiene.py
import re, hashlib, unicodedata
from typing import Tuple

# Defaults; each chunk_*.py can override after argparse.
MIN_CHARS = {"en": 50, "fr": 50, "ar": 60}
MIN_WORDS = 6
DIGITS_PUNCT_RATIO_DROP = 0.60
AR_MOVIE_DIGITS_RATIO_DROP = 0.30

TITLE_PATTERNS = {
    "en": re.compile(r"(cover|about this text|acknowledg|table of contents|index|bibliograph|copyright|license)", re.I),
    "fr": re.compile(r"(couverture|à propos|remerciements|sommaire|table des matières|bibliograph|droits|licen[cs]e)", re.I),
    "ar": re.compile(r"(الفهرس|شكر(?:\s*وتقدير)?|المحتويات|الخاتمة|المراجع|حقوق|ترخيص)"),
}
AR_MOVIE_TERMS = re.compile(r"(ترميناتور|تي\s*٢|تي2|T2|ترميناتور\s*2)", re.I)

LETTER_RE = re.compile(r"\p{L}", re.UNICODE)
DIGIT_RE  = re.compile(r"\d", re.UNICODE)
PUNCT_RE  = re.compile(r"[^\w\s]", re.UNICODE)

def normalize_arabic_basic(s: str) -> str:
    if not s: return s
    s = unicodedata.normalize("NFKC", s)
    s = (s.replace("أ","ا").replace("إ","ا").replace("آ","ا")
           .replace("ى","ي").replace("ئ","ي").replace("ؤ","و")
           .replace("ة","ه").replace("ـ",""))
    s = re.sub(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]", "", s)
    return s

def char_ratios(txt: str) -> Tuple[float, float, float]:
    if not txt:
        return 0.0, 0.0, 1.0
    total = len(txt)
    letters = len(LETTER_RE.findall(txt))
    digits  = len(DIGIT_RE.findall(txt))
    punct   = len(PUNCT_RE.findall(txt))
    letters_ratio = letters / total if total else 0.0
    non_letters_ratio = 1 - letters_ratio
    digits_ratio = digits / total if total else 0.0
    return letters_ratio, digits_ratio, non_letters_ratio

def is_boilerplate_title(lang: str, section_title: str) -> bool:
    pat = TITLE_PATTERNS.get(lang)
    return bool(pat and section_title and pat.search(section_title))

def too_short(lang: str, content: str) -> bool:
    if not content: return True
    if len(content) < MIN_CHARS.get(lang, 50): return True
    if len(content.split()) < MIN_WORDS: return True
    return False

def looks_like_ar_movie_subs(content: str) -> bool:
    if not content: return False
    _, digits_ratio, _ = char_ratios(content)
    return bool(AR_MOVIE_TERMS.search(content)) and (digits_ratio > AR_MOVIE_DIGITS_RATIO_DROP)

def should_drop_chunk(lang: str, section_title: str, content: str):
    if too_short(lang, content):
        return True, "too_short"
    if is_boilerplate_title(lang, section_title):
        return True, "boilerplate_title"
    _, _, non_letters_ratio = char_ratios(content)
    if non_letters_ratio > DIGITS_PUNCT_RATIO_DROP:
        return True, "digits_punct_heavy"
    if lang == "ar" and looks_like_ar_movie_subs(content):
        return True, "ar_movie_subs"
    return False, ""

def content_fingerprint(content: str, lang: str) -> str:
    if lang == "ar":
        content = normalize_arabic_basic(content)
    content = unicodedata.normalize("NFKC", content).strip()
    return hashlib.sha1(content.encode("utf-8", errors="ignore")).hexdigest()
