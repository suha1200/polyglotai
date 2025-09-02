import unicodedata, re

# Regex to detect Arabic diacritical marks (tashkeel)
AR_DIACRITICS = re.compile(r'[\u0617-\u061A\u064B-\u0652]')

def normalize_text(text: str, lang: str | None = None) -> str:
    """
    Normalize text before creating embeddings.
    
    Why?
    - Unicode normalization (NFKC) makes sure that visually identical characters
      are represented the same way in memory.
      Example: Arabic "ﻻ" vs "لا", or French accents like "é".
    - For Arabic, we optionally strip diacritics (tashkeel) because they
      don’t usually change meaning in modern search, but can hurt retrieval.
    
    Args:
        text: the input string
        lang: optional language code ("ar" for Arabic, "fr" for French, etc.)
    
    Returns:
        A cleaned and normalized string.
    """
    # Normalize Unicode (handles ligatures, accent forms, etc.)
    text = unicodedata.normalize("NFKC", text)
    
    # Special handling for Arabic
    if lang == "ar":
        text = AR_DIACRITICS.sub("", text)
    
    return text
