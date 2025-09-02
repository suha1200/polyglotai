import os
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

# A set of E5 models that require special prefixes ("query:" and "passage:")
E5_MODELS = {
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-small",
    "intfloat/multilingual-e5-large",
}

def _model_name(model: SentenceTransformer) -> str:
    """
    Get the name of the loaded SentenceTransformer model.
    We use this to check whether it's an E5 model.
    """
    try:
        return model._first_module().__dict__.get("pretrained_model_name_or_path", "")
    except Exception:
        return ""

def load_model() -> SentenceTransformer:
    """
    Load the embedding model defined in the .env file.
    Default: 'intfloat/multilingual-e5-base'.
    """
    name = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    return SentenceTransformer(name)

def _ensure_np(x) -> np.ndarray:
    """
    Ensure the embeddings are returned as a NumPy array (float32).
    This helps later when saving to FAISS or doing math operations.
    """
    return np.array(x, dtype=np.float32)

def _maybe_prefix(texts: List[str], is_query: bool, model_name: str) -> List[str]:
    """
    Add the required prefix if the model is from the E5 family.
    - Queries: 'query: ...'
    - Documents/Passages: 'passage: ...'
    Other models (like MiniLM or LaBSE) don’t need prefixes.
    """
    if model_name in E5_MODELS:
        prefix = "query: " if is_query else "passage: "
        return [prefix + t for t in texts]
    return texts

def encode_queries(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """
    Encode queries into embeddings.
    Adds the 'query:' prefix automatically if needed.
    """
    name = _model_name(model)
    texts = _maybe_prefix(texts, is_query=True, model_name=name)
    vecs = model.encode(texts, normalize_embeddings=True)
    return _ensure_np(vecs)

def encode_passages(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """
    Encode passages/documents into embeddings.
    Adds the 'passage:' prefix automatically if needed.
    """
    name = _model_name(model)
    texts = _maybe_prefix(texts, is_query=False, model_name=name)
    vecs = model.encode(texts, normalize_embeddings=True)
    return _ensure_np(vecs)
