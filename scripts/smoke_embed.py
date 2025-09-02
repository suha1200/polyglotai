# scripts/smoke_embed.py
import os
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")

def cos_sim(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))

if __name__ == "__main__":
    model = SentenceTransformer(MODEL_NAME)

    # NOTE: E5 expects prefixes:
    q_en = "query: Phase 2 goals and deliverables"
    d_en = "passage: The main goals of Phase 2 include building a multilingual retrieval baseline."
    d_ar = "passage: تشمل الأهداف الرئيسية للمرحلة الثانية بناء نظام استرجاع متعدد اللغات."
    d_fr = "passage: Les principaux objectifs de la phase 2 incluent un pipeline multilingue."

    q_vec = model.encode(q_en, normalize_embeddings=True)
    d_en_vec = model.encode(d_en, normalize_embeddings=True)
    d_ar_vec = model.encode(d_ar, normalize_embeddings=True)
    d_fr_vec = model.encode(d_fr, normalize_embeddings=True)

    print("Embedding dim:", len(q_vec))
    print("sim(q, EN):", round(cos_sim(q_vec, d_en_vec), 4))
    print("sim(q, AR):", round(cos_sim(q_vec, d_ar_vec), 4))
    print("sim(q, FR):", round(cos_sim(q_vec, d_fr_vec), 4))
