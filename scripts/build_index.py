"""
build_index.py
---------------
This script builds a FAISS index from all documents in the data/ folder.
Steps:
1. Load and read text files.
2. Normalize the text (esp. Arabic/French).
3. Split text into chunks (~400 tokens).
4. Encode each chunk into embeddings using embed_utils.
5. Store embeddings + metadata in FAISS.
6. Save FAISS index to disk (index/ folder).
"""

import os, glob, json
import faiss
import numpy as np

from embed_utils import load_model, encode_passages
from text_norm import normalize_text

# ==============================
# CONFIG
# ==============================
DATA_DIR = "data"
INDEX_DIR = "index"
CHUNK_SIZE = 400       # number of characters per chunk (simple for now)
CHUNK_OVERLAP = 80     # overlap between chunks

# Make sure index folder exists
os.makedirs(INDEX_DIR, exist_ok=True)

# ==============================
# STEP 1 — Load the model
# ==============================
model = load_model()

# ==============================
# STEP 2 — Helper: split text into chunks
# ==============================
def chunk_text(text: str, lang: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Splits text into overlapping chunks so that long documents
    can be encoded and retrieved more effectively.
    """
    text = normalize_text(text, lang=lang)  # normalize before splitting
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():  # avoid empty chunks
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ==============================
# STEP 3 — Collect all docs
# ==============================
all_chunks = []
metadata = []  # parallel list to keep track of where each chunk came from

for filepath in glob.glob(os.path.join(DATA_DIR, "*.*")):
    filename = os.path.basename(filepath)
    
    # detect language from filename (en_, ar_, fr_)
    if filename.startswith("en_"):
        lang = "en"
    elif filename.startswith("ar_"):
        lang = "ar"
    elif filename.startswith("fr_"):
        lang = "fr"
    else:
        lang = None  # unknown, we’ll still process
    
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    
    # chunk the document
    chunks = chunk_text(text, lang)
    
    # store text + metadata
    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        metadata.append({
            "file": filename,
            "lang": lang,
            "chunk_id": i
        })

print(f"Collected {len(all_chunks)} chunks from {len(metadata)} entries")

# ==============================
# STEP 4 — Encode chunks
# ==============================
embeddings = encode_passages(model, all_chunks)
embeddings = np.array(embeddings, dtype=np.float32)

print("Embeddings shape:", embeddings.shape)

# ==============================
# STEP 5 — Build FAISS index
# ==============================
dim = embeddings.shape[1]  # embedding dimension
index = faiss.IndexFlatIP(dim)  # Inner Product (cosine if normalized)
index.add(embeddings)

print(f"FAISS index built with {index.ntotal} vectors.")

# ==============================
# STEP 6 — Save index + metadata
# ==============================
faiss.write_index(index, os.path.join(INDEX_DIR, "docs.index"))

with open(os.path.join(INDEX_DIR, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("Index and metadata saved to 'index/' folder.")
