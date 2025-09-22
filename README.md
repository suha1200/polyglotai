# PolyglotAI

## 📌 Project Overview

PolyglotAI is an AI-powered assistant designed to work with **English, Arabic, and French documents**.
The goal is to build a multilingual retrieval-augmented generation (RAG) system that can index, search, and answer questions across these languages.

## 📂 Repository Structure (initial)

```
PolyglotAI/
│
├── docs/              # Documentation (EN/AR/FR sample files)
├── src/               # Source code
├── data/              # Datasets (raw/processed)
├── tests/             # Test scripts
├── .gitignore
├── README.md
```

## 🚀 Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/PolyglotAI.git
   cd PolyglotAI
   ```
2. Create a virtual environment (Python 3.10–3.11 recommended).

   ```bash
   python -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windows
   ```
3. Install dependencies (when `requirements.txt` is ready):

   ```bash
   pip install -r requirements.txt
   ```

## 🌍 Languages Supported

* English (en)
* Arabic (ar)
* French (fr)

## 📖 Documentation

* `docs/en_sample.md` – English doc example
* `docs/ar_sample.md` – Arabic doc example
* `docs/fr_sample.md` – French doc example

## ✅ Next Steps

* Set up initial data indexing
* Define evaluation metrics
* Build minimal working RAG prototype


1) Clean (post-chunk):
   python scripts/clean_chunks.py --infile data/processed/chunks_all.v4.jsonl \
     --outfile data/processed/chunks_all_clean.jsonl \
     --discarded_outfile data/processed/chunks_all_discarded.jsonl

2) (If cleaning pre-chunk instead)
   python scripts/clean_merged_corpus.py --infile data/unified/merged_corpus.jsonl \
     --outfile data/unified/merged_corpus_clean.jsonl \
     --discarded_outfile data/unified/merged_corpus_discarded.jsonl
   # then chunk_en/fr/ar + merge_chunks.py

3) Embed to Pinecone:
   python scripts/embed_upsert_v4.py --infile data/processed/chunks_all_clean.jsonl \
     --index polyglotai-v6 --batch_size 32 --namespace_by_lang true --show_progress true

4) Smoke test with rerank:
   python scripts/search_v5.py --lang all --top_k 20 --show_n 5 \
     --content_file data/processed/chunks_all_clean.jsonl
