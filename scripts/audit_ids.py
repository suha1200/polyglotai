import json
from collections import Counter, defaultdict
from pathlib import Path

INFILE = Path("data/processed/chunks_all.jsonl")

def infer_language(pack_id, chunk_id):
    p = (pack_id or "")
    c = (chunk_id or "")
    cl = c.lower(); pl = p.lower()
    if c.startswith("AR-") or c.startswith("AR_") or p.startswith("AR-") or p.startswith("AR_"):
        return "ar"
    if cl.startswith("french_pack_") or pl.startswith("french_pack_"):
        return "fr"
    if cl.startswith("english_pack_") or pl.startswith("english_pack_"):
        return "en"
    return "unknown"

ids = []
langs = []
with INFILE.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        row = json.loads(line)
        cid = row.get("chunk_id")
        if not cid: continue
        ids.append(cid)
        langs.append(infer_language(row.get("pack_id"), cid))

total = len(ids)
ctr = Counter(ids)
dups = [k for k,v in ctr.items() if v > 1]
print(f"Total lines: {total}")
print(f"Unique chunk_ids: {len(ctr)}")
print(f"Duplicate IDs: {len(dups)}")
if dups[:10]:
    print("Examples of duplicate IDs:", dups[:10])

# bonus: expected per-namespace unique counts (what Pinecone should show if no overwrites)
lang_counts = defaultdict(int)
seen = set()
with INFILE.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        row = json.loads(line)
        cid = row.get("chunk_id")
        if not cid or cid in seen: continue
        seen.add(cid)
        lang = infer_language(row.get("pack_id"), cid)
        lang_counts[lang] += 1

print("Expected unique per language:", dict(lang_counts))
