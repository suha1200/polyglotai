import json

file = "data/processed/chunks_all.jsonl"

shortest = []
with open(file, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        content = row.get("content", "").strip()
        if not content:
            continue
        shortest.append((len(content), content[:80], row.get("chunk_id")))

# Sort by length, take the first 20
shortest = sorted(shortest, key=lambda x: x[0])[:20]

for length, preview, cid in shortest:
    print(f"{length:3} chars | {cid} | {preview}")
