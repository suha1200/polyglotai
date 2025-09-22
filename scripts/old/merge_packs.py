# scripts/merge_jsonl.py
import argparse, json, pathlib, hashlib

def norm_record(rec, source):
    # ensure required keys exist / fix types
    rec.setdefault("pack_id", "UNKNOWN")
    rec.setdefault("language", "xx")
    rec.setdefault("book", "UNKNOWN_BOOK")
    rec.setdefault("section_title", "")
    rec.setdefault("section_path", [])
    if not isinstance(rec["section_path"], list):
        rec["section_path"] = [str(rec["section_path"])]
    # coerce page to int if possible
    try:
        rec["page"] = int(rec.get("page", 0))
    except Exception:
        rec["page"] = 0
    rec.setdefault("content", "")
    rec["source_file"] = source
    return rec

def rec_key(rec):
    # stable de-dup key (tune if you like)
    h = hashlib.md5()
    h.update((rec["book"] + "||" + rec["section_title"] + "||" + rec["content"]).encode("utf-8", "ignore"))
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", required=True, help="output .jsonl")
    ap.add_argument("inputs", nargs="+", help="input .jsonl files or globs")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # expand globs
    paths = []
    for pattern in args.inputs:
        paths.extend(pathlib.Path().glob(pattern))
    if not paths:
        raise SystemExit("No input files matched.")

    seen = set()
    count_in = 0
    count_out = 0

    with out.open("w", encoding="utf-8") as fo:
        for p in paths:
            with p.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    count_in += 1
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    rec = norm_record(rec, source=str(p))
                    k = rec_key(rec)
                    if k in seen:
                        continue
                    seen.add(k)
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    count_out += 1
    print(f"✅ merged {len(paths)} files: {count_in} → {count_out} records into {out}")

if __name__ == "__main__":
    main()
