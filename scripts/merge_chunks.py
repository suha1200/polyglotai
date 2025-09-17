import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="Merge multiple chunk files into one")
    p.add_argument("--outfile", required=True, help="Path to merged output file")
    p.add_argument("inputs", nargs="+", help="List of chunk files to merge (e.g., chunks_en.jsonl chunks_fr.jsonl chunks_ar.jsonl)")
    args = p.parse_args()

    outfile = Path(args.outfile)

    with outfile.open("w", encoding="utf-8") as fout:
        for infile in args.inputs:
            infile = Path(infile)
            with infile.open("r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)

    print(f"Merged {len(args.inputs)} files into {outfile}")

if __name__ == "__main__":
    main()
