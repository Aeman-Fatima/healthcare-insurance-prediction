#!/usr/bin/env python3
# src/ingest_pipeline.py
import argparse
from src.ingest_complaints import main as ingest_cfpb_cli

def main():
    ap = argparse.ArgumentParser(description="Ingest CFPB complaints (aggregate by state).")
    ap.add_argument("--in", dest="inp", required=True, help="Path to raw CFPB complaints CSV")
    ap.add_argument("--out", dest="out", required=False,
                    help="Output path for aggregated CSV (default inside data/processed/)")
    ap.add_argument("--max_rows", type=int, default=200000, help="Optional row cap for faster runs")
    args = ap.parse_args()

    # Reuse the CLI-style main from src.ingest_complaints
    # It reads argparse inside, so we forward via sys.argv-like semantics
    import sys
    argv_backup = sys.argv[:]
    try:
        sys.argv = ["ingest_complaints.py", "--in", args.inp]
        if args.out:
            sys.argv += ["--out", args.out]
        if args.max_rows is not None:
            sys.argv += ["--max_rows", str(args.max_rows)]
        ingest_cfpb_cli()  # will parse sys.argv internally
    finally:
        sys.argv = argv_backup

if __name__ == "__main__":
    main()
