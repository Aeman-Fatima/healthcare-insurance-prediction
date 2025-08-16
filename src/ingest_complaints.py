"""
Create a SMALL aggregate from the CFPB complaints dataset.

Input (CSV): --in data/raw/complaints-2025-08-16_14_37.csv  (your sample or full file)
Output:      --out data/processed/complaints_agg_by_state.csv

Aggregates per 'state':
- complaints_total
- pct_disputed
- pct_timely
- complaints_recent  (last 365 days, if date available)

Usage:
  python src/ingest_complaints.py \
    --in data/raw/complaints-2025-08-16_14_37.csv \
    --out data/processed/complaints_agg_by_state.csv \
    --max_rows 200000
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Map many possible CFPB header variants -> canonical names
CANONICAL_MAP = {
    # dates
    "Date received": "date_received",
    "date_received": "date_received",
    # company
    "Company": "company",
    "company": "company",
    # state
    "State": "state",
    "state": "state",
    # timely
    "Timely response?": "timely_response",
    "timely_response": "timely_response",
    # disputed
    "Consumer disputed?": "consumer_disputed",
    "consumer_disputed": "consumer_disputed",
    # public response (optional)
    "Company public response": "company_public_response",
    "company_public_response": "company_public_response",
}

REQUIRED = ["date_received", "state", "company", "timely_response", "consumer_disputed"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", default="data/processed/complaints_agg_by_state.csv")
    ap.add_argument("--max_rows", type=int, default=200000)
    args = ap.parse_args()

    # Read (cap rows for speed)
    df = pd.read_csv(args.inp, nrows=args.max_rows, low_memory=False)

    # Standardise headers
    rename = {c: CANONICAL_MAP[c] for c in df.columns if c in CANONICAL_MAP}
    df = df.rename(columns=rename)

    # Check essentials
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns after rename: {missing}\n"
            f"First 25 columns present: {list(df.columns)[:25]}"
        )

    # Clean state
    df["state"] = df["state"].astype(str).str.upper().fillna("NA")

    # Flags
    def to_yes_no(x):
        s = str(x).strip().lower()
        if s in ("yes", "y", "true", "1"): return 1
        if s in ("no", "n", "false", "0"): return 0
        return np.nan

    df["is_disputed"] = df["consumer_disputed"].apply(to_yes_no)
    df["is_timely"]   = df["timely_response"].apply(to_yes_no)

    # Recent flag (last 365 days) if we can parse date
    dt = pd.to_datetime(df["date_received"], errors="coerce", utc=True)
    if dt.notna().any():
        cutoff = dt.max() - pd.Timedelta(days=365)
        df["recent"] = (dt >= cutoff).astype(int)
    else:
        df["recent"] = 0

    # Aggregate by state
    agg = df.groupby("state", dropna=False).agg(
        complaints_total=("company", "size"),
        pct_disputed=("is_disputed", "mean"),
        pct_timely=("is_timely", "mean"),
        complaints_recent=("recent", "sum"),
    ).reset_index()

    # Clean NaNs and clip rates
    for c in ["pct_disputed", "pct_timely"]:
        agg[c] = agg[c].fillna(0).clip(0, 1)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    agg.to_csv(args.out, index=False)
    print(f"[OK] complaints aggregate → {args.out} ({len(agg)} rows)")

if __name__ == "__main__":
    main()
