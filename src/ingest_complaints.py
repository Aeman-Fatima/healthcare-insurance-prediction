"""
Create small per-state aggregates from CFPB complaints CSV.

Usage:
  python src/ingest_complaints.py --in data/raw/complaints.csv \
                                  --out data/processed/complaints_agg_by_state.csv \
                                  --max_rows 200000
"""
import argparse, pandas as pd, numpy as np
from pathlib import Path
from src.paths import PROCESSED_DIR

CANONICAL = {
    "Date received": "date_received", "date_received": "date_received",
    "Company": "company", "company": "company",
    "State": "state", "state": "state",
    "Timely response?": "timely_response", "timely_response": "timely_response",
    "Consumer disputed?": "consumer_disputed", "consumer_disputed": "consumer_disputed",
    "Company public response": "company_public_response", "company_public_response": "company_public_response",
}
REQUIRED = ["date_received","state","company","timely_response","consumer_disputed"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", default=str(PROCESSED_DIR / "complaints_agg_by_state.csv"))
    ap.add_argument("--max_rows", type=int, default=200000)
    args = ap.parse_args()

    df = pd.read_csv(args.inp, nrows=args.max_rows, low_memory=False)
    df = df.rename(columns={c: CANONICAL[c] for c in df.columns if c in CANONICAL})
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing} (present: {list(df.columns)[:25]})")

    def to01(x):
        s=str(x).strip().lower()
        return 1 if s in ("yes","y","true","1") else (0 if s in ("no","n","false","0") else np.nan)

    df["state"] = df["state"].astype(str).str.upper().fillna("NA")
    df["is_disputed"] = df["consumer_disputed"].apply(to01)
    df["is_timely"]   = df["timely_response"].apply(to01)

    dt = pd.to_datetime(df["date_received"], errors="coerce", utc=True)
    cutoff = dt.max() - pd.Timedelta(days=365) if dt.notna().any() else None
    df["recent"] = ((dt >= cutoff).astype(int) if cutoff else 0)

    agg = df.groupby("state", dropna=False).agg(
        complaints_total=("company","size"),
        pct_disputed=("is_disputed","mean"),
        pct_timely=("is_timely","mean"),
        complaints_recent=("recent","sum"),
    ).reset_index()
    for c in ("pct_disputed","pct_timely"):
        agg[c] = agg[c].fillna(0).clip(0,1)

    Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    agg.to_csv(args.out, index=False)
    print(f"[OK] complaints aggregate → {args.out} ({len(agg)})")

if __name__ == "__main__":
    main()
