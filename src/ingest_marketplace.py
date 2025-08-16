"""
Aggregate HHS marketplace data by state for lightweight joins.

Input:  data/raw/hhs_marketplace.csv  (or subset)
Output: data/processed/marketplace_agg_by_state.csv

Aggregates:
- avg_premium
- pct_bronze/silver/gold/platinum
"""
import argparse, pandas as pd, numpy as np
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", default="data/processed/marketplace_agg_by_state.csv")
    ap.add_argument("--max_rows", type=int, default=300000)
    args = ap.parse_args()

    # Adjust usecols to your file’s columns
    usecols = ["state","metal_level","individual_rate"]
    df = pd.read_csv(args.inp, usecols=usecols, nrows=args.max_rows)

    df["state"] = df["state"].fillna("NA").str.upper()
    df["metal_level"] = df["metal_level"].str.title().fillna("Unknown")

    # Premium stats
    prem = df.groupby("state").agg(avg_premium=("individual_rate","mean")).reset_index()

    # Metal distribution
    metal = (
        df.pivot_table(index="state", columns="metal_level", values="individual_rate", aggfunc="size", fill_value=0)
          .pipe(lambda t: t.div(t.sum(axis=1), axis=0))  # row-normalize to proportions
          .reset_index()
    )
    # Expected columns include Bronze/Silver/Gold/Platinum — fill missing:
    for c in ["Bronze","Silver","Gold","Platinum"]:
        if c not in metal.columns: metal[c] = 0.0

    agg = prem.merge(metal, on="state", how="left")

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    agg.to_csv(args.out, index=False)
    print(f"[OK] marketplace aggregate → {args.out} ({len(agg)} rows)")

if __name__ == "__main__":
    main()
