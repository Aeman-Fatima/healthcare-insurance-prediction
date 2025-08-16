# src/generate_loyalty.py
"""
Generate a synthetic renewals/loyalty dataset to join with claims/customers.
This avoids bundling private CSVs while keeping your pipeline reproducible.

Output schema (CSV):
- SUB_ID (int): subscriber/customer id for joining
- Tenure_Years (float): years since first policy
- Loyalty_Points (int): reward points (roughly tenure * engagement factor)
- Previous_Claims (int): number of past claims
- Renewal_Status (str): 'Renewed'/'Not_Renewed' (optional, for retention analysis)

Usage:
  python src/generate_loyalty.py \
    --n_customers 8000 \
    --seed 42 \
    --out data/raw/renewals_loyalty.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def clipped_normal(mean, sd, size, low=None, high=None):
    x = np.random.normal(mean, sd, size)
    if low is not None:  x = np.maximum(x, low)
    if high is not None: x = np.minimum(x, high)
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_customers", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/raw/renewals_loyalty.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n = args.n_customers

    # SUB_ID to join with your customers/claims tables
    sub_ids = np.arange(1, n + 1)

    # Tenure ~ skewed to lower years but with a tail (0-15 years)
    tenure_years = clipped_normal(mean=4.0, sd=3.0, size=n, low=0.0, high=15.0)

    # Engagement factor to introduce heterogeneity in loyalty
    engagement = rng.beta(a=2.0, b=3.0, size=n)  # 0..1

    # Loyalty_Points ~ tenure * engagement with noise; clamp to [0, 10_000]
    loyalty_points = (tenure_years * (200 + 800 * engagement) +
                      rng.normal(0, 100, n))
    loyalty_points = np.clip(loyalty_points, 0, 10_000).astype(int)

    # Previous_Claims: zero-inflated Poisson-like
    zero_mask = rng.random(n) < 0.55
    base_lambda = 0.8 + 1.2 * engagement + 0.05 * tenure_years
    prev_claims = rng.poisson(lam=np.maximum(base_lambda, 0.05))
    prev_claims[zero_mask] = 0
    prev_claims = np.clip(prev_claims, 0, 12)

    # Renewal probability rises with tenure & loyalty, falls with many past claims
    logit = (
        -0.8                                # base
        + 0.18 * tenure_years               # more tenure -> more likely to renew
        + 0.00015 * loyalty_points          # more loyalty -> more likely
        - 0.25 * prev_claims                # many claims -> less likely
    )
    p_renew = 1 / (1 + np.exp(-logit))
    renewal = (rng.random(n) < p_renew)
    renewal_status = np.where(renewal, "Renewed", "Not_Renewed")

    df = pd.DataFrame({
        "SUB_ID": sub_ids,
        "Tenure_Years": np.round(tenure_years, 2),
        "Loyalty_Points": loyalty_points,
        "Previous_Claims": prev_claims,
        "Renewal_Status": renewal_status
    })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[OK] Wrote {len(df):,} rows → {out_path}")

if __name__ == "__main__":
    main()
