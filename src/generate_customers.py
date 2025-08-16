"""
Create a synthetic customers table with demographics & policy info.

Output: data/raw/coil2000_customers.csv
Columns:
- SUB_ID (int)
- Age (int)
- Gender (str)            ['Male','Female','Other']
- Policy_Status (str)     ['Active','Inactive']
- Region (str)            ['North','South','East','West']
"""

import argparse, numpy as np, pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_customers", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", type=str, default="data/raw/coil2000_customers.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n = args.n_customers
    sub_id = np.arange(1, n + 1)

    age = np.clip(rng.normal(55, 14, n).round().astype(int), 18, 95)
    gender = rng.choice(["Male","Female","Other"], size=n, p=[0.48, 0.50, 0.02])
    policy = rng.choice(["Active","Inactive"], size=n, p=[0.75, 0.25])
    region = rng.choice(["North","South","East","West"], size=n, p=[0.26,0.24,0.25,0.25])

    df = pd.DataFrame({
        "SUB_ID": sub_id,
        "Age": age,
        "Gender": gender,
        "Policy_Status": policy,
        "Region": region
    })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[OK] customers → {out} ({len(df):,} rows)")

if __name__ == "__main__":
    main()
