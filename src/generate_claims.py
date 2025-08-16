"""
Create a synthetic claims table with disease, amounts, and outcomes.

Output: data/raw/claims.csv
Columns:
- SUB_ID (int)                 join key to customers
- Disease_Name (str)
- Claim_Amount (float)
- claim_outcome (str)          ['Approved','Rejected']
"""

import argparse, numpy as np, pandas as pd
from pathlib import Path

DISEASES = ["Hypertension","Diabetes","Asthma","Cardiac","Orthopedic","Oncology","Ophthalmic","ENT"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_claims", type=int, default=10000)
    ap.add_argument("--n_customers", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=456)
    ap.add_argument("--out", type=str, default="data/raw/claims.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # random SUB_IDs (some customers have multiple claims)
    sub_ids = rng.integers(1, args.n_customers + 1, size=args.n_claims)

    disease = rng.choice(DISEASES, size=args.n_claims,
                         p=[0.18,0.16,0.12,0.14,0.12,0.10,0.10,0.08])

    # claim amounts: right-skewed
    base = rng.lognormal(mean=8.5, sigma=0.7, size=args.n_claims)   # ~ 4k–12k typical
    # bump oncology/orthopedic costs
    bump = np.where((disease=="Oncology") | (disease=="Orthopedic"), 1.35, 1.0)
    claim_amount = (base * bump).round(2)

    # outcome probability mildly depends on amount and disease
    # higher amount => slightly lower approval
    # some diseases (e.g., oncology) slightly lower approval
    disease_penalty = np.where(disease=="Oncology", -0.35,
                        np.where(disease=="Cardiac", -0.15, 0.0))
    logit = 0.6 + disease_penalty - 0.00005*claim_amount
    p_approved = 1/(1+np.exp(-logit))
    approved = rng.random(args.n_claims) < p_approved
    outcome = np.where(approved, "Approved", "Rejected")

    df = pd.DataFrame({
        "SUB_ID": sub_ids,
        "Disease_Name": disease,
        "Claim_Amount": claim_amount,
        "claim_outcome": outcome
    })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[OK] claims → {out} ({len(df):,} rows)")

if __name__ == "__main__":
    main()
