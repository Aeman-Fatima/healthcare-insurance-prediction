import argparse, numpy as np, pandas as pd
from src.paths import RAW_DIR, ensure_dir

DISEASES = ["Hypertension","Diabetes","Asthma","Cardiac","Orthopedic","Oncology","Ophthalmic","ENT"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_claims", type=int, default=10000)
    ap.add_argument("--n_customers", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=456)
    ap.add_argument("--out", default=str(RAW_DIR / "claims.csv"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sub_ids = rng.integers(1, args.n_customers + 1, size=args.n_claims)
    disease = rng.choice(DISEASES, size=args.n_claims,
                         p=[0.18,0.16,0.12,0.14,0.12,0.10,0.10,0.08])
    base = rng.lognormal(mean=8.5, sigma=0.7, size=args.n_claims)
    bump = (disease == "Oncology") | (disease == "Orthopedic")
    claim_amount = (base * (1.35*bump + (~bump))).round(2)

    disease_penalty = np.select(
        [disease=="Oncology", disease=="Cardiac"],
        [-0.35, -0.15], default=0.0
    )
    logit = 0.6 + disease_penalty - 0.00005*claim_amount
    p_approved = 1/(1+np.exp(-logit))
    approved = rng.random(args.n_claims) < p_approved
    outcome = np.where(approved, "Approved", "Rejected")

    df = pd.DataFrame({"SUB_ID": sub_ids, "Disease_Name": disease,
                       "Claim_Amount": claim_amount, "claim_outcome": outcome})
    ensure_dir(RAW_DIR); df.to_csv(args.out, index=False)
    print(f"[OK] claims → {args.out} ({len(df)})")

if __name__ == "__main__":
    main()
