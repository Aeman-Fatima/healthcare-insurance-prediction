import argparse, numpy as np, pandas as pd
from src.paths import RAW_DIR, ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_customers", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(RAW_DIR / "renewals_loyalty.csv"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sub_ids = np.arange(1, args.n_customers + 1)
    tenure = np.clip(rng.normal(4.0, 3.0, args.n_customers), 0.0, 15.0)
    engagement = rng.beta(2.0, 3.0, args.n_customers)
    loyalty = np.clip(tenure * (200 + 800*engagement) + rng.normal(0,100,args.n_customers), 0, 10000).astype(int)
    base_lambda = np.maximum(0.8 + 1.2*engagement + 0.05*tenure, 0.05)
    prev_claims = np.clip(rng.poisson(base_lambda), 0, 12)
    zero_mask = rng.random(args.n_customers) < 0.55
    prev_claims[zero_mask] = 0
    logit = -0.8 + 0.18*tenure + 0.00015*loyalty - 0.25*prev_claims
    p_renew = 1/(1+np.exp(-logit))
    renewal = (rng.random(args.n_customers) < p_renew)
    status = np.where(renewal, "Renewed", "Not_Renewed")

    df = pd.DataFrame({"SUB_ID": sub_ids,"Tenure_Years": tenure.round(2),
                       "Loyalty_Points": loyalty,"Previous_Claims": prev_claims,"Renewal_Status": status})
    ensure_dir(RAW_DIR); df.to_csv(args.out, index=False)
    print(f"[OK] renewals/loyalty → {args.out} ({len(df)})")

if __name__ == "__main__":
    main()
