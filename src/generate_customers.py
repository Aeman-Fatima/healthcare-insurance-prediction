import argparse, numpy as np, pandas as pd
from src.paths import RAW_DIR, ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_customers", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default=str(RAW_DIR / "coil2000_customers.csv"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sub_id = np.arange(1, args.n_customers + 1)
    age = np.clip(rng.normal(55, 14, args.n_customers).round().astype(int), 18, 95)
    gender = rng.choice(["Male","Female","Other"], size=args.n_customers, p=[0.48,0.50,0.02])
    policy = rng.choice(["Active","Inactive"], size=args.n_customers, p=[0.75,0.25])
    region = rng.choice(["North","South","East","West"], size=args.n_customers, p=[0.26,0.24,0.25,0.25])

    df = pd.DataFrame({"SUB_ID": sub_id,"Age": age,"Gender": gender,"Policy_Status": policy,"Region": region})
    ensure_dir(RAW_DIR)
    df.to_csv(args.out, index=False)
    print(f"[OK] customers → {args.out} ({len(df)})")

if __name__ == "__main__":
    main()
