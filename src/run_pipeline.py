#!/usr/bin/env python3
# src/run_pipeline.py
import argparse
from pathlib import Path

from src.paths import RAW_DIR, RESULTS_DIR, FIGS_DIR, TABLES_DIR, ensure_dir
from src.generate_customers import main as gen_customers
from src.generate_claims   import main as gen_claims
from src.generate_loyalty  import main as gen_loyalty

from src.preprocessing import (
    load_and_merge, clean_and_engineer, split_data, make_preprocessor
)
from src.visualize_eda import generate_eda
from src.modelling import build_models
from src.evaluation import evaluate_and_plot, save_comparison
from src.prescriptions import recommend_actions

def main():
    ap = argparse.ArgumentParser(description="Healthcare Insurance Prediction Pipeline (A→D)")
    ap.add_argument(
        "--steps", default="all",
        choices=["all","a","b","c","d","eda","model","prescriptions"],
        help="Which stage(s) to run"
    )
    ap.add_argument("--generate", action="store_true",
                    help="Generate synthetic datasets if missing")
    ap.add_argument("--customers_path", default=str(RAW_DIR / "coil2000_customers.csv"))
    ap.add_argument("--claims_path",    default=str(RAW_DIR / "claims.csv"))
    ap.add_argument("--renewals_path",  default=str(RAW_DIR / "renewals_loyalty.csv"))
    ap.add_argument("--test_size", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Ensure output folders exist
    for p in (RESULTS_DIR, FIGS_DIR, TABLES_DIR):
        ensure_dir(p)

    # ---------- Part A: data creation (synthetic) ----------
    if args.steps in ("all","a"):
        if args.generate or not Path(args.customers_path).exists():
            gen_customers()
        if args.generate or not Path(args.claims_path).exists():
            gen_claims()
        if args.generate or not Path(args.renewals_path).exists():
            gen_loyalty()

    # Load once if downstream steps will need data
    need_df = args.steps in ("all","a","b","c","d","eda","model","prescriptions")
    if need_df:
        df = load_and_merge(args.customers_path, args.claims_path, args.renewals_path)
        df = clean_and_engineer(df)

    # ---------- Part B/C: EDA ----------
    if args.steps in ("all","b","c","eda"):
        print("[EDA] Saving figures to results/eda/")
        generate_eda(df)

    # ---------- Part C/D: Modelling & evaluation ----------
    if args.steps in ("all","c","model","d","prescriptions"):
        X_train, X_test, y_train, y_test = split_data(df, test_size=args.test_size, seed=args.seed)
        pre = make_preprocessor(X_train.columns.tolist())
        models = build_models(pre)

        rows = []
        for name, pipe in models.items():
            pipe.fit(X_train, y_train)
            row = evaluate_and_plot(pipe, X_test, y_test, name)
            rows.append(row)
            print(f"{name}: acc={row['accuracy']:.3f} f1={row['f1']:.3f} roc_auc={row['roc_auc']:.3f}")

        comp_csv = save_comparison(rows)
        print(f"[OK] model comparison → {comp_csv}")

        # ---------- Part D: Prescriptions ----------
        if args.steps in ("all","d","prescriptions"):
            best = sorted(rows, key=lambda r: (r.get("roc_auc", 0), r.get("f1", 0)), reverse=True)[0]
            print("\n=== Prescriptions (draft) ===")
            for i, rec in enumerate(recommend_actions(best), 1):
                print(f"{i}. {rec}")

if __name__ == "__main__":
    main()
