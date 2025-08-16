import argparse, os
from config import Paths, SplitConfig, Columns
from preprocessing import load_and_merge, clean_and_engineer, split_data, make_preprocessor
from modelling import build_models
from evaluation import evaluate_and_plot, save_comparison
from utils import ensure_dir

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--customers_path", required=True)
    p.add_argument("--claims_path", required=True)
    p.add_argument("--marketplace_path", default=None)
    p.add_argument("--complaints_path", default=None)
    p.add_argument("--renewals_path", default="data/raw/renewals_loyalty.csv")
    p.add_argument("--output_dir", default="results")
    return p.parse_args()

def main():
    args = parse_args()

    ensure_dir(args.output_dir)
    figs_dir = os.path.join(args.output_dir, "figures")
    tables_dir = os.path.join(args.output_dir, "tables")
    ensure_dir(figs_dir); ensure_dir(tables_dir)

    # 1) Load & preprocess
    df = load_and_merge(
        customers_path=args.customers_path,
        claims_path=args.claims_path,
        marketplace_path=args.marketplace_path,
        complaints_path=args.complaints_path,
        renewals_path=args.renewals_path,
    )
    df = clean_and_engineer(df)

    # 2) Split
    split_cfg = SplitConfig()
    cols = Columns()
    X_train, X_test, y_train, y_test = split_data(df, cols, split_cfg)

    # 3) Preprocessor + models
    preprocessor = make_preprocessor(cols, X_train.columns.tolist())
    models = build_models(preprocessor)

    # 4) Train/evaluate
    rows = []
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        row = evaluate_and_plot(pipe, X_test, y_test, figs_dir, name)
        rows.append(row)
        print(f"{name}: acc={row['accuracy']:.3f} f1={row['f1']:.3f} roc_auc={row['roc_auc']:.3f}")

    # 5) Save table
    out_csv = os.path.join(tables_dir, "model_comparison.csv")
    save_comparison(rows, out_csv)
    print(f"Saved model comparison to {out_csv}")

if __name__ == "__main__":
    main()
