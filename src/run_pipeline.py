#!/usr/bin/env python3
"""
Healthcare Insurance Pipeline — single-file reproducible script.

Usage (from repo root):
  python run_pipeline.py \
    --customers data/raw/coil2000_customers.csv \
    --claims data/raw/claims.csv \
    --renewals data/raw/renewals_loyalty.csv \
    --complaints_agg data/processed/complaints_agg_by_state.csv \
    --out results

If the CSVs are missing, this script can generate synthetic versions:
  python run_pipeline.py --generate --out results
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
)

# ------------------------- utils -------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_fig(fig, path: Path):
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

# -------------------- synthetic generators --------------------

def gen_customers(n_customers=8000, seed=123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sub_id = np.arange(1, n_customers + 1)
    age = np.clip(rng.normal(55, 14, n_customers).round().astype(int), 18, 95)
    gender = rng.choice(["Male","Female","Other"], size=n_customers, p=[0.48,0.50,0.02])
    policy = rng.choice(["Active","Inactive"], size=n_customers, p=[0.75,0.25])
    region = rng.choice(["North","South","East","West"], size=n_customers, p=[0.26,0.24,0.25,0.25])
    return pd.DataFrame({"SUB_ID": sub_id, "Age": age, "Gender": gender,
                         "Policy_Status": policy, "Region": region})

DISEASES = ["Hypertension","Diabetes","Asthma","Cardiac","Orthopedic","Oncology","Ophthalmic","ENT"]

def gen_claims(n_claims=10000, n_customers=8000, seed=456) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sub_ids = rng.integers(1, n_customers + 1, size=n_claims)
    disease = rng.choice(DISEASES, size=n_claims,
                         p=[0.18,0.16,0.12,0.14,0.12,0.10,0.10,0.08])
    base = rng.lognormal(mean=8.5, sigma=0.7, size=n_claims)  # right-skewed
    bump = np.where((disease=="Oncology") | (disease=="Orthopedic"), 1.35, 1.0)
    claim_amount = (base * bump).round(2)
    disease_penalty = np.where(disease=="Oncology", -0.35,
                        np.where(disease=="Cardiac", -0.15, 0.0))
    logit = 0.6 + disease_penalty - 0.00005*claim_amount
    p_approved = 1/(1+np.exp(-logit))
    approved = rng.random(n_claims) < p_approved
    outcome = np.where(approved, "Approved", "Rejected")
    return pd.DataFrame({"SUB_ID": sub_ids, "Disease_Name": disease,
                         "Claim_Amount": claim_amount, "claim_outcome": outcome})

def gen_renewals(n_customers=8000, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sub_ids = np.arange(1, n_customers + 1)
    tenure_years = np.clip(rng.normal(4.0, 3.0, n_customers), 0.0, 15.0)
    engagement = rng.beta(a=2.0, b=3.0, size=n_customers)
    loyalty_points = (tenure_years * (200 + 800 * engagement) + rng.normal(0, 100, n_customers))
    loyalty_points = np.clip(loyalty_points, 0, 10000).astype(int)
    zero_mask = rng.random(n_customers) < 0.55
    base_lambda = 0.8 + 1.2 * engagement + 0.05 * tenure_years
    prev_claims = rng.poisson(lam=np.maximum(base_lambda, 0.05))
    prev_claims[zero_mask] = 0
    prev_claims = np.clip(prev_claims, 0, 12)
    logit = -0.8 + 0.18*tenure_years + 0.00015*loyalty_points - 0.25*prev_claims
    p_renew = 1/(1+np.exp(-logit))
    renewal = (rng.random(n_customers) < p_renew)
    renewal_status = np.where(renewal, "Renewed", "Not_Renewed")
    return pd.DataFrame({"SUB_ID": sub_ids,
                         "Tenure_Years": np.round(tenure_years, 2),
                         "Loyalty_Points": loyalty_points,
                         "Previous_Claims": prev_claims,
                         "Renewal_Status": renewal_status})

# -------------------- optional complaints aggregate --------------------

STATE_TO_REGION = {
    "CA":"West","WA":"West","OR":"West","NV":"West","AZ":"West","UT":"West","CO":"West","NM":"West","ID":"West","MT":"West","WY":"West","AK":"West","HI":"West",
    "TX":"South","FL":"South","GA":"South","NC":"South","VA":"South","SC":"South","AL":"South","MS":"South","TN":"South","KY":"South","OK":"South","AR":"South","LA":"South","WV":"South","MD":"South","DC":"South","DE":"South",
    "NY":"East","NJ":"East","MA":"East","PA":"East","CT":"East","RI":"East","NH":"East","VT":"East","ME":"East",
    "IL":"North","MI":"North","OH":"North","MN":"North","WI":"North","IN":"North","IA":"North","MO":"North","ND":"North","SD":"North","NE":"North","KS":"North",
    "NA":"Unknown"
}

def attach_complaints_region_features(df: pd.DataFrame, complaints_agg_path: Path) -> pd.DataFrame:
    if not complaints_agg_path or not complaints_agg_path.exists():
        print("[INFO] complaints aggregate not found; continuing without it")
        return df
    comp = pd.read_csv(complaints_agg_path)
    comp.columns = [c.strip().lower() for c in comp.columns]
    if "state" not in comp.columns:
        print("[WARN] 'state' column missing in complaints aggregate; skipping join")
        return df
    comp["region_from_state"] = comp["state"].str.upper().map(STATE_TO_REGION).fillna("Unknown")
    comp_reg = comp.groupby("region_from_state", dropna=False).agg(
        cfpb_complaints_total=("complaints_total","mean"),
        cfpb_pct_disputed=("pct_disputed","mean"),
        cfpb_pct_timely=("pct_timely","mean"),
        cfpb_complaints_recent=("complaints_recent","mean"),
    ).reset_index().rename(columns={"region_from_state":"Region"})
    if "Region" not in df.columns:
        print("[WARN] customers lack 'Region'; cannot join complaints features")
        return df
    out = df.merge(comp_reg, on="Region", how="left")
    return out

# -------------------- preprocessing & modeling --------------------

NUMERIC_COLS = [
    "Age","Claim_Amount","Loyalty_Points","Previous_Claims","Tenure_Years",
    "cfpb_complaints_total","cfpb_pct_disputed","cfpb_pct_timely","cfpb_complaints_recent"
]
CATEGORICAL_COLS = ["Gender","Policy_Status","Disease_Name","Region","Renewal_Status"]
TARGET_COL = "claim_outcome"

def clean(df: pd.DataFrame) -> pd.DataFrame:
    # target to {0,1}
    mapping = {"Approved": 1, "Rejected": 0, 1:1, 0:0, "1":1, "0":0}
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COL}'")
    df[TARGET_COL] = df[TARGET_COL].map(mapping)
    df = df[df[TARGET_COL].isin([0,1])].copy()
    # numeric impute
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median())
    # categorical impute
    for c in CATEGORICAL_COLS:
        if c in df.columns:
            df[c] = df[c].astype("category")
            mode = df[c].mode()
            df[c] = df[c].fillna(mode.iloc[0] if not mode.empty else "Unknown")
    return df

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_feats = [c for c in NUMERIC_COLS if c in X.columns]
    cat_feats = [c for c in CATEGORICAL_COLS if c in X.columns]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_feats),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
        ],
        remainder="drop",
    )

def build_models(preprocessor) -> dict:
    models = {
        "LogisticRegression": Pipeline([("preprocess", preprocessor), ("model", LogisticRegression(max_iter=1000))]),
        "RandomForest":       Pipeline([("preprocess", preprocessor), ("model", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))]),
        "XGBoost":            Pipeline([("preprocess", preprocessor), ("model", XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=42, eval_metric="logloss"
        ))]),
    }
    return models

def evaluate_and_plot(clf, X_test, y_test, out_dir: Path, name: str) -> dict:
    # proba / decision
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)[:, 1]
    else:
        df = clf.decision_function(X_test)
        y_prob = (df - df.min()) / (df.max() - df.min() + 1e-9)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) == 2 else float("nan")

    # ROC (plot to the figure we save)
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
    ax.set_title(f"ROC – {name}")
    save_fig(fig, out_dir / f"{name}_roc.png")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(f"Confusion Matrix – {name}")
    fig.colorbar(im, ax=ax)
    ticks = np.arange(2)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(['Rejected','Approved']); ax.set_yticklabels(['Rejected','Approved'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    save_fig(fig, out_dir / f"{name}_confusion.png")

    return {"model": name, "accuracy": acc, "f1": f1, "roc_auc": roc}

# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--customers", default="data/raw/coil2000_customers.csv")
    ap.add_argument("--claims", default="data/raw/claims.csv")
    ap.add_argument("--renewals", default="data/raw/renewals_loyalty.csv")
    ap.add_argument("--complaints_agg", default="data/processed/complaints_agg_by_state.csv")
    ap.add_argument("--generate", action="store_true", help="Generate synthetic CSVs if missing.")
    ap.add_argument("--out", default="results")
    ap.add_argument("--test_size", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out) / "figures"
    tables_dir = Path(args.out) / "tables"
    ensure_dir(out_dir); ensure_dir(tables_dir)

    # generate synthetic if needed / requested
    if args.generate or not Path(args.customers).exists():
        print("[INFO] generating customers.csv")
        dfc = gen_customers()
        ensure_dir(Path(args.customers).parent)
        dfc.to_csv(args.customers, index=False)
    if args.generate or not Path(args.claims).exists():
        print("[INFO] generating claims.csv")
        dfl = gen_claims()
        ensure_dir(Path(args.claims).parent)
        dfl.to_csv(args.claims, index=False)
    if args.generate or not Path(args.renewals).exists():
        print("[INFO] generating renewals_loyalty.csv")
        dfr = gen_renewals()
        ensure_dir(Path(args.renewals).parent)
        dfr.to_csv(args.renewals, index=False)

    # load
    customers = pd.read_csv(args.customers)
    claims = pd.read_csv(args.claims)
    renewals = pd.read_csv(args.renewals)

    # merge base
    df = claims.merge(customers, on="SUB_ID", how="left")
    df = df.merge(renewals, on="SUB_ID", how="left")

    # optional complaints features
    df = attach_complaints_region_features(df, Path(args.complaints_agg))

    # clean
    df = clean(df)

    # split
    feat_cols = [c for c in NUMERIC_COLS + CATEGORICAL_COLS if c in df.columns]
    X = df[feat_cols].copy()
    y = df[TARGET_COL].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # models
    pre = build_preprocessor(X_train)
    models = build_models(pre)

    # train/eval
    rows = []
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        row = evaluate_and_plot(pipe, X_test, y_test, out_dir, name)
        rows.append(row)
        print(f"{name}: acc={row['accuracy']:.3f} f1={row['f1']:.3f} roc_auc={row['roc_auc']:.3f}")

    # save table
    comp = pd.DataFrame(rows).sort_values(by="roc_auc", ascending=False)
    comp.to_csv(tables_dir / "model_comparison.csv", index=False)
    print(f"[OK] saved {tables_dir / 'model_comparison.csv'}")

if __name__ == "__main__":
    main()
