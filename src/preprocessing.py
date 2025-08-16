from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from config import SplitConfig, Columns
import numpy as np

def load_and_merge(customers_path: str,
                   claims_path: str,
                   marketplace_path: str,
                   complaints_path: str) -> pd.DataFrame:
    """Load CSVs and perform a minimal, domain-appropriate merge.
    Replace this logic to match your actual keys & joins described in Parts A–C.
    """
    customers = pd.read_csv(customers_path)
    claims = pd.read_csv(claims_path)
    # Minimal example: assume claims has SUB_ID / customer link
    df = claims.merge(customers, how="left", on="SUB_ID")

    # Marketplace & complaints can be joined by region/plan tiers if available;
    # else, aggregate and join back (feature engineering opportunity).
    # Placeholder: no-op
    # marketplace = pd.read_csv(marketplace_path)
    # complaints = pd.read_csv(complaints_path)

    return df

def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning consistent with Part B: impute, rename, create features."""
    # Example standardisations — adjust to your real columns
    if "Claim_Or_Rejected" in df.columns:
        df = df.rename(columns={"Claim_Or_Rejected": "claim_outcome"})
    if "claim_amount" in df.columns:
        df = df.rename(columns={"claim_amount": "Claim_Amount"})
    if "Disease_Name" not in df.columns and "Disease" in df.columns:
        df = df.rename(columns={"Disease": "Disease_Name"})

    # Drop rows with missing target
    df = df.dropna(subset=["claim_outcome"])

    # Map target to binary
    df["claim_outcome"] = df["claim_outcome"].map({"Approved": 1, "Rejected": 0}).astype(int)

    # Example imputation (median for numeric)
    for col in ["Age", "Claim_Amount", "Loyalty_Points", "Previous_Claims"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # Example mode imputation for categoricals
    for col in ["Gender", "Policy_Status", "Disease_Name", "Region"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")

    return df

def make_preprocessor(cols: Columns) -> ColumnTransformer:
    numeric_features = [c for c in cols.numeric if c in df_feature_guard]
    categorical_features = [c for c in cols.categorical if c in df_feature_guard]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor

# Small guard to avoid NameError inside make_preprocessor; set in split_data()
df_feature_guard = []

def split_data(df: pd.DataFrame, cols: Columns, split_cfg: SplitConfig):
    global df_feature_guard
    df_feature_guard = df.columns.tolist()

    X = df[[c for c in list(cols.numeric) + list(cols.categorical) if c in df.columns]].copy()
    y = df[cols.target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_cfg.test_size, random_state=split_cfg.random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
