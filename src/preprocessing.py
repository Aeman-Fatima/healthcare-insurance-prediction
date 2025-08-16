from typing import List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from config import SplitConfig, Columns
import numpy as np

from pathlib import Path

STATE_TO_REGION = {
    "CA":"West","WA":"West","OR":"West","NV":"West","AZ":"West","UT":"West","CO":"West","NM":"West","ID":"West","MT":"West","WY":"West","AK":"West","HI":"West",
    "TX":"South","FL":"South","GA":"South","NC":"South","VA":"South","SC":"South","AL":"South","MS":"South","TN":"South","KY":"South","OK":"South","AR":"South","LA":"South","WV":"South","MD":"South","DC":"South","DE":"South",
    "NY":"East","NJ":"East","MA":"East","PA":"East","CT":"East","RI":"East","NH":"East","VT":"East","ME":"East",
    "IL":"North","MI":"North","OH":"North","MN":"North","WI":"North","IN":"North","IA":"North","MO":"North","ND":"North","SD":"North","NE":"North","KS":"North",
    "NA":"Unknown"
}


def load_and_merge(customers_path: str,
                   claims_path: str,
                   marketplace_path: str = None,
                   complaints_path: str = None,
                   renewals_path: str = None) -> pd.DataFrame:
    customers = pd.read_csv(customers_path)
    claims = pd.read_csv(claims_path)

    # Basic join: claims ↔ customers
    df = claims.merge(customers, how="left", on="SUB_ID")

    # Join CFPB complaints aggregate by Region (if file exists)
    comp_agg_path = Path("data/processed/complaints_agg_by_state.csv")
    if comp_agg_path.exists():
        comp = pd.read_csv(comp_agg_path)
        # expected cols: state, complaints_total, pct_disputed, pct_timely, complaints_recent
        comp.columns = [c.strip().lower() for c in comp.columns]
        if "state" in comp.columns:
            comp["region_from_state"] = comp["state"].str.upper().map(STATE_TO_REGION).fillna("Unknown")
            comp_reg = comp.groupby("region_from_state", dropna=False).agg(
                cfpb_complaints_total=("complaints_total","mean"),
                cfpb_pct_disputed=("pct_disputed","mean"),
                cfpb_pct_timely=("pct_timely","mean"),
                cfpb_complaints_recent=("complaints_recent","mean"),
            ).reset_index().rename(columns={"region_from_state":"Region"})
            if "Region" in df.columns:
                df = df.merge(comp_reg, on="Region", how="left")
            else:
                # if customers have State not Region, map it
                if "State" in df.columns:
                    df["Region"] = df["State"].str.upper().map(STATE_TO_REGION).fillna("Unknown")
                    df = df.merge(comp_reg, on="Region", how="left")
                else:
                    print("[WARN] No Region/State in customers to join complaints aggregate")
    else:
        print("[INFO] complaints aggregate not found at data/processed/complaints_agg_by_state.csv")


    # Optional: join synthetic renewals/loyalty by SUB_ID
    if renewals_path:
        try:
            renewals = pd.read_csv(renewals_path)
            df = df.merge(renewals, how="left", on="SUB_ID")
        except FileNotFoundError:
            print(f"[WARN] renewals file not found at {renewals_path}; continuing without it")

    # Stubs for marketplace/complaints if you want to add them later
    # if marketplace_path: ...
    # if complaints_path: ...
    return df

def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    # Standardise names
    if "Claim_Or_Rejected" in df.columns and "claim_outcome" not in df.columns:
        df = df.rename(columns={"Claim_Or_Rejected": "claim_outcome"})
    if "claim_amount" in df.columns and "Claim_Amount" not in df.columns:
        df = df.rename(columns={"claim_amount": "Claim_Amount"})
    if "Disease" in df.columns and "Disease_Name" not in df.columns:
        df = df.rename(columns={"Disease": "Disease_Name"})

    if "claim_outcome" not in df.columns:
        raise ValueError("Target column 'claim_outcome' missing after merge.")
    df = df.dropna(subset=["claim_outcome"])

    # Map target to binary
    mapping = {"Approved": 1, "Rejected": 0, 1: 1, 0: 0, "1": 1, "0": 0}
    df["claim_outcome"] = df["claim_outcome"].map(mapping)
    if df["claim_outcome"].isna().any():
        df = df[df["claim_outcome"].isin([0, 1])]
    df["claim_outcome"] = df["claim_outcome"].astype(int)

    # Numeric impute
    for col in ["Age", "Claim_Amount", "Loyalty_Points", "Previous_Claims", "Tenure_Years"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # Categorical impute
    for col in ["Gender", "Policy_Status", "Disease_Name", "Region", "Renewal_Status"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")
    return df

def make_preprocessor(cols: Columns, available_features: List[str]) -> ColumnTransformer:
    num_feats = [c for c in cols.numeric if c in available_features]
    cat_feats = [c for c in cols.categorical if c in available_features]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_feats),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
        ],
        remainder="drop",
    )

def split_data(df: pd.DataFrame, cols: Columns, split_cfg: SplitConfig):
    X = df[[c for c in list(cols.numeric) + list(cols.categorical) if c in df.columns]].copy()
    y = df[cols.target].copy()
    return train_test_split(
        X, y, test_size=split_cfg.test_size, random_state=split_cfg.random_state, stratify=y
    )
