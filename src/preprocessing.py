import pandas as pd
from pathlib import Path
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.paths import RAW_DIR, PROCESSED_DIR

STATE_TO_REGION = {
    "CA":"West","WA":"West","OR":"West","NV":"West","AZ":"West","UT":"West","CO":"West","NM":"West","ID":"West","MT":"West","WY":"West","AK":"West","HI":"West",
    "TX":"South","FL":"South","GA":"South","NC":"South","VA":"South","SC":"South","AL":"South","MS":"South","TN":"South","KY":"South","OK":"South","AR":"South","LA":"South","WV":"South","MD":"South","DC":"South","DE":"South",
    "NY":"East","NJ":"East","MA":"East","PA":"East","CT":"East","RI":"East","NH":"East","VT":"East","ME":"East",
    "IL":"North","MI":"North","OH":"North","MN":"North","WI":"North","IN":"North","IA":"North","MO":"North","ND":"North","SD":"North","NE":"North","KS":"North",
    "NA":"Unknown"
}

NUMERIC = [
    "Age","Claim_Amount","Loyalty_Points","Previous_Claims","Tenure_Years",
    "cfpb_complaints_total","cfpb_pct_disputed","cfpb_pct_timely","cfpb_complaints_recent"
]
CATEG = ["Gender","Policy_Status","Disease_Name","Region","Renewal_Status"]
TARGET = "claim_outcome"

def load_and_merge(customers_path: str, claims_path: str, renewals_path: str) -> pd.DataFrame:
    customers = pd.read_csv(customers_path)
    claims = pd.read_csv(claims_path)
    renewals = pd.read_csv(renewals_path)
    df = claims.merge(customers, on="SUB_ID", how="left").merge(renewals, on="SUB_ID", how="left")

    # Optional complaints join
    comp_path = PROCESSED_DIR / "complaints_agg_by_state.csv"
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        comp.columns = [c.strip().lower() for c in comp.columns]
        if "state" in comp.columns:
            comp["Region_from_state"] = comp["state"].str.upper().map(STATE_TO_REGION).fillna("Unknown")
            comp_reg = comp.groupby("Region_from_state").agg(
                cfpb_complaints_total=("complaints_total","mean"),
                cfpb_pct_disputed=("pct_disputed","mean"),
                cfpb_pct_timely=("pct_timely","mean"),
                cfpb_complaints_recent=("complaints_recent","mean"),
            ).reset_index().rename(columns={"Region_from_state":"Region"})
            if "Region" in df.columns:
                df = df.merge(comp_reg, on="Region", how="left")
    return df

def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    # normalize target to {0,1}
    map_t = {"Approved":1, "Rejected":0, 1:1, 0:0, "1":1, "0":0}
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column '{TARGET}'")
    df[TARGET] = df[TARGET].map(map_t)
    df = df[df[TARGET].isin([0,1])].copy()

    # numeric impute
    for c in NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median())

    # categorical impute
    for c in CATEG:
        if c in df.columns:
            df[c] = df[c].astype("category")
            mode = df[c].mode()
            df[c] = df[c].fillna(mode.iloc[0] if not mode.empty else "Unknown")
    return df

def split_data(df: pd.DataFrame, test_size=0.30, seed=42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df[[c for c in NUMERIC + CATEG if c in df.columns]].copy()
    y = df[TARGET].copy()
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

def make_preprocessor(X_cols: List[str]) -> ColumnTransformer:
    num_feats = [c for c in NUMERIC if c in X_cols]
    cat_feats = [c for c in CATEG if c in X_cols]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_feats),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
        ],
        remainder="drop",
    )
