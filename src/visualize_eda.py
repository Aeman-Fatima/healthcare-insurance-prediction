import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.paths import EDA_DIR, ensure_dir

def _save(fig, name: str):
    ensure_dir(EDA_DIR)
    fig.tight_layout()
    fig.savefig(EDA_DIR / name, dpi=300, bbox_inches="tight")
    plt.close(fig)

def generate_eda(df: pd.DataFrame) -> None:
    df = df.copy()
    df["Claim_Amount"] = pd.to_numeric(df["Claim_Amount"], errors="coerce")

    # --- Outcome distribution (handle numeric 0/1 or string) ---
    if "claim_outcome" in df.columns:
        co = df["claim_outcome"]
        # If numeric, map to human-readable for plotting
        if np.issubdtype(co.dropna().dtype, np.number):
            co_plot = co.map({0: "Rejected", 1: "Approved"})
        else:
            # Normalize common string variants just in case
            co_plot = co.astype(str).str.strip().str.title()
            co_plot = co_plot.replace({"1": "Approved", "0": "Rejected"})
        vc = co_plot.value_counts()
        if not vc.empty:
            fig = plt.figure()
            # Ensure order shows both bars even if one is 0
            ordered = vc.reindex(["Rejected", "Approved"]).fillna(0)
            ordered.plot(kind="bar")
            plt.title("Claim Outcome Distribution")
            plt.xlabel("Outcome"); plt.ylabel("Count")
            _save(fig, "outcome_distribution.png")

    # --- Claim amount by disease (robust) ---
    if {"Claim_Amount", "Disease_Name"}.issubset(df.columns):
        tmp = df[["Claim_Amount", "Disease_Name"]].dropna()
        plot_df = df.dropna(subset=["Claim_Amount", "Disease_Name"])
        # Guard: require at least some data
        if len(tmp) > 0 and tmp["Disease_Name"].nunique() > 0:
            # Focus on top 8 most frequent diseases for a readable plot
            top = (tmp["Disease_Name"].value_counts().head(8).index)
            tmp = tmp[tmp["Disease_Name"].isin(top)]
            if len(tmp) > 0:
                plt.figure(figsize=(12,6))
                sns.boxplot(x="Disease_Name", y="Claim_Amount", data=plot_df)
                plt.xticks(rotation=45)
                plt.title("Claim Amount by Disease")
                plt.tight_layout()
                plt.savefig("results/eda/claim_amount_by_disease.png")
                plt.show()

    # --- Age distribution ---
    if "Age" in df.columns:
        fig = plt.figure()
        df["Age"].dropna().astype(float).plot(kind="hist", bins=30)
        plt.title("Age Distribution")
        plt.xlabel("Age"); plt.ylabel("Frequency")
        _save(fig, "age_distribution.png")

    # --- Correlation heatmap (numeric only) ---
    num = df.select_dtypes(include=[np.number])
    if not num.empty:
        corr = num.corr(numeric_only=True)
        fig = plt.figure(figsize=(6, 5))
        plt.imshow(corr, interpolation="nearest")
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.title("Numeric Feature Correlation")
        _save(fig, "correlation_heatmap.png")
