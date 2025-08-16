from dataclasses import dataclass

@dataclass
class Paths:
    customers_path: str
    claims_path: str
    marketplace_path: str | None = None
    complaints_path: str | None = None
    renewals_path: str | None = "data/raw/renewals_loyalty.csv"
    output_dir: str = "results"

@dataclass
class SplitConfig:
    test_size: float = 0.30
    random_state: int = 42

@dataclass
class Columns:
    target: str = "claim_outcome"
    numeric: tuple = ("Age", "Claim_Amount", "Loyalty_Points", "Previous_Claims", "Tenure_Years")
    categorical: tuple = ("Gender", "Policy_Status", "Disease_Name", "Region", "Renewal_Status")
