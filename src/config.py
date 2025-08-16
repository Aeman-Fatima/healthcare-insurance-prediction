
from dataclasses import dataclass

@dataclass
class Paths:
    customers_path: str
    claims_path: str
    marketplace_path: str
    complaints_path: str
    output_dir: str = "results"

@dataclass
class SplitConfig:
    test_size: float = 0.30
    random_state: int = 42

@dataclass
class Columns:
    # Minimal schema expected after merge
    target: str = "claim_outcome"  # values: Approved/Rejected (convert to 1/0)
    numeric: tuple = ("Age", "Claim_Amount", "Loyalty_Points", "Previous_Claims")
    categorical: tuple = ("Gender", "Policy_Status", "Disease_Name", "Region")

@dataclass
class ModelNames:
    lr: str = "LogisticRegression"
    rf: str = "RandomForest"
    xgb: str = "XGBoost"
