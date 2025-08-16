from typing import Dict
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from preprocessing import make_preprocessor
from config import Columns, ModelNames

def build_models(preprocessor, cols: Columns) -> Dict[str, Pipeline]:
    models = {}

    lr = LogisticRegression(max_iter=1000, n_jobs=None)  # simple baseline
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_split=2, random_state=42, n_jobs=-1
    )
    xgb = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=42, eval_metric="logloss"
    )

    models[ModelNames.lr] = Pipeline(steps=[("preprocess", preprocessor), ("model", lr)])
    models[ModelNames.rf] = Pipeline(steps=[("preprocess", preprocessor), ("model", rf)])
    models[ModelNames.xgb] = Pipeline(steps=[("preprocess", preprocessor), ("model", xgb)])

    return models
