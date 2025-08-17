from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def build_models(preprocessor):
    return {
        "LogisticRegression": Pipeline([("preprocess", preprocessor), ("model", LogisticRegression(max_iter=1000))]),
        "RandomForest": Pipeline([("preprocess", preprocessor), ("model", RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1))]),
        "XGBoost": Pipeline([("preprocess", preprocessor), ("model", XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=42, eval_metric="logloss"))]),
    }
