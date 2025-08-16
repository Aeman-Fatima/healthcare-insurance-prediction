import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, RocCurveDisplay
)
from utils import ensure_dir, save_fig

def evaluate_and_plot(clf, X_test, y_test, out_dir: str, model_name: str) -> dict:
    ensure_dir(out_dir)

    y_prob = None
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)[:, 1]
    else:
        # fallback for models without predict_proba
        y_prob = clf.decision_function(X_test)
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-9)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) == 2 else np.nan

    # ROC curve
    fig = plt.figure()
    RocCurveDisplay.from_predictions(y_test, y_prob)
    save_fig(fig, os.path.join(out_dir, f"{model_name}_roc.png"))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(f"Confusion Matrix – {model_name}")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Rejected', 'Approved'])
    plt.yticks(tick_marks, ['Rejected', 'Approved'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center')
    save_fig(fig, os.path.join(out_dir, f"{model_name}_confusion.png"))

    return {"model": model_name, "accuracy": acc, "f1": f1, "roc_auc": roc}

def save_comparison(rows: list, out_csv: str):
    df = pd.DataFrame(rows)
    df = df.sort_values(by="roc_auc", ascending=False)
    df.to_csv(out_csv, index=False)
    return df
