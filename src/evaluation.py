import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
from src.paths import FIGS_DIR, TABLES_DIR, ensure_dir

def evaluate_and_plot(clf, X_test, y_test, model_name: str) -> dict:
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)[:, 1]
    else:
        df = clf.decision_function(X_test)
        y_prob = (df - df.min()) / (df.max() - df.min() + 1e-9)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) == 2 else float("nan")

    # ROC
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
    ax.set_title(f"ROC – {model_name}")
    ensure_dir(FIGS_DIR); fig.savefig(FIGS_DIR / f"{model_name}_roc.png", dpi=300); plt.close(fig)

    # Confusion
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(f"Confusion Matrix – {model_name}")
    fig.colorbar(im, ax=ax)
    ticks = np.arange(2)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(['Rejected','Approved']); ax.set_yticklabels(['Rejected','Approved'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    fig.savefig(FIGS_DIR / f"{model_name}_confusion.png", dpi=300); plt.close(fig)

    return {"model": model_name, "accuracy": acc, "f1": f1, "roc_auc": roc}

def save_comparison(rows: list, out_csv: Path = TABLES_DIR / "model_comparison.csv"):
    ensure_dir(TABLES_DIR)
    pd.DataFrame(rows).sort_values("roc_auc", ascending=False).to_csv(out_csv, index=False)
    return out_csv
