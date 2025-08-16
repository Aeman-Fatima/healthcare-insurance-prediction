import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
from utils import ensure_dir, save_fig

def evaluate_and_plot(clf, X_test, y_test, out_dir: str, model_name: str) -> dict:
    ensure_dir(out_dir)

    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)[:, 1]
    else:
        df = clf.decision_function(X_test)
        df = (df - df.min()) / (df.max() - df.min() + 1e-9)
        y_prob = df
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) == 2 else float("nan")

    # ROC
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
    ax.set_title(f"ROC – {model_name}")
    save_fig(fig, os.path.join(out_dir, f"{model_name}_roc.png"))


    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(f"Confusion Matrix – {model_name}")
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ['Rejected', 'Approved'])
    plt.yticks(ticks, ['Rejected', 'Approved'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center')
    plt.xlabel('Predicted'); plt.ylabel('True')
    save_fig(fig, os.path.join(out_dir, f"{model_name}_confusion.png"))

    return {"model": model_name, "accuracy": acc, "f1": f1, "roc_auc": roc}

def save_comparison(rows: list, out_csv: str):
    df = pd.DataFrame(rows).sort_values(by="roc_auc", ascending=False)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df
