import os
import json
import matplotlib.pyplot as plt

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def save_fig(fig, path: str, tight=True):
    ensure_dir(os.path.dirname(path))
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
