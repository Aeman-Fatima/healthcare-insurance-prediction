from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT_DIR / "results"
FIGS_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
EDA_DIR = RESULTS_DIR / "eda"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
