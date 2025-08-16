### Aeman-Fatima (a1946845)

# Healthcare Insurance Big Data Pipeline

## Project Overview

This project builds a big data pipeline for analyzing health insurance datasets. The pipeline simulates customer, claims, and loyalty data while optionally integrating real-world complaint aggregates from the CFPB (Consumer Financial Protection Bureau) dataset.

The pipeline demonstrates:

- Data generation and ingestion
- Preprocessing and feature engineering
- Model training & evaluation (XGBoost, Random Forest, Logistic Regression, etc.)
- Output of metrics, visualizations, and comparisons

## Datasets
1. Simulated Datasets (Core)

Since many large datasets cannot be uploaded to this repository, simulated data is generated to ensure reproducibility:

- Customer Dataset → generated from UCI COIL 2000 structure (customer demographics & attributes)
- Claims Dataset → synthetic claims history with policy codes, claim status, claim type
- Renewals & Loyalty Dataset → simulated renewal/loyalty data for policyholders

These generators produce data that mirrors real-world structures and are the default for running the pipeline.

2. Optional Real-World Aggregate (Enhancement)

- Complaints Dataset (CFPB, 2024) – Public consumer complaints about financial/insurance companies.
  - Due to its large size, the raw file is not stored here.
  - Instead, a lightweight pre-aggregated file (complaints_agg_by_state.csv) can be included under data/processed/ for reproducibility.
  - If not present, the pipeline still runs using only simulated datasets.

## How to Run
1. Setup Environment
```
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
2. Generate Data
```
# Generate synthetic datasets
python src/generate_customers.py --out data/raw/coil2000_customers.csv
python src/generate_claims.py --out data/raw/claims.csv
python src/generate_loyalty.py --out data/raw/renewals_loyalty.csv
```

(Optional: Preprocess real complaints dataset if available)

```
python src/ingest_complaints.py --in data/raw/complaints.csv \
                                --out data/processed/complaints_agg_by_state.csv
```
3. Run Pipeline
```
python src/main.py \
  --customers_path data/raw/coil2000_customers.csv \
  --claims_path data/raw/claims.csv \
  --renewals_path data/raw/renewals_loyalty.csv \
  --output_dir results
```

## Repository Structure
```bash
.
├── src/                # Source code (generation, ingestion, preprocessing, modelling, evaluation)
├── data/
│   ├── raw/            # Input datasets (simulated or ingested)
│   └── processed/      # Aggregated/cleaned datasets
├── results/
│   ├── figures/        # ROC curves, model plots
│   └── tables/         # Comparison tables (CSV)
├── README.md
├── requirements.txt
└── .gitignore
```
## Outputs

- ROC curves & evaluation plots → results/figures/
- Model performance comparison table → results/tables/model_comparison.csv

## Notes & Limitations

- Large raw datasets are not stored due to size limits. Simulated equivalents are generated to reproduce the workflow.
- Complaints data is optional. If not available, the pipeline falls back on purely simulated inputs.
- Results may vary slightly depending on seed/random splits.
- This project is for academic purposes only and does not represent production insurance analytics.

