# healthcare-insurance-prediction

This repository contains the code to reproduce the data preprocessing, modelling, and evaluation
for the **health insurance claims approval classification** task.

## Datasets (not included)
Download these publicly available datasets and place them under `data/raw/`:
- UCI COIL 2000 (customer features)
- HHS Health Insurance Marketplace (plan features)
- CFPB Consumer Complaints (complaints text/labels)
- Simulated Renewals/Loyalty (we provide a small generator script or CSV schema)

> We do **not** commit raw data. Follow instructions below to set paths.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/main.py \
  --customers_path data/raw/coil2000_customers.csv \
  --claims_path data/raw/claims.csv \
  --marketplace_path data/raw/hhs_marketplace.csv \
  --complaints_path data/raw/cfpb_complaints.csv \
  --output_dir results
