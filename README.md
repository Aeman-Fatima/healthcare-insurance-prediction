# Health Insurance Big Data Pipeline – Predictive Modeling

## Overview
This project explores predictive modeling in the health insurance domain, focusing on customer behavior, policy renewal, and market trends. Using a mix of public and simulated datasets, the pipeline demonstrates how machine learning can be applied to:

- Predict customer churn and policy renewals
- Analyze complaint data to identify key service issues
- Understand policy pricing and adoption trends
- Generate performance metrics and visualizations for evaluation

The work follows the scope of the COMP SCI 7319OL project (Milestone 1A–D), combining both research and practical implementation.

## Datasets
The project uses four datasets:

1) Health Insurance Customer Dataset (COIL 2000 – UCI Repository)
  - COIL 2000 Dataset
  - Contains demographic and insurance-related variables for customer churn modeling.
2) Health Insurance Marketplace Dataset (Kaggle – HHS, 2023)
  - Kaggle: Health Insurance Marketplace Data
  - Provides insurance plan and pricing information for marketplace analysis.
3) Health Insurance Reviews and Complaints Dataset (CFPB, 2024)
  - Consumer Financial Protection Bureau Complaints
  - Includes structured complaints data relevant to service quality and regulatory insights.
4) Policy Renewal and Loyalty Dataset (Simulated)
  - Since real-world renewal/loyalty datasets are not publicly available due to privacy concerns, a simulated dataset was created.
  - It mirrors realistic patterns of customer loyalty, retention, and policy renewal.
  - This ensures the pipeline covers end-to-end use cases while maintaining data integrity.

## Methods
- Data preprocessing and cleaning (handling missing values, encoding, normalization)
- Feature engineering for customer behavior and insurance attributes
- Model development: Logistic Regression, Random Forests, Gradient Boosting, XGBoost
- Evaluation metrics: Accuracy, Precision, Recall, ROC-AUC
- Visualization: Confusion matrices, ROC curves, feature importance plots

## Results

The pipeline outputs include:
- Predictive performance metrics (classification reports)
- ROC-AUC curves for model comparison
- Insights into policyholder behavior and complaint trends
- Demonstration of how simulated loyalty data complements public datasets

Notes
- This project is academic in nature and not intended for production deployment.
- All datasets are publicly available except the simulated loyalty dataset.
- The simulated dataset was designed to resemble realistic customer features for the purposes of model evaluation and pipeline completeness.


# How to Run

1) Set up the environment
```
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

Tip: Pin versions in requirements.txt to avoid TA env issues (e.g., scikit-learn==1.4.2, xgboost==2.0.3, pandas==2.2.2, matplotlib==3.8.4, numpy==1.26.4).

2) Generate synthetic datasets (reproducible)
```
# Customers (demographics/policy)
python src/generate_customers.py \
  --n_customers 8000 \
  --seed 123 \
  --out data/raw/coil2000_customers.csv

# Claims (with disease, amount, outcome)
python src/generate_claims.py \
  --n_claims 10000 \
  --n_customers 8000 \
  --seed 456 \
  --out data/raw/claims.csv

# Renewals/Loyalty (simulated)
python src/generate_loyalty.py \
  --n_customers 8000 \
  --seed 42 \
  --out data/raw/renewals_loyalty.csv
```

We do not commit these CSVs. They’re generated locally to keep the repo lightweight and academically transparent.

3) (Optional) Place public datasets

If you want to run with public data locally, download and place the CSVs here (or update paths accordingly):

```
data/raw/
  coil2000_customers.csv           # or real COIL 2000 export
  claims.csv                       # your own claims file, if available
  renewals_loyalty.csv             # from the generator script above
  cfpb_complaints_sample.csv       # optional small sample for testing
  hhs_marketplace.csv              # optional
```

The pipeline does not require marketplace/complaints to run. They’re documented as optional integrations.

4) Run the pipeline
```
python src/main.py \
  --customers_path data/raw/coil2000_customers.csv \
  --claims_path data/raw/claims.csv \
  --renewals_path data/raw/renewals_loyalty.csv \
  --output_dir results
```

Optional args if you have the files:
```
  --marketplace_path data/raw/hhs_marketplace.csv \
  --complaints_path data/raw/cfpb_complaints_sample.csv
```

5) Outputs you should see
```
results/
  figures/
    LogisticRegression_roc.png
    LogisticRegression_confusion.png
    RandomForest_roc.png
    RandomForest_confusion.png
    XGBoost_roc.png
    XGBoost_confusion.png
  tables/
    model_comparison.csv   # Accuracy, F1, ROC-AUC (sorted)
```

7) Reproducibility notes

The policy renewals & loyalty dataset is simulated via src/generate_loyalty.py, as real-world equivalents are not publicly available.
Public datasets (UCI COIL 2000, Kaggle HHS Marketplace, CFPB Complaints) are linked in this README and can be used locally. Due to size, they are not shipped in the repo.
Results may differ slightly from prior milestones because this repository focuses on transparent, reproducible generation and processing while preserving the methodology.