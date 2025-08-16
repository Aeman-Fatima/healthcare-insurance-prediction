# Health Insurance Big Data Pipeline

This repository contains the implementation for COMP SCI 7319OL — Milestone 1A, focusing on building a big data pipeline for health insurance analytics. The project demonstrates dataset integration, preprocessing, and predictive modeling for various use cases in the insurance domain.


### Datasets Used:

The project draws from a mix of publicly available datasets and a simulated dataset (for renewals/loyalty analysis where no open data exists):

- Health Insurance Customer Dataset (COIL 2000) — UCI Machine Learning Repository
- Health Insurance Marketplace Dataset — Kaggle (HHS, 2023)
- Consumer Complaints Dataset — Consumer Financial Protection Bureau (CFPB, 2024)

Policy Renewal & Loyalty Dataset — Simulated (structured using realistic insurance attributes such as age, policy duration, and claim history).

Note: Due to file size constraints, large datasets are not included in this repo. Public datasets can be accessed directly via the provided links. For demonstration, small subsets are used locally.

### Project Structure:
.
├── data/                 # Sample subsets of datasets (where applicable)
├── results/              # Model outputs, metrics, and ROC plots
├── src/                  # Source code for preprocessing & modeling
├── notebooks/            # Jupyter notebooks for exploratory analysis
└── README.md             # Project documentation

### Pipeline Overview:

The project pipeline covers:

- Data Collection: Integrating datasets from UCI, Kaggle, CFPB, and simulated data.
- Preprocessing: Cleaning, feature engineering, handling missing values.
- Modeling: Applying ML models (e.g., Logistic Regression, Random Forest, XGBoost).
- Evaluation: Accuracy, precision, recall, F1-score, ROC curves.


### Notes on Data:

- *Simulated Dataset*: Only the policy renewals & loyalty dataset is simulated. This is clearly separated in both code and documentation.
- *Reproducibility*: The code can run on subsets of the public datasets included here. For full-scale results, download the datasets directly from their sources.
- *Transparency*: No proprietary or private data is used — only open-access or self-simulated datasets.

### Setup:

Clone the repository:

git clone https://github.com/your-repo/health-insurance-big-data.git
cd health-insurance-big-data

pip install -r requirements.txt

python src/main.py

### Disclaimer

This project is for academic purposes only. The simulated dataset was created solely to demonstrate methodology where real-world open datasets are unavailable.

