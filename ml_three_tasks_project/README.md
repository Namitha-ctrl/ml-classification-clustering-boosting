# Machine Learning: Classification, Clustering & Boosting

This repository demonstrates three end-to-end machine-learning workflows using the supplied datasets.

## Project structure

```text
ml_three_tasks_project/
├── data/
│   ├── Social_Network_Ads.csv
│   ├── Mall_Customers.csv
│   └── insurance_pre.csv
├── results/
│   ├── classification_metrics.csv
│   ├── clustering_metrics.csv
│   ├── boosting_metrics.csv
│   └── plots/
├── train_models.py
├── requirements.txt
└── README.md
```

## Tasks and algorithms

1. **Classification — Social Network Ads**: predicts `Purchased`; compares Logistic Regression, Random Forest, and Gradient Boosting. The `User ID` field is excluded because it is an identifier, not a useful behavioral feature.
2. **Clustering — Mall Customers**: K-Means on standardized `Annual Income (k$)` and `Spending Score (1-100)`. The script compares k=2 through k=8 using silhouette score and writes customer cluster assignments.
3. **Boosting — Insurance Charges**: predicts `charges` using Gradient Boosting Regressor; compares against a mean-prediction baseline and reports MAE, RMSE, and R².

All transformations are fitted within sklearn pipelines where applicable. A fixed random seed is used for reproducibility.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train_models.py
```

Outputs are written to `results/`. Model metrics depend on the supplied dataset and the train/test split. Clustering is unsupervised; silhouette score measures separation and cohesion, not predictive accuracy.

## GitHub

Create a repository on GitHub (or connect the GitHub integration in ChatGPT), then push this project folder. Do not commit credentials or access tokens.
