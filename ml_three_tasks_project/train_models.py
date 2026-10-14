from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, silhouette_score, mean_absolute_error, mean_squared_error, r2_score
)

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"
RESULTS.mkdir(exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

def save_df(df, path):
    df.to_csv(path, index=False)

# ---------------- Classification: Social Network Ads ----------------
ads = pd.read_csv(DATA / "Social_Network_Ads.csv")
X = ads.drop(columns=["Purchased", "User ID"], errors="ignore")
y = ads["Purchased"]
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=np.number).columns.tolist()
preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced"),
    "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=RANDOM_STATE),
}
class_rows, predictions = [], {}
for name, model in classifiers.items():
    pipe = Pipeline([("preprocess", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
    row = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba) if proba is not None else np.nan,
    }
    class_rows.append(row)
    predictions[name] = (pipe, pred)
class_df = pd.DataFrame(class_rows).sort_values("f1", ascending=False)
save_df(class_df, RESULTS / "classification_metrics.csv")
best_name = class_df.iloc[0]["model"]
best_pipe, best_pred = predictions[best_name]
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No purchase", "Purchase"], yticklabels=["No purchase", "Purchase"])
plt.title(f"Classification Confusion Matrix — {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(PLOTS / "classification_confusion_matrix.png", dpi=160)
plt.close()

# ---------------- Clustering: Mall Customers ----------------
mall = pd.read_csv(DATA / "Mall_Customers.csv")
cluster_features = ["Annual Income (k$)", "Spending Score (1-100)"]
cluster_X = mall[cluster_features].copy()
scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(cluster_X)
cluster_rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    labels = km.fit_predict(cluster_scaled)
    cluster_rows.append({"k": k, "inertia": km.inertia_, "silhouette_score": silhouette_score(cluster_scaled, labels)})
cluster_metrics = pd.DataFrame(cluster_rows)
save_df(cluster_metrics, RESULTS / "clustering_metrics.csv")
best_k = int(cluster_metrics.loc[cluster_metrics["silhouette_score"].idxmax(), "k"])
best_km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
mall["Cluster"] = best_km.fit_predict(cluster_scaled)
save_df(mall, RESULTS / "mall_customer_clusters.csv")
plt.figure(figsize=(7, 5))
sns.scatterplot(data=mall, x=cluster_features[0], y=cluster_features[1], hue="Cluster", palette="tab10", s=65)
plt.title(f"Mall Customer Segments (K-Means, k={best_k})")
plt.tight_layout()
plt.savefig(PLOTS / "mall_customer_clusters.png", dpi=160)
plt.close()
plt.figure(figsize=(6, 4))
plt.plot(cluster_metrics["k"], cluster_metrics["inertia"], marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("K-Means Elbow Curve")
plt.tight_layout()
plt.savefig(PLOTS / "kmeans_elbow_curve.png", dpi=160)
plt.close()

# ---------------- Boosting / Regression: Insurance Charges ----------------
insurance = pd.read_csv(DATA / "insurance_pre.csv")
Xr = insurance.drop(columns=["charges"])
yr = insurance["charges"]
cat_r = Xr.select_dtypes(include=["object", "category"]).columns.tolist()
num_r = Xr.select_dtypes(include=np.number).columns.tolist()
reg_preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_r),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_r),
])
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr, yr, test_size=0.20, random_state=RANDOM_STATE
)
reg_models = {
    "Mean baseline": DummyRegressor(strategy="mean"),
    "Gradient Boosting Regressor": GradientBoostingRegressor(
        n_estimators=250, learning_rate=0.04, max_depth=2, loss="squared_error", random_state=RANDOM_STATE
    ),
}
reg_rows, reg_predictions = [], {}
for name, model in reg_models.items():
    pipe = Pipeline([("preprocess", reg_preprocess), ("model", model)])
    pipe.fit(Xr_train, yr_train)
    pred = pipe.predict(Xr_test)
    reg_predictions[name] = pred
    reg_rows.append({
        "model": name,
        "MAE": mean_absolute_error(yr_test, pred),
        "RMSE": float(np.sqrt(mean_squared_error(yr_test, pred))),
        "R2": r2_score(yr_test, pred),
    })
reg_df = pd.DataFrame(reg_rows).sort_values("RMSE")
save_df(reg_df, RESULTS / "boosting_metrics.csv")
gb_pred = reg_predictions["Gradient Boosting Regressor"]
plt.figure(figsize=(6, 5))
plt.scatter(yr_test, gb_pred, alpha=0.65)
limits = [min(float(yr_test.min()), float(gb_pred.min())), max(float(yr_test.max()), float(gb_pred.max()))]
plt.plot(limits, limits, "r--", linewidth=1)
plt.xlabel("Actual insurance charges")
plt.ylabel("Predicted insurance charges")
plt.title("Gradient Boosting Regressor: Actual vs Predicted")
plt.tight_layout()
plt.savefig(PLOTS / "insurance_actual_vs_predicted.png", dpi=160)
plt.close()

# Short machine-readable summary.
summary = {
    "classification_best_by_f1": class_df.iloc[0].to_dict(),
    "clustering_best_k_by_silhouette": best_k,
    "clustering_best_silhouette": float(cluster_metrics.loc[cluster_metrics["k"] == best_k, "silhouette_score"].iloc[0]),
    "boosting_gradient_boosting": reg_df[reg_df["model"] == "Gradient Boosting Regressor"].iloc[0].to_dict(),
}
(RESULTS / "summary.json").write_text(__import__("json").dumps(summary, indent=2), encoding="utf-8")
print("Classification metrics:")
print(class_df.to_string(index=False))
print("\nClustering metrics:")
print(cluster_metrics.to_string(index=False))
print(f"\nChosen k by silhouette score: {best_k}")
print("\nInsurance regression metrics:")
print(reg_df.to_string(index=False))
print("\nSummary saved to results/summary.json")
