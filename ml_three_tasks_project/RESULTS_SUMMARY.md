# Results summary

Metrics are calculated by running `python train_models.py` on the included datasets with a fixed random seed.

## Classification

| model                        |   accuracy |   precision |   recall |       f1 |   roc_auc |
|:-----------------------------|-----------:|------------:|---------:|---------:|----------:|
| Random Forest                |       0.9  |    0.842105 | 0.888889 | 0.864865 |  0.939019 |
| Gradient Boosting Classifier |       0.9  |    0.842105 | 0.888889 | 0.864865 |  0.944444 |
| Logistic Regression          |       0.81 |    0.814815 | 0.611111 | 0.698413 |  0.91059  |

## Clustering

|   k |   inertia |   silhouette_score |
|----:|----------:|-------------------:|
|   2 |  269.691  |           0.321271 |
|   3 |  157.704  |           0.466585 |
|   4 |  108.921  |           0.493907 |
|   5 |   65.5684 |           0.554657 |
|   6 |   55.0573 |           0.53988  |
|   7 |   44.8648 |           0.528149 |
|   8 |   37.1481 |           0.456721 |

## Insurance boosting/regression

| model                       |     MAE |    RMSE |           R2 |
|:----------------------------|--------:|--------:|-------------:|
| Gradient Boosting Regressor | 2429.94 |  4316.2 |  0.880002    |
| Mean baseline               | 9593.34 | 12465.6 | -0.000919249 |

See `results/plots/` for figures and `results/summary.json` for a machine-readable summary.
