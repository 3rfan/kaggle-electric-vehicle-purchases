# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

train_path = "../input/competitions/playground-series-s6e9/train.csv"
test_path = "../input/competitions/playground-series-s6e9/test.csv"

train_df = pd.read_csv(train_path, index_col="id")
test_df = pd.read_csv(test_path, index_col="id")
sample_submission_path = (
    "../input/competitions/playground-series-s6e9/sample_submission.csv"
)

# %%
print(f"train shape: {train_df.shape}")
print(f"test shape: {test_df.shape}")

print()
print("General info:")
train_df.info()
print()
print("Num of distinct values per column")
print(train_df.nunique())
print()
print("Num of missing values per column")
print(train_df.isna().sum())
print()
print("Detailed descrption of numeric cols")
train_df.describe()

# %%
train_df.head()

# %%
numeric_features = train_features.select_dtypes(
    include="number"
).columns

numeric_comparison = pd.DataFrame({
    "train_mean": train_features[numeric_features].mean(),
    "test_mean": test_df[numeric_features].mean(),
    "train_median": train_features[numeric_features].median(),
    "test_median": test_df[numeric_features].median(),
    "train_std": train_features[numeric_features].std(),
    "test_std": test_df[numeric_features].std()
})

numeric_comparison

# %%
train_sample = train_features.sample(
    100_000,
    random_state=42
)

test_sample = test_df.sample(
    100_000,
    random_state=42
)

fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for col, ax in zip(numeric_features, axes.flat):
    sns.histplot(
        train_sample[col],
        bins=40,
        stat="density",
        element="step",
        fill=False,
        label="Train",
        ax=ax
    )

    sns.histplot(
        test_sample[col],
        bins=40,
        stat="density",
        element="step",
        fill=False,
        label="Test",
        ax=ax
    )

    ax.set_title(col)
    ax.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# Baseline catboost model

# %%
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

TARGET = "Will_Buy_EV"
SEED = 42

X = train_df.drop(columns=TARGET)
y = train_df[TARGET].map({"No": 0, "Yes": 1})
X_test = test_df.loc[:, X.columns]

cat_features = X.select_dtypes(include=["object", "string"]).columns.tolist()

X_fit, X_valid, y_fit, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=SEED,
)

print(f"Training rows: {len(X_fit):,}")
print(f"Validation rows: {len(X_valid):,}")
print(f"Buyer proportion: {y.mean():.2%}")

# %%
params = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    depth=6,
    learning_rate=0.08,
    random_seed=SEED,
    task_type="CPU",
    thread_count=4,
    counter_calc_method="SkipTest",
    allow_writing_files=False,
)

model = CatBoostClassifier(**params, iterations=1600)

model.fit(
    X_fit,
    y_fit,
    cat_features=cat_features,
    eval_set=(X_valid, y_valid),
    early_stopping_rounds=120,
    use_best_model=True,
    verbose=100,
)

valid_pred = model.predict_proba(X_valid)[:, 1]
valid_auc = roc_auc_score(y_valid, valid_pred)
best_iterations = model.tree_count_

print(f"\nHoldout ROC-AUC: {valid_auc:.6f}")
print(f"Selected trees: {best_iterations}")

# Save predictions for comparisons with future experiments.
pd.DataFrame({
    "id": X_valid.index,
    "target": y_valid.to_numpy(),
    "prediction": valid_pred,
}).to_csv("/kaggle/working/holdout_predictions.csv", index=False)

# %%
final_model = CatBoostClassifier(
    **params,
    iterations=best_iterations,
)

final_model.fit(
    X,
    y,
    cat_features=cat_features,
    verbose=100,
)

test_pred = final_model.predict_proba(X_test)[:, 1]

submission = pd.read_csv(sample_submission_path, index_col="id")
assert submission.columns.tolist() == [TARGET]
assert submission.index.equals(X_test.index)
assert submission.index.is_unique
assert np.isfinite(test_pred).all()
assert ((test_pred >= 0) & (test_pred <= 1)).all()

submission[TARGET] = test_pred
submission.to_csv("/kaggle/working/submission.csv", index_label="id")

print("Saved submission.csv:", submission.shape)
display(submission.head())
