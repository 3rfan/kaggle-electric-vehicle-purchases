# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path

import pandas as pd

# Locate the repository from either the root or the notebooks directory.
PROJECT_ROOT = next(
    (path for path in (Path.cwd(), *Path.cwd().parents)
     if (path / "data" / "raw" / "train.csv").is_file()),
    None,
)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Run from the repository root or notebooks directory.")

TARGET = "Will_Buy_EV"
train_df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "train.csv")

print(f"Dataset shape (including ID and target): {train_df.shape}")
target_balance = pd.DataFrame({
    "count": train_df[TARGET].value_counts(dropna=False),
    "percent": train_df[TARGET].value_counts(dropna=False, normalize=True) * 100,
})
print("\nTarget balance:")
print(target_balance.to_string())
print("\nMissing values per column:")
print(train_df.isna().sum().to_string())

# %%
print("First five rows:")
print(train_df.head().to_string(index=False))
