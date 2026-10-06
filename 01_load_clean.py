"""
Step 1 & 2: Load Wine Quality Dataset + Data Cleaning
---------------------------------------------------------
Dataset: UCI Wine Quality (Red) - Cortez et al., 2009
1,599 red "Vinho Verde" wine samples, 11 physicochemical inputs,
1 quality score (0-10, expert median rating) as target.
"""
import pandas as pd
import numpy as np

# ---------- Step 1: Load dataset ----------
df = pd.read_csv("./winequality-red.csv", sep=";")
print("Raw shape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nFirst rows:\n", df.head())

# ---------- Step 2: Data cleaning ----------
print("\n--- Data Cleaning ---")

# 2a. Missing values
missing = df.isnull().sum()
print("\nMissing values per column:\n", missing[missing > 0] if missing.sum() > 0 else "None")

# 2b. Duplicate rows
n_dupes = df.duplicated().sum()
print(f"\nDuplicate rows found: {n_dupes}")
df_clean = df.drop_duplicates().reset_index(drop=True)
print(f"Shape after removing duplicates: {df_clean.shape}")

# 2c. Data types
print("\nData types:\n", df_clean.dtypes)

# 2d. Sanity-check value ranges (chemistry values should be non-negative)
neg_counts = (df_clean.select_dtypes("number") < 0).sum()
print("\nNegative-value check (should all be 0):\n", neg_counts[neg_counts > 0] if neg_counts.sum() > 0 else "None found")

# 2e. Outlier check via IQR (report only -- we keep them, since in wine
# chemistry legitimate extreme values are common and informative for RF)
print("\nOutlier counts per feature (IQR method, 1.5x rule):")
for col in df_clean.columns[:-1]:
    q1, q3 = df_clean[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
    print(f"  {col:24s}: {n_out}")

# 2f. Quality class distribution (target)
print("\nQuality score distribution:\n", df_clean["quality"].value_counts().sort_index())

df_clean.to_csv("./winequality_clean.csv", index=False)
print(f"\nSaved cleaned dataset: {df_clean.shape[0]} rows x {df_clean.shape[1]} columns -> winequality_clean.csv")
