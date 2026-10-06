"""
Steps 4-5: Train/test split -> Apply Random Forest classification model
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)

df = pd.read_csv("./winequality_clean.csv")

X = df.drop(columns="quality")
y = df["quality"]

# ---------- Step 4: Train/test split ----------
# stratify to preserve the (imbalanced) quality class proportions in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
print("\nTrain class distribution:\n", y_train.value_counts().sort_index())
print("\nTest class distribution:\n", y_test.value_counts().sort_index())

# ---------- Step 5: Apply classification model (Random Forest) ----------
rf_baseline = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_baseline.fit(X_train, y_train)

y_pred_base = rf_baseline.predict(X_test)
baseline_acc = accuracy_score(y_test, y_pred_base)
baseline_f1 = f1_score(y_test, y_pred_base, average="weighted")

print(f"\n--- Baseline Random Forest (default-ish params, n_estimators=200) ---")
print(f"Accuracy: {baseline_acc:.4f}")
print(f"Weighted F1: {baseline_f1:.4f}")
print("\nClassification report:\n", classification_report(y_test, y_pred_base, zero_division=0))

# Save splits for reuse in later steps
X_train.to_csv("./X_train.csv", index=False)
X_test.to_csv("./X_test.csv", index=False)
y_train.to_csv("./y_train.csv", index=False)
y_test.to_csv("./y_test.csv", index=False)
joblib.dump(rf_baseline, "./rf_baseline_model.joblib")

with open("./baseline_metrics.txt", "w") as f:
    f.write(f"Baseline Random Forest Accuracy: {baseline_acc:.4f}\n")
    f.write(f"Baseline Random Forest Weighted F1: {baseline_f1:.4f}\n")

print("\nSaved: X_train/X_test/y_train/y_test.csv, rf_baseline_model.joblib, baseline_metrics.txt")
