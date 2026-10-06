"""
Step 7: Tune hyperparameters (GridSearchCV)
"""
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report

X_train = pd.read_csv("./X_train.csv")
X_test = pd.read_csv("./X_test.csv")
y_train = pd.read_csv("./y_train.csv").squeeze()
y_test = pd.read_csv("./y_test.csv").squeeze()

param_grid = {
    "n_estimators": [100, 200, 400],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Full grid = 3*4*3*3*2 = 216 combos x 5 folds = 1080 fits -> use a
# RandomizedSearch-style budget via GridSearchCV on a slightly trimmed
# grid to keep runtime reasonable while still being a genuine grid search.
param_grid_trimmed = {
    "n_estimators": [200, 400],
    "max_depth": [None, 20, 30],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"],
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid_trimmed,
    cv=cv,
    scoring="f1_weighted",
    n_jobs=-1,
    verbose=1,
)

print("Running GridSearchCV ... (this covers "
      f"{np.prod([len(v) for v in param_grid_trimmed.values()])} combinations x 5-fold CV)")
t0 = time.time()
grid_search.fit(X_train, y_train)
elapsed = time.time() - t0
print(f"Grid search completed in {elapsed:.1f}s")

print("\nBest hyperparameters:")
for k, v in grid_search.best_params_.items():
    print(f"  {k}: {v}")
print(f"\nBest CV weighted-F1: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)
tuned_acc = accuracy_score(y_test, y_pred_tuned)
tuned_f1 = f1_score(y_test, y_pred_tuned, average="weighted")

print(f"\n--- Tuned Random Forest — Test Set Performance ---")
print(f"Accuracy: {tuned_acc:.4f}")
print(f"Weighted F1: {tuned_f1:.4f}")
print("\nClassification report:\n", classification_report(y_test, y_pred_tuned, zero_division=0))

# Compare to baseline
with open("./baseline_metrics.txt") as f:
    print(f.read())

joblib.dump(best_model, "./rf_tuned_model.joblib")

comparison = pd.DataFrame({
    "Model": ["Baseline RF (n_estimators=200, defaults)", "Tuned RF (GridSearchCV)"],
    "Accuracy": [0.6140, tuned_acc],
    "Weighted F1": [0.5949, tuned_f1],
})
comparison.to_csv("./model_comparison.csv", index=False)
print("\nModel comparison:\n", comparison.round(4).to_string(index=False))

pd.DataFrame([grid_search.best_params_]).to_csv("./best_hyperparameters.csv", index=False)
print("\nSaved: rf_tuned_model.joblib, model_comparison.csv, best_hyperparameters.csv")
