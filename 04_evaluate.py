"""
Step 6: Evaluate accuracy and metrics (in depth)
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_recall_fscore_support
)

sns.set_style("whitegrid")
plt.rcParams.update({"figure.facecolor": "white", "font.size": 10})

X_train = pd.read_csv("./X_train.csv")
X_test = pd.read_csv("./X_test.csv")
y_train = pd.read_csv("./y_train.csv").squeeze()
y_test = pd.read_csv("./y_test.csv").squeeze()
model = joblib.load("./rf_baseline_model.joblib")

y_pred = model.predict(X_test)
labels = sorted(y_test.unique())

# ---------- Confusion matrix ----------
cm = confusion_matrix(y_test, y_pred, labels=labels)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_xlabel("Predicted Quality")
ax.set_ylabel("Actual Quality")
ax.set_title(f"Confusion Matrix — Random Forest (Accuracy={accuracy_score(y_test, y_pred):.3f})",
             fontweight="bold")
plt.tight_layout()
plt.savefig("./eval_confusion_matrix.png", dpi=150)
plt.close()

# ---------- Precision/Recall/F1 per class ----------
prec, rec, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=labels, zero_division=0)
metrics_df = pd.DataFrame({
    "Quality": labels, "Precision": prec, "Recall": rec, "F1-score": f1, "Support": support
})
metrics_df.to_csv("./eval_per_class_metrics.csv", index=False)
print(metrics_df.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(labels))
width = 0.25
ax.bar(x - width, prec, width, label="Precision", color="#722F37")
ax.bar(x, rec, width, label="Recall", color="#C9A66B")
ax.bar(x + width, f1, width, label="F1-score", color="#4C72B0")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel("Quality Score")
ax.set_ylabel("Score")
ax.set_title("Precision / Recall / F1 by Quality Class", fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("./eval_per_class_metrics.png", dpi=150)
plt.close()

# ---------- Feature importance ----------
importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.barh(importances.index, importances.values, color="#722F37")
ax.set_xlabel("Importance")
ax.set_title("Random Forest Feature Importance", fontweight="bold")
ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
plt.tight_layout()
plt.savefig("./eval_feature_importance.png", dpi=150)
plt.close()

print("\nTop 5 most important features:")
print(importances.sort_values(ascending=False).head(5).round(4).to_string())

print("\nSaved: eval_confusion_matrix.png, eval_per_class_metrics.png/.csv, eval_feature_importance.png")
