import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score, mean_absolute_error)

from .config import FIGURES, LABELS, MODELS, TABLES
from .data import load_split


def quality_predictions(model, features):
    predictions = model.predict(features)
    # Regressors preserve ordering during fitting; round only for classification metrics.
    return np.clip(np.rint(predictions), min(LABELS), max(LABELS)).astype(int)


def metrics(actual, predicted):
    return {
        "Accuracy": accuracy_score(actual, predicted),
        "Weighted F1": f1_score(actual, predicted, labels=LABELS, average="weighted", zero_division=0),
        "Macro F1": f1_score(actual, predicted, labels=LABELS, average="macro", zero_division=0),
        "Balanced accuracy": balanced_accuracy_score(actual, predicted),
        "MAE": mean_absolute_error(actual, predicted),
        "Within one point": float(np.mean(np.abs(np.asarray(actual) - predicted) <= 1)),
        "Quadratic weighted kappa": cohen_kappa_score(actual, predicted, labels=LABELS, weights="quadratic"),
    }


def evaluate_models():
    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    features, actual = load_split("test")
    rows = []
    for name in ["dummy", "baseline", "balanced", "regression", "tuned"]:
        model = joblib.load(MODELS / f"{name}.joblib")
        predicted = quality_predictions(model, features)
        rows.append({"Model": name, **metrics(actual, predicted)})
        report = classification_report(actual, predicted, labels=LABELS, output_dict=True, zero_division=0)
        pd.DataFrame(report).T.to_csv(TABLES / f"{name}_per_class.csv")
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(confusion_matrix(actual, predicted, labels=LABELS), annot=True,
                    fmt="d", xticklabels=LABELS, yticklabels=LABELS, cmap="Reds", ax=ax)
        ax.set(xlabel="Predicted quality", ylabel="Actual quality", title=f"{name}: confusion matrix")
        fig.tight_layout()
        fig.savefig(FIGURES / f"{name}_confusion_matrix.png", dpi=150)
        plt.close(fig)
        if hasattr(model, "feature_importances_"):
            importance = pd.Series(model.feature_importances_, index=features.columns).sort_values()
            importance.to_csv(TABLES / f"{name}_feature_importance.csv", header=["importance"])
            fig, ax = plt.subplots(figsize=(8, 6))
            importance.plot.barh(ax=ax, title=f"{name}: impurity-based importance")
            fig.tight_layout()
            fig.savefig(FIGURES / f"{name}_feature_importance.png", dpi=150)
            plt.close(fig)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(TABLES / "model_comparison.csv", index=False)
    (TABLES / "evaluation_summary.md").write_text(
        "# Held-out test results\n\n```text\n" + comparison.round(4).to_string(index=False)
        + "\n```\n\nModel selection uses training cross-validation only. "
        "These test results describe one fixed split; they do not establish a performance ceiling.\n")
    print(comparison.round(4).to_string(index=False))
    return comparison
