import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from .config import FEATURES, LABELS, MODELS, SEED, TABLES
from .data import load_split
from .evaluation import quality_predictions


def rounded_macro_f1(estimator, features, actual):
    return f1_score(actual, quality_predictions(estimator, features), labels=LABELS,
                    average="macro", zero_division=0)


def train_models(jobs=1):
    features, actual = load_split("train")
    cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(features, actual))
    candidates = {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "baseline": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=1),
        "balanced": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                            random_state=SEED, n_jobs=1),
        "regression": RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=1),
    }
    scores = []
    for name, model in candidates.items():
        result = cross_validate(model, features, actual, cv=cv, scoring=rounded_macro_f1, n_jobs=jobs)
        scores.append({"Model": name, "CV macro F1": np.mean(result["test_score"]),
                       "CV standard deviation": np.std(result["test_score"])})
        model.fit(features, actual)
        joblib.dump(model, MODELS / f"{name}.joblib")
        print(f"Trained {name}; CV macro F1={scores[-1]['CV macro F1']:.4f}", flush=True)

    # max_features='sqrt' and 'log2' both select 3 of these 11 features.
    grid = GridSearchCV(RandomForestClassifier(random_state=SEED, n_jobs=1), {
        "n_estimators": [200, 400], "max_depth": [None, 20],
        "min_samples_leaf": [1, 2], "class_weight": [None, "balanced"],
    }, scoring=make_scorer(f1_score, labels=LABELS, average="macro", zero_division=0),
        cv=cv, n_jobs=jobs)
    grid.fit(features, actual)
    candidates["tuned"] = grid.best_estimator_
    joblib.dump(grid.best_estimator_, MODELS / "tuned.joblib")
    pd.DataFrame(grid.cv_results_).to_csv(TABLES / "tuning_cv_results.csv", index=False)
    pd.DataFrame([grid.best_params_]).to_csv(TABLES / "best_hyperparameters.csv", index=False)
    scores.append({"Model": "tuned", "CV macro F1": grid.best_score_,
                   "CV standard deviation": grid.cv_results_["std_test_score"][grid.best_index_]})
    pd.DataFrame(scores).to_csv(TABLES / "cv_comparison.csv", index=False)
    winner = max(scores, key=lambda row: row["CV macro F1"])["Model"]
    joblib.dump(candidates[winner], MODELS / "selected.joblib")
    metadata = {"selected_model": winner, "selection_metric": "5-fold CV macro F1",
                "features": FEATURES, "labels": LABELS, "random_seed": SEED,
                "sklearn_version": sklearn.__version__, "training_rows": len(features),
                "note": "CV selection scores are not unbiased generalization estimates; see held-out evaluation."}
    (MODELS / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Selected {winner} using training CV only.", flush=True)
