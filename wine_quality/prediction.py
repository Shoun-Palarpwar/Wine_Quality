import joblib
import pandas as pd

from .config import MODELS, ROOT, TABLES
from .data import validate_features
from .evaluation import quality_predictions


def predict(input_path=None, output_path=None):
    input_path = input_path or ROOT / "data/examples/wines.csv"
    output_path = output_path or TABLES / "example_predictions.csv"
    features = validate_features(pd.read_csv(input_path))
    model = joblib.load(MODELS / "selected.joblib")
    results = features.copy()
    results["Predicted Quality"] = quality_predictions(model, features)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        for index, label in enumerate(model.classes_):
            results[f"Uncalibrated P(quality={label})"] = probabilities[:, index]
    else:
        results["Raw regression score"] = model.predict(features)
    results.to_csv(output_path, index=False)
    print(f"Saved {len(results)} predictions to {output_path}")
    return results
