import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CLEAN, FEATURES, LABELS, RAW, SEED, SPLITS, TABLES, prepare_directories


def validate_features(frame):
    missing = set(FEATURES) - set(frame.columns)
    extra = set(frame.columns) - set(FEATURES)
    if missing or extra:
        raise ValueError(f"Feature schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    ordered = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="raise")
    if ordered.empty or not np.isfinite(ordered.to_numpy()).all():
        raise ValueError("Features must contain nonempty, finite numeric values.")
    if (ordered < 0).any().any():
        raise ValueError("Chemistry measurements must be nonnegative.")
    return ordered


def clean_data():
    prepare_directories()
    raw = pd.read_csv(RAW, sep=";")
    validate_features(raw.drop(columns="quality"))
    if not raw.quality.isin(LABELS).all():
        raise ValueError("Training quality labels must be integers from 3 to 8.")
    clean = raw.drop_duplicates().reset_index(drop=True)
    clean.to_csv(CLEAN, index=False)
    summary = {"raw_rows": len(raw), "clean_rows": len(clean),
               "duplicates_removed": len(raw) - len(clean),
               "quality_counts": clean.quality.value_counts().sort_index().to_dict()}
    (TABLES / "data_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return clean


def split_data():
    frame = pd.read_csv(CLEAN)
    parts = train_test_split(frame[FEATURES], frame.quality, test_size=0.2,
                             random_state=SEED, stratify=frame.quality)
    for name, part in zip(["X_train", "X_test", "y_train", "y_test"], parts):
        part.to_csv(SPLITS / f"{name}.csv", index=False)


def load_split(part):
    return (pd.read_csv(SPLITS / f"X_{part}.csv"),
            pd.read_csv(SPLITS / f"y_{part}.csv")["quality"])
