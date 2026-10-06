from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/winequality-red.csv"
CLEAN = ROOT / "data/processed/winequality_clean.csv"
SPLITS = ROOT / "data/splits"
MODELS = ROOT / "models"
TABLES = ROOT / "reports/tables"
FIGURES = ROOT / "reports/figures"
FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
]
LABELS = [3, 4, 5, 6, 7, 8]
SEED = 42


def prepare_directories():
    for path in [CLEAN.parent, SPLITS, MODELS, TABLES, FIGURES]:
        path.mkdir(parents=True, exist_ok=True)
