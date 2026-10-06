# Wine Quality Prediction — Analysis Report

## 1. Overview

This project builds a Random Forest classifier to predict red wine quality
from physicochemical lab measurements, using the well-known **UCI Wine
Quality (Red) dataset** (Cortez et al., 2009).

**Pipeline:** Load dataset → data cleaning → EDA → train/test split →
Random Forest classification → evaluation → hyperparameter tuning →
prediction on new samples.

| Metric | Value |
|---|---|
| Source | UCI Wine Quality (Red), via public GitHub mirror |
| Raw records | 1,599 |
| Records after cleaning | 1,359 (240 duplicates removed) |
| Features | 11 physicochemical measurements |
| Target | Quality score (expert median rating, 3–8 observed) |
| Train / Test split | 1,087 / 272 (80/20, stratified) |
| Baseline RF accuracy | 61.4% |
| Tuned RF accuracy | 61.0% (weighted F1: 0.585) |

## 2. Data Loading

Loaded directly from the original semicolon-delimited CSV format used by
the UCI repository:
```python
df = pd.read_csv("winequality-red.csv", sep=";")
```
11 input features (fixed acidity, volatile acidity, citric acid, residual
sugar, chlorides, free/total sulfur dioxide, density, pH, sulphates,
alcohol) plus the integer `quality` target (expert-rated 0–10 scale; this
dataset only contains scores 3–8).

## 3. Data Cleaning

- **Missing values:** none found.
- **Duplicate rows:** **240 exact duplicates** found and removed (a known,
  documented quirk of this real-world dataset) → 1,359 clean rows remain.
- **Data types:** all correct on load (10 floats + 1 int target).
- **Negative-value sanity check:** no invalid negative chemistry values.
- **Outliers:** checked via IQR rule per feature and reported (not
  removed) — residual sugar (126) and chlorides (87) have the most
  outliers, which reflects genuine chemical variability across wines
  rather than data-entry error, so Random Forest (robust to outliers) was
  used without additional outlier-removal.

## 4. Exploratory Data Analysis (EDA)

**Quality distribution** is imbalanced and roughly bell-shaped, dominated
by mid-range scores:

| Quality | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| Count | 10 | 53 | 577 | 535 | 167 | 17 |

Scores 5 and 6 make up **82%** of all wines; scores 3 and 8 (the extremes)
are rare — this class imbalance directly limits achievable accuracy on the
minority classes (see Section 6).

**Correlation with quality** (sorted):

| Feature | Correlation |
|---|---|
| alcohol | **+0.480** |
| sulphates | +0.249 |
| citric acid | +0.228 |
| fixed acidity | +0.119 |
| residual sugar | +0.014 |
| free sulfur dioxide | −0.050 |
| pH | −0.055 |
| chlorides | −0.131 |
| total sulfur dioxide | −0.178 |
| density | −0.184 |
| volatile acidity | **−0.395** |

**Alcohol content** is the single strongest predictor of quality (higher
alcohol → higher rated wine), while **volatile acidity** (acetic
acid/vinegar taste) is the strongest negative predictor — both match
established wine science.

Visuals: `eda_quality_distribution.png`, `eda_feature_distributions.png`,
`eda_correlation.png`, `eda_top_features_vs_quality.png` (boxplots of the
6 most correlated features against quality).

## 5. Train/Test Split

80/20 split, **stratified** on quality score to preserve the imbalanced
class proportions in both sets (critical given how few quality-3 and
quality-8 samples exist):

- Train: 1,087 samples
- Test: 272 samples

## 6. Random Forest Classification & Evaluation

Baseline model: `RandomForestClassifier(n_estimators=200, random_state=42)`

**Test set performance:**

| Metric | Value |
|---|---|
| Accuracy | 61.4% |
| Weighted F1 | 0.595 |

**Per-class metrics:**

| Quality | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 3 | 0.00 | 0.00 | 0.00 | 2 |
| 4 | 0.00 | 0.00 | 0.00 | 11 |
| 5 | 0.69 | 0.72 | 0.71 | 116 |
| 6 | 0.57 | 0.66 | 0.61 | 107 |
| 7 | 0.52 | 0.36 | 0.43 | 33 |
| 8 | 0.00 | 0.00 | 0.00 | 3 |

**Key finding:** the model performs well on the common mid-range classes
(5, 6) but **cannot learn classes 3, 4, and 8 at all** — there simply
aren't enough training examples (as few as 2 in the test set). The
confusion matrix (`eval_confusion_matrix.png`) confirms errors are almost
entirely between *adjacent* quality scores (5↔6, 6↔7) rather than wild
misses — expected, since quality is an ordinal scale and a "6" rated wine
genuinely resembles a "5" or "7" chemically.

**Feature importance** (Random Forest's built-in importance):

| Feature | Importance |
|---|---|
| alcohol | 0.150 |
| sulphates | 0.113 |
| volatile acidity | 0.105 |
| total sulfur dioxide | 0.104 |
| density | 0.086 |

This confirms the correlation analysis — alcohol is the dominant driver,
consistent across two independent methods.

## 7. Hyperparameter Tuning

Used `GridSearchCV` (5-fold stratified CV, scoring = weighted F1) over:

```python
{
    "n_estimators": [200, 400],
    "max_depth": [None, 20, 30],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"],
}
```
48 combinations × 5 folds = 240 model fits.

**Best hyperparameters found:**
```
max_depth: 20
max_features: sqrt
min_samples_leaf: 2
min_samples_split: 5
n_estimators: 200
```

**Result:** tuned test accuracy was **61.0%**, slightly below the baseline's
**61.4%**. Weighted F1 was **0.585**, also slightly below the baseline's
**0.595**.

**Honest takeaway:** hyperparameter tuning did not meaningfully improve
this model. The baseline's default-ish settings were already close to
optimal for this data — the accuracy ceiling here is driven by the class
imbalance and inherent overlap between adjacent quality ratings, not by
suboptimal tree parameters. Further gains would more likely come from:
- **Rebalancing techniques** (SMOTE, class weighting) for the rare classes
- **Collapsing quality into 3 bands** (low/medium/high) for a more
  learnable, business-relevant classification task
- **Additional features** (e.g. grape variety, vintage) not present in
  this physicochemical-only dataset

## 8. Predicting Wine Quality — New Samples

The tuned model was applied to 3 illustrative new wine profiles:

| Sample | Alcohol | Volatile Acidity | Sulphates | Predicted Quality | Confidence |
|---|---|---|---|---|---|
| 1 (low alcohol, high VA) | 9.8 | 0.88 | 0.68 | **5** | 63% |
| 2 (balanced/average) | 10.2 | 0.50 | 0.62 | **6** | 52% |
| 3 (high alcohol, low VA, high sulphates) | 13.0 | 0.28 | 0.90 | **7** | 78% |

The predictions track the EDA findings exactly: as alcohol rises and
volatile acidity falls, predicted quality climbs from 5 → 6 → 7, with
prediction confidence also increasing — the model is most confident on the
extreme profile, less confident on borderline cases, which is the correct
behavior for an ordinal, overlapping-class problem like this one.

## 9. Files in This Project

| File | Description |
|---|---|
| `01_load_clean.py` | Loads raw CSV, cleans duplicates, sanity-checks data |
| `02_eda.py` | Full EDA: distributions, correlations, boxplots |
| `03_split_baseline_rf.py` | Train/test split + baseline Random Forest |
| `04_evaluate.py` | Confusion matrix, per-class metrics, feature importance |
| `05_hyperparameter_tuning.py` | GridSearchCV tuning + baseline vs. tuned comparison |
| `06_predict_new.py` | Predicts quality for 3 new wine samples |
| `winequality-red.csv` | Original raw dataset |
| `winequality_clean.csv` | Cleaned dataset (duplicates removed) |
| `X_train/X_test/y_train/y_test.csv` | Saved train/test splits |
| `rf_baseline_model.joblib`, `rf_tuned_model.joblib` | Saved trained models |
| `model_comparison.csv`, `best_hyperparameters.csv` | Tuning results |
| `eval_per_class_metrics.csv`, `new_wine_predictions.csv` | Result tables |
| `eda_*.png` | 4 EDA visualizations |
| `eval_*.png` | 3 evaluation visualizations |

## 10. How to Reproduce

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python 01_load_clean.py
python 02_eda.py
python 03_split_baseline_rf.py
python 04_evaluate.py
python 05_hyperparameter_tuning.py     # ~2-3 minutes (240 model fits)
python 06_predict_new.py
```

Run the commands from the project directory. The repository includes the
source data, generated analysis outputs, and trained models.

*Data source: [UCI Machine Learning Repository — Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)
(P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis, 2009), retrieved
via a public GitHub mirror of the original UCI CSV.*

Citation: Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J.
(2009). Modeling wine preferences by data mining from physicochemical
properties. *Decision Support Systems, 47*(4), 547–553.
https://doi.org/10.1016/j.dss.2009.05.016
