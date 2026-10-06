# Red wine quality prediction

This project estimates expert quality ratings from 11 physicochemical measurements
in the UCI red Wine Quality dataset. It is an experimental analysis of 1,599 source
records, with 1,359 remaining after exact duplicate removal. Observed ratings range
from 3 to 8. Ratings 5 and 6 dominate, so exact accuracy alone gives an incomplete
picture of model performance.

## Run

From this repository, create an isolated environment and install the package:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m wine_quality run --jobs 2
python -m unittest discover -s tests -v
```

The editable installation is intentional: data and artifacts are resolved relative
to the repository containing the package, independent of the working directory.
After installation, `wine-quality` is also available from another directory.
This is a repository workflow, not a standalone distributable model service.
`requirements-tested.txt` records the direct dependency versions used for the
verified run (Python 3.13.5); install it with `pip install -r requirements-tested.txt`
when recreating that environment. It is not a complete transitive dependency lock.

Individual steps are `clean`, `eda`, `split`, `train`, `evaluate`, and `predict`.
Run them in that order when rebuilding artifacts. Training performs 5-fold
cross-validation and 80 grid-search fits plus refitting; runtime depends on hardware.
Use `--jobs 1` for sequential execution, or a positive number to parallelize CV.
Each forest uses one worker to avoid nested CPU oversubscription.

For your own data:

```sh
python -m wine_quality predict --input /path/to/wines.csv --output /path/to/predictions.csv
```

Input columns must match `data/examples/wines.csv`; their order may differ. Missing,
extra, negative, nonnumeric, and nonfinite values are rejected. Measurements must
use the source dataset's units. This basic validation does not establish that a
wine lies within the model's training distribution.

## Structure

```text
wine_quality/       Importable data, EDA, training, evaluation, prediction, and CLI modules
data/raw/           Original semicolon-delimited dataset
data/processed/     Deduplicated dataset
data/splits/        Stratified 80/20 train/test split
data/examples/      Illustrative prediction inputs
models/            Fitted candidates, selected model, and selection metadata
reports/figures/    EDA, confusion matrices, and feature-importance plots
reports/tables/     CV results, test metrics, per-class reports, and predictions
tests/             Input validation, metric, and data separation checks
archive/original_run/  Original scripts, README, models, and reports for reference
```

## Modeling and evaluation

The training candidates are a most-frequent-class dummy model, a baseline Random
Forest classifier, a class-weighted Random Forest classifier, a Random Forest
regressor, and a tuned classifier. Regression respects numerical distance during
training; its predictions are rounded to the nearest integer and clipped to 3–8
for evaluation. It is a comparison, not an assumption that regression is superior.

All candidates use the same stratified CV folds within the training set. Selection
maximizes macro F1, which gives each quality class equal weight. The selected
estimator is fitted on the training set only. The held-out test set is used afterward
for reporting, never for selecting a candidate or hyperparameters.

Read current results in [the generated evaluation summary](reports/tables/evaluation_summary.md)
and [model comparison](reports/tables/model_comparison.csv). These are regenerated
from model predictions, rather than copied into this README. The selected model
and runtime version are recorded in `models/metadata.json`. CV scores for the tuned
candidate are selection scores and may be optimistic; the test results are separate.

Metrics include accuracy, weighted and macro F1, balanced accuracy, mean absolute
rating error, within-one-point accuracy, and quadratic weighted kappa. Per-class
reports expose failure on rare ratings. Within-one-point accuracy is permissive;
compare it with the dummy baseline as well as exact accuracy.

The three example profiles demonstrate inference only. They have no ground-truth
labels and provide no additional evidence of generalization. Classifier probability
outputs are explicitly uncalibrated, not verified confidence estimates. Feature
importances and correlations describe associations, not causal effects.

This small dataset and single held-out split cannot establish an accuracy ceiling
or prove that rare ratings are impossible to learn. Further validation would require
more data or repeated/nested evaluation. Changing to low/medium/high bands would
change the prediction task and should follow an explicit use-case decision.

## Original results and provenance

The original saved comparison reported baseline accuracy 0.6140 / weighted F1
0.5949 and tuned accuracy 0.6103 / weighted F1 0.5849. Its winning recorded depth
was 20. The archived README had inconsistent values; use the new generated reports
for the current workflow. Old artifacts are preserved but are not loaded by the CLI.

Dataset attribution: P. Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis (2009),
UCI Wine Quality, red-wine subset. The original project supplied a local CSV and
described obtaining it via a public GitHub mirror; the exact mirror was not recorded.
