# Experiment roadmap

Competition metric: ROC AUC. User-reported baseline leaderboard score: **0.94157**.
This leaderboard result is separate from local holdout and OOF validation.

Only the preserved starter and initial EDA pair have been created. Future work
will proceed one or two hypotheses at a time, after the user runs validation.
Every new notebook will have an `.ipynb` / percent-format `.py` pair.

| Notebook | Planned comparison and rationale |
| --- | --- |
| `01_eda` | Inspect distributions, category levels, duplicates, train/test differences, and synthetic precision artifacts. Use findings to select focused hypotheses. |
| `02_catboost_cv_baseline` | Establish a reproducible stratified 5-fold baseline using the starter settings. Common folds make later OOF comparisons and blending meaningful. |
| `03_ordinal_encoding` | Compare native CatBoost categories with fixed Low < Medium < High range-anxiety codes. Ordered thresholds may simplify splits. Environmental concern already has numeric levels 1–5; verify semantics before changing its representation. |
| `04_catboost_small` | Compare a smaller CatBoost candidate with the CV baseline to measure the accuracy/runtime tradeoff. |
| `05_xgboost_small` | Evaluate a compact XGBoost candidate on the same folds; different errors may help an ensemble. |
| `06_lightgbm_small` | Evaluate a compact LightGBM candidate on the same folds and compare error diversity. |
| `07_neural_network` | Evaluate a small tabular neural network with fold-fitted preprocessing; keep it only if standalone or blend validation supports it. |
| `08_simple_ensemble` | Compare a simple probability average against the best single model using aligned OOF predictions. Model count alone does not guarantee improvement. |
| `09_parameter_tuning_submission` | Use bounded tuning for promising candidates, then compare the tuned blend with the untuned blend. The user runs final fitting/prediction and creates a submission after validation. |
| `10_feature_engineering` | Test one or two EDA-motivated features at a time, such as charging access or commute/access combinations, against the tuned baseline. Reject changes that degrade OOF ROC AUC. |
| `11_hill_climbing` | Optimize ensemble weights using saved, aligned OOF predictions and compare with the simple blend. Judge selected weights on a separate meta-validation split or nested procedure: optimizing and scoring on the same OOF labels can overfit. |

## Validation rules for future notebooks

- Fit all learned preprocessing strictly within each training fold.
- Save row IDs, fold assignments, targets, and OOF probabilities, plus fold and aggregate ROC AUC scores.
- Use identical folds across comparable experiments. Evaluate any original-source augmentation only on synthetic competition validation rows.
- Keep reusable transformations, metrics, and model wrappers in `src/`.
- The user runs all training, prediction, and modeling pipelines.

## Preserved starter notes

The starter excludes `id`, maps No/Yes to 0/1, and uses all 13 remaining predictors
without feature engineering. Six string columns receive native CatBoost handling;
seven numeric columns include environmental concern.

It uses a stratified 80/20 split with seed 42, CPU CatBoost, Logloss training and
AUC evaluation, depth 6, learning rate 0.08, up to 1,600 trees, and early stopping
after 120 rounds. It refits on all training rows using the selected tree count.
It writes holdout predictions but does not implement full OOF validation.

The original code is preserved: two exploratory cells reference undefined
`train_features`, and input/output paths assume Kaggle. These need adjustment
before a clean local run; no starter code was executed during organization.
