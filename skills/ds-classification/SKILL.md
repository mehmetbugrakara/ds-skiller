---
name: ds-classification
description: Classification benchmark (XGBoost, CatBoost, LightGBM, RandomForest, LogReg) with stratified CV, ROC curves, confusion matrix, feature importance and a saved model bundle. Use for /ds-classification, 'sınıflandırma modeli kur', 'churn modeli', binary or multi-class targets.
argument-hint: --data <path> --target <col> [--cv 5] [--format landscape|portrait] [--theme light|dark] [--out <dir>]
---

# /ds-classification — Classification Benchmark

Run the bundled script. Do not write ad-hoc pandas/matplotlib code for this task.

```bash
uv run --project "{{DS_SKILLER_ROOT}}" python "{{DS_SKILLER_ROOT}}/skills/ds-classification/scripts/run.py" --data "<data_path>" --target "<col>" [--cv 5] [--format landscape|portrait] [--theme light|dark]
```

- The absolute repo path above is written by the installer. If you see a raw placeholder, you are inside the ds-skiller repo: use `.`.
- Data paths and `--out` are relative to the current working directory. Default output: `outputs/ds-classification/`.
- `--format portrait` = 9:16 Reels/Shorts PNG. `--theme dark` = graphite theme.
- PNG needs Chrome or Edge (override path with `DS_SKILLER_BROWSER`); without it only the HTML is written.

## Input
Tabular file, categorical/binary target (>=2 classes, each with enough rows for stratified CV).

## Output
- `best_model.joblib` (model, label_encoder, features, target)
- `test_predictions.csv` (y_true, y_pred, y_prob)
- `classification_dashboard.png / .html`
- `summary.json`

## Chaining
`ds-explain` (SHAP), `ds-metrics` (binary only), `ds-audit` (fairness), `ds-predict` (new data), `ds-tune`.

## Response Rules (token-friendly)
- Never print raw DataFrames. Read `summary.json` (or the <=10 stdout lines), not the CSVs.
- Show/link the PNG dashboard (fallback: the HTML).
- Reply in **<= 4 short bullets**: best model, test ROC-AUC & F1, imbalance warning, saved model path.
- If the script fails, report the last error line and the likely fix (wrong column name, file path, missing model).
