---
name: ds-regression
description: Regression benchmark (XGBoost, CatBoost, LightGBM, RandomForest, Ridge) with K-fold CV, RMSE/MAE/R2/MAPE leaderboard, actual-vs-predicted, residual diagnostics and a saved model bundle. Use for /ds-regression, 'fiyat tahmini', 'regresyon modeli', numeric targets.
argument-hint: --data <path> --target <col> [--cv 5] [--format landscape|portrait] [--theme light|dark] [--out <dir>]
---

# /ds-regression — Regression Benchmark

Run the bundled script. Do not write ad-hoc pandas/matplotlib code for this task.

```bash
uv run --project "{{DS_SKILLER_ROOT}}" python "{{DS_SKILLER_ROOT}}/skills/ds-regression/scripts/run.py" --data "<data_path>" --target "<col>" [--cv 5] [--format landscape|portrait] [--theme light|dark]
```

- The absolute repo path above is written by the installer. If you see a raw placeholder, you are inside the ds-skiller repo: use `.`.
- Data paths and `--out` are relative to the current working directory. Default output: `outputs/ds-regression/`.
- `--format portrait` = 9:16 Reels/Shorts PNG. `--theme dark` = graphite theme.
- PNG needs Chrome or Edge (override path with `DS_SKILLER_BROWSER`); without it only the HTML is written.

## Input
Tabular file with a numeric target.

## Output
- `best_model.joblib`
- `test_predictions.csv`
- `regression_dashboard.png / .html`
- `summary.json`

## Chaining
`ds-explain` (SHAP), `ds-predict` (new data), `ds-tune`.

## Response Rules (token-friendly)
- Never print raw DataFrames. Read `summary.json` (or the <=10 stdout lines), not the CSVs.
- Show/link the PNG dashboard (fallback: the HTML).
- Reply in **<= 4 short bullets**: best model, test R2 / RMSE / MAE, residual bias warning, saved model path.
- If the script fails, report the last error line and the likely fix (wrong column name, file path, missing model).
