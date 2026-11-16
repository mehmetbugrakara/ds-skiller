---
name: ds-tune
description: Randomized hyperparameter search across XGBoost, CatBoost and LightGBM plus a bias/variance learning curve; saves tuned_model.joblib. Use for /ds-tune, 'modeli optimize et', 'hiperparametre ayarı'.
argument-hint: --data <path> --target <col> [--algo all|xgboost|lightgbm|catboost] [--trials 12] [--cv 4] [--format landscape|portrait] [--theme light|dark] [--out <dir>]
---

# /ds-tune — Hyperparameter Tuning

Run the bundled script. Do not write ad-hoc pandas/matplotlib code for this task.

```bash
uv run --project "{{DS_SKILLER_ROOT}}" python "{{DS_SKILLER_ROOT}}/skills/ds-tune/scripts/run.py" --data "<data_path>" --target "<col>" [--algo all] [--trials 12] [--cv 4] [--format landscape|portrait] [--theme light|dark]
```

- The absolute repo path above is written by the installer. If you see a raw placeholder, you are inside the ds-skiller repo: use `.`.
- Data paths and `--out` are relative to the current working directory. Default output: `outputs/ds-tune/`.
- `--format portrait` = 9:16 Reels/Shorts PNG. `--theme dark` = graphite theme.
- PNG needs Chrome or Edge (override path with `DS_SKILLER_BROWSER`); without it only the HTML is written.

## Input
Same as classification/regression (task auto-detected). Slow: trials x cv x algos fits.

## Output
- `tuned_model.joblib`
- `tune_dashboard.png / .html`
- `summary.json`

## Chaining
`ds-predict --model outputs/ds-tune/tuned_model.joblib`, `ds-explain`.

## Response Rules (token-friendly)
- Never print raw DataFrames. Read `summary.json` (or the <=10 stdout lines), not the CSVs.
- Show/link the PNG dashboard (fallback: the HTML).
- Reply in **<= 3 short bullets**: winning GBDT family & best CV score, bias/variance diagnosis, saved model path.
- If the script fails, report the last error line and the likely fix (wrong column name, file path, missing model).
