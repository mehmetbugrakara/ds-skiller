---
name: ds-pipeline
description: One command end-to-end run: EDA -> Clean -> Features -> Classification/Regression (auto-detected) -> SHAP -> Metrics, plus an index dashboard of every step's KPIs and warnings. Use for /ds-pipeline, 'baştan sona analiz', 'full pipeline', 'her şeyi çalıştır'.
argument-hint: --data <path> --target <col> [--task auto|classification|regression] [--skip ds-explain ...] [--cv 5] [--format landscape|portrait] [--theme light|dark] [--out <dir>]
---

# /ds-pipeline — End-to-End DS Pipeline

Run the bundled script. Do not write ad-hoc pandas/matplotlib code for this task.

```bash
uv run --project "{{DS_SKILLER_ROOT}}" python "{{DS_SKILLER_ROOT}}/skills/ds-pipeline/scripts/run.py" --data "<data_path>" --target "<col>" [--task auto] [--skip ds-features] [--cv 5] [--format landscape|portrait] [--theme light|dark]
```

- The absolute repo path above is written by the installer. If you see a raw placeholder, you are inside the ds-skiller repo: use `.`.
- Data paths and `--out` are relative to the current working directory. Default output: `outputs/ds-pipeline/`.
- `--format portrait` = 9:16 Reels/Shorts PNG. `--theme dark` = graphite theme.
- PNG needs Chrome or Edge (override path with `DS_SKILLER_BROWSER`); without it only the HTML is written.

## Input
Tabular file with a target column. Takes a few minutes (5 models x CV + SHAP).

## Output
- `<out>/ds-*/` (each step's normal outputs)
- `pipeline_dashboard.png / .html` (index)
- `summary.json` (all steps)

## Chaining
Then `ds-predict --model <out>/ds-classification/best_model.joblib` on new data.

## Response Rules (token-friendly)
- Never print raw DataFrames. Read `summary.json` (or the <=10 stdout lines), not the CSVs.
- Show/link the PNG dashboard (fallback: the HTML).
- Reply in **<= 4 short bullets**: task & best model with headline metric, failed steps (if any), top 2 warnings, index dashboard path.
- If the script fails, report the last error line and the likely fix (wrong column name, file path, missing model).
