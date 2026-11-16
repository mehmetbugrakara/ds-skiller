---
name: ds-eda
description: 8-panel senior EDA: target balance/skew, Mutual Information signal ranking, missingness, IQR/Z outliers, quantiles/kurtosis, categorical cardinality, correlation heatmap and an automated hygiene/leakage audit. Use for /ds-eda, 'analyze this dataset', 'veriyi incele', 'EDA yap', or as the first step on any new tabular file.
argument-hint: --data <path> [--target <col>] [--format landscape|portrait] [--theme light|dark] [--out <dir>]
---

# /ds-eda — Exploratory Data Analysis

Run the bundled script. Do not write ad-hoc pandas/matplotlib code for this task.

```bash
uv run --project "{{DS_SKILLER_ROOT}}" python "{{DS_SKILLER_ROOT}}/skills/ds-eda/scripts/run.py" --data "<data_path>" [--target "<col>"] [--format landscape|portrait] [--theme light|dark]
```

- The absolute repo path above is written by the installer. If you see a raw placeholder, you are inside the ds-skiller repo: use `.`.
- Data paths and `--out` are relative to the current working directory. Default output: `outputs/ds-eda/`.
- `--format portrait` = 9:16 Reels/Shorts PNG. `--theme dark` = graphite theme.
- PNG needs Chrome or Edge (override path with `DS_SKILLER_BROWSER`); without it only the HTML is written.

## Input
Tabular file (csv/tsv/parquet/feather/xlsx/json). `--target` optional; enables MI ranking and leakage checks.

## Output
- `eda_dashboard.png / .html`
- `summary.json`

## Chaining
Next: `ds-clean` if hygiene issues, then `ds-features` / `ds-pipeline`.

## Response Rules (token-friendly)
- Never print raw DataFrames. Read `summary.json` (or the <=10 stdout lines), not the CSVs.
- Show/link the PNG dashboard (fallback: the HTML).
- Reply in **<= 4 short bullets**: shape + hygiene score, top target drivers (MI), missing/outlier/skew flags, leakage/ID/collinearity warnings.
- If the script fails, report the last error line and the likely fix (wrong column name, file path, missing model).
