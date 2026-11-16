---
name: ds-clean
description: Automated cleaning: dedup, drop constant/ID columns, median/mode imputation, optional IQR outlier capping, cleaned CSV export with a before/after audit log. Use for /ds-clean, /preprocess, 'veriyi temizle', 'eksik değerleri doldur'.
argument-hint: --data <path> [--target <col>] [--clip-outliers] [--format landscape|portrait] [--theme light|dark] [--out <dir>]
---

# /ds-clean — Data Cleaning & Preprocessing

Run the bundled script. Do not write ad-hoc pandas/matplotlib code for this task.

```bash
uv run --project "{{DS_SKILLER_ROOT}}" python "{{DS_SKILLER_ROOT}}/skills/ds-clean/scripts/run.py" --data "<data_path>" [--target "<col>"] [--clip-outliers] [--format landscape|portrait] [--theme light|dark]
```

- The absolute repo path above is written by the installer. If you see a raw placeholder, you are inside the ds-skiller repo: use `.`.
- Data paths and `--out` are relative to the current working directory. Default output: `outputs/ds-clean/`.
- `--format portrait` = 9:16 Reels/Shorts PNG. `--theme dark` = graphite theme.
- PNG needs Chrome or Edge (override path with `DS_SKILLER_BROWSER`); without it only the HTML is written.

## Input
Tabular file. `--target` is protected from imputation/dropping.

## Output
- `<name>_cleaned.csv`
- `clean_dashboard.png / .html`
- `summary.json`

## Chaining
Feed `<name>_cleaned.csv` into `ds-features`, `ds-classification`, `ds-regression`.

## Response Rules (token-friendly)
- Never print raw DataFrames. Read `summary.json` (or the <=10 stdout lines), not the CSVs.
- Show/link the PNG dashboard (fallback: the HTML).
- Reply in **<= 3 short bullets**: actions taken (rows removed, columns dropped, cells imputed) and cleaned file path.
- If the script fails, report the last error line and the likely fix (wrong column name, file path, missing model).
