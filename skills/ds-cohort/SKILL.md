---
name: ds-cohort
description: Cohort analysis from an event log: monthly/weekly retention matrix, size-weighted retention curve, cohort sizes and optional cumulative LTV per user. Use for /ds-cohort, 'retention analizi', 'kohort analizi', 'müşteri elde tutma', 'LTV'.
argument-hint: --data <path> --user <user_col> --date <date_col> [--value <revenue_col>] [--period month|week] [--format landscape|portrait] [--theme light|dark] [--out <dir>]
---

# /ds-cohort — Cohort Retention & LTV

Run the bundled script. Do not write ad-hoc pandas/matplotlib code for this task.

```bash
uv run --project "{{DS_SKILLER_ROOT}}" python "{{DS_SKILLER_ROOT}}/skills/ds-cohort/scripts/run.py" --data "<events_file>" --user "<user_col>" --date "<date_col>" [--value "<revenue_col>"] [--period month] [--format landscape|portrait] [--theme light|dark]
```

- The absolute repo path above is written by the installer. If you see a raw placeholder, you are inside the ds-skiller repo: use `.`.
- Data paths and `--out` are relative to the current working directory. Default output: `outputs/ds-cohort/`.
- `--format portrait` = 9:16 Reels/Shorts PNG. `--theme dark` = graphite theme.
- PNG needs Chrome or Edge (override path with `DS_SKILLER_BROWSER`); without it only the HTML is written.

## Input
Event-level rows (one row per activity/order) with user id and date; optional revenue.

## Output
- `retention_matrix.csv`
- `cohort_dashboard.png / .html`
- `summary.json`

## Chaining
Low early retention -> build a churn target and run `ds-classification`.

## Response Rules (token-friendly)
- Never print raw DataFrames. Read `summary.json` (or the <=10 stdout lines), not the CSVs.
- Show/link the PNG dashboard (fallback: the HTML).
- Reply in **<= 3 short bullets**: M1 / M3 retention, best vs worst cohort, LTV per user (if --value).
- If the script fails, report the last error line and the likely fix (wrong column name, file path, missing model).
