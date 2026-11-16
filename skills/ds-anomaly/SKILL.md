---
name: ds-anomaly
description: Unsupervised outlier detection with Isolation Forest + LOF + Robust Z-score consensus (>=2 of 3 votes), 2D PCA anomaly map and flagged-records CSV. Use for /ds-anomaly, /outliers, 'aykırı değerleri bul', 'fraud/anomali tespiti'.
argument-hint: --data <path> [--target <col_to_exclude>] [--contamination 0.05] [--format landscape|portrait] [--theme light|dark] [--out <dir>]
---

# /ds-anomaly — Consensus Anomaly Detection

Run the bundled script. Do not write ad-hoc pandas/matplotlib code for this task.

```bash
uv run --project "{{DS_SKILLER_ROOT}}" python "{{DS_SKILLER_ROOT}}/skills/ds-anomaly/scripts/run.py" --data "<data_path>" [--target "<col>"] [--contamination 0.05] [--format landscape|portrait] [--theme light|dark]
```

- The absolute repo path above is written by the installer. If you see a raw placeholder, you are inside the ds-skiller repo: use `.`.
- Data paths and `--out` are relative to the current working directory. Default output: `outputs/ds-anomaly/`.
- `--format portrait` = 9:16 Reels/Shorts PNG. `--theme dark` = graphite theme.
- PNG needs Chrome or Edge (override path with `DS_SKILLER_BROWSER`); without it only the HTML is written.

## Input
Tabular file with >=2 numeric columns. `--target` is excluded from detection.

## Output
- `anomalies_flagged.csv`
- `anomaly_dashboard.png / .html`
- `summary.json`

## Chaining
Inspect flagged rows; re-run `ds-clean` or exclude them before modeling.

## Response Rules (token-friendly)
- Never print raw DataFrames. Read `summary.json` (or the <=10 stdout lines), not the CSVs.
- Show/link the PNG dashboard (fallback: the HTML).
- Reply in **<= 3 short bullets**: consensus anomaly rate, top features driving anomalies, CSV path.
- If the script fails, report the last error line and the likely fix (wrong column name, file path, missing model).
