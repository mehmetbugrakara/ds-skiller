---
name: ds-metrics
description: Deep evaluation of a binary classifier: PR-AUC, KS statistic, Brier calibration curve, cumulative gains/lift and a threshold table. Use for /ds-metrics, 'eşik değeri seç', 'lift / KS', after ds-classification.
argument-hint: [--preds outputs/ds-classification/test_predictions.csv] [--format landscape|portrait] [--theme light|dark] [--out <dir>]
---

# /ds-metrics — Model Evaluation & Business Metrics

Run the bundled script. Do not write ad-hoc pandas/matplotlib code for this task.

```bash
uv run --project "{{DS_SKILLER_ROOT}}" python "{{DS_SKILLER_ROOT}}/skills/ds-metrics/scripts/run.py" --preds "<preds_csv>" [--format landscape|portrait] [--theme light|dark]
```

- The absolute repo path above is written by the installer. If you see a raw placeholder, you are inside the ds-skiller repo: use `.`.
- Data paths and `--out` are relative to the current working directory. Default output: `outputs/ds-metrics/`.
- `--format portrait` = 9:16 Reels/Shorts PNG. `--theme dark` = graphite theme.
- PNG needs Chrome or Edge (override path with `DS_SKILLER_BROWSER`); without it only the HTML is written.

## Input
Predictions CSV with `y_true` and `y_prob` columns (written by ds-classification for binary targets).

## Output
- `metrics_dashboard.png / .html`
- `summary.json`

## Chaining
Use the chosen threshold in `ds-predict --threshold`.

## Response Rules (token-friendly)
- Never print raw DataFrames. Read `summary.json` (or the <=10 stdout lines), not the CSVs.
- Show/link the PNG dashboard (fallback: the HTML).
- Reply in **<= 3 short bullets**: KS, Brier / calibration, optimal threshold and lift at top 20%.
- If the script fails, report the last error line and the likely fix (wrong column name, file path, missing model).
