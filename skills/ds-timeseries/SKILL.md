---
name: ds-timeseries
description: Time series diagnostics and forecasting: ADF stationarity, seasonal decomposition, holdout backtest (MAPE/RMSE), Holt-Winters forecast with 95% intervals. Use for /ds-timeseries, /forecast, 'satış tahmini', 'zaman serisi'.
argument-hint: --data <path> --date <date_col> --target <value_col> [--horizon 14] [--period 7] [--format landscape|portrait] [--theme light|dark] [--out <dir>]
---

# /ds-timeseries — Time Series Forecasting

Run the bundled script. Do not write ad-hoc pandas/matplotlib code for this task.

```bash
uv run --project "{{DS_SKILLER_ROOT}}" python "{{DS_SKILLER_ROOT}}/skills/ds-timeseries/scripts/run.py" --data "<data_path>" --date "<date_col>" --target "<value_col>" [--horizon 14] [--period 7] [--format landscape|portrait] [--theme light|dark]
```

- The absolute repo path above is written by the installer. If you see a raw placeholder, you are inside the ds-skiller repo: use `.`.
- Data paths and `--out` are relative to the current working directory. Default output: `outputs/ds-timeseries/`.
- `--format portrait` = 9:16 Reels/Shorts PNG. `--theme dark` = graphite theme.
- PNG needs Chrome or Edge (override path with `DS_SKILLER_BROWSER`); without it only the HTML is written.

## Input
One row per timestamp (regular frequency). `--period` = seasonal cycle length (7 for daily-weekly, 12 for monthly-yearly).

## Output
- `forecast_dashboard.png / .html`
- `summary.json`

## Chaining
Event logs per user instead of an aggregate series? -> `ds-cohort`.

## Response Rules (token-friendly)
- Never print raw DataFrames. Read `summary.json` (or the <=10 stdout lines), not the CSVs.
- Show/link the PNG dashboard (fallback: the HTML).
- Reply in **<= 3 short bullets**: ADF result, backtest RMSE & MAPE, forecast trend direction.
- If the script fails, report the last error line and the likely fix (wrong column name, file path, missing model).
