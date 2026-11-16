---
name: ds-drift
description: Reference vs current drift monitor: PSI, bin-level CSI, Jensen-Shannon, Wasserstein, KS-test, Chi-square. Use for /ds-drift, /psi, 'veri kayması', 'model yeniden eğitilmeli mi', production monitoring.
argument-hint: --ref <ref_file> --curr <curr_file> [--target <col>] [--format landscape|portrait] [--theme light|dark] [--out <dir>]
---

# /ds-drift — Data Drift Monitor

Run the bundled script. Do not write ad-hoc pandas/matplotlib code for this task.

```bash
uv run --project "{{DS_SKILLER_ROOT}}" python "{{DS_SKILLER_ROOT}}/skills/ds-drift/scripts/run.py" --ref "<ref_file>" --curr "<curr_file>" [--target "<col>"] [--format landscape|portrait] [--theme light|dark]
```

- The absolute repo path above is written by the installer. If you see a raw placeholder, you are inside the ds-skiller repo: use `.`.
- Data paths and `--out` are relative to the current working directory. Default output: `outputs/ds-drift/`.
- `--format portrait` = 9:16 Reels/Shorts PNG. `--theme dark` = graphite theme.
- PNG needs Chrome or Edge (override path with `DS_SKILLER_BROWSER`); without it only the HTML is written.

## Input
Two tabular files with the same columns (training snapshot vs recent data).

## Output
- `drift_dashboard.png / .html`
- `summary.json`

## Chaining
High drift -> retrain with `ds-classification` / `ds-regression`; check live accuracy with `ds-predict`.

## Response Rules (token-friendly)
- Never print raw DataFrames. Read `summary.json` (or the <=10 stdout lines), not the CSVs.
- Show/link the PNG dashboard (fallback: the HTML).
- Reply in **<= 4 short bullets**: high/moderate drift counts, max PSI & JS, CSI hotspot, retrain/monitor action.
- If the script fails, report the last error line and the likely fix (wrong column name, file path, missing model).
