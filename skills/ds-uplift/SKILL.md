---
name: ds-uplift
description: Uplift / heterogeneous treatment effect with a LightGBM T-learner: ATE, Qini curve vs random, observed uplift by decile, uplift drivers and per-row uplift scores. Use for /ds-uplift, 'kampanyayı kime göstermeliyim', 'persuadables', 'treatment effect'.
argument-hint: --data <path> --treatment <binary_col> --outcome <col> [--format landscape|portrait] [--theme light|dark] [--out <dir>]
---

# /ds-uplift — Uplift Modeling

Run the bundled script. Do not write ad-hoc pandas/matplotlib code for this task.

```bash
uv run --project "{{DS_SKILLER_ROOT}}" python "{{DS_SKILLER_ROOT}}/skills/ds-uplift/scripts/run.py" --data "<data_path>" --treatment "<treatment_col>" --outcome "<outcome_col>" [--format landscape|portrait] [--theme light|dark]
```

- The absolute repo path above is written by the installer. If you see a raw placeholder, you are inside the ds-skiller repo: use `.`.
- Data paths and `--out` are relative to the current working directory. Default output: `outputs/ds-uplift/`.
- `--format portrait` = 9:16 Reels/Shorts PNG. `--theme dark` = graphite theme.
- PNG needs Chrome or Edge (override path with `DS_SKILLER_BROWSER`); without it only the HTML is written.

## Input
Randomized experiment data: binary treatment flag, outcome (binary or numeric), pre-treatment features only.

## Output
- `uplift_scores.csv` (sorted by uplift_score)
- `uplift_dashboard.png / .html`
- `summary.json`

## Chaining
Run `ds-abtest` first for the overall effect; target top deciles from uplift_scores.csv.

## Response Rules (token-friendly)
- Never print raw DataFrames. Read `summary.json` (or the <=10 stdout lines), not the CSVs.
- Show/link the PNG dashboard (fallback: the HTML).
- Reply in **<= 3 short bullets**: ATE, top-30% uplift vs ATE, Qini area (>0 = useful), share with negative uplift.
- If the script fails, report the last error line and the likely fix (wrong column name, file path, missing model).
