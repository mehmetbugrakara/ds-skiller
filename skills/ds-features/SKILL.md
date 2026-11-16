---
name: ds-features
description: Feature selection audit: VIF multicollinearity, Mutual Information, Spearman |r|, redundant pairs and Keep/Drop recommendations. Use for /ds-features, /vif, 'hangi değişkenleri atayım', 'feature selection'.
argument-hint: --data <path> --target <col> [--format landscape|portrait] [--theme light|dark] [--out <dir>]
---

# /ds-features — Feature Selection & Multicollinearity

Run the bundled script. Do not write ad-hoc pandas/matplotlib code for this task.

```bash
uv run --project "{{DS_SKILLER_ROOT}}" python "{{DS_SKILLER_ROOT}}/skills/ds-features/scripts/run.py" --data "<data_path>" --target "<col>" [--format landscape|portrait] [--theme light|dark]
```

- The absolute repo path above is written by the installer. If you see a raw placeholder, you are inside the ds-skiller repo: use `.`.
- Data paths and `--out` are relative to the current working directory. Default output: `outputs/ds-features/`.
- `--format portrait` = 9:16 Reels/Shorts PNG. `--theme dark` = graphite theme.
- PNG needs Chrome or Edge (override path with `DS_SKILLER_BROWSER`); without it only the HTML is written.

## Input
Tabular file with a target column (classification or regression).

## Output
- `features_dashboard.png / .html`
- `summary.json`

## Chaining
Drop recommended columns, then `ds-classification` / `ds-regression`.

## Response Rules (token-friendly)
- Never print raw DataFrames. Read `summary.json` (or the <=10 stdout lines), not the CSVs.
- Show/link the PNG dashboard (fallback: the HTML).
- Reply in **<= 3 short bullets**: top MI features, high-VIF (>5) features, recommended drops.
- If the script fails, report the last error line and the likely fix (wrong column name, file path, missing model).
