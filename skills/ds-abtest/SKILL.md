---
name: ds-abtest
description: A/B test analysis: SRM check, two-proportion Z-test or Welch t-test + Mann-Whitney U, 95% CI, statistical power and a Ship/Hold/Stop decision card. Use for /ds-abtest, 'A/B testi sonucu', 'kampanya anlamlı mı'.
argument-hint: --data <path> --group <group_col> --metric <metric_col> [--control A] [--variant B] [--format landscape|portrait] [--theme light|dark] [--out <dir>]
---

# /ds-abtest — A/B Test & Hypothesis Testing

Run the bundled script. Do not write ad-hoc pandas/matplotlib code for this task.

```bash
uv run --project "{{DS_SKILLER_ROOT}}" python "{{DS_SKILLER_ROOT}}/skills/ds-abtest/scripts/run.py" --data "<data_path>" --group "<group_col>" --metric "<metric_col>" [--control A] [--variant B] [--format landscape|portrait] [--theme light|dark]
```

- The absolute repo path above is written by the installer. If you see a raw placeholder, you are inside the ds-skiller repo: use `.`.
- Data paths and `--out` are relative to the current working directory. Default output: `outputs/ds-abtest/`.
- `--format portrait` = 9:16 Reels/Shorts PNG. `--theme dark` = graphite theme.
- PNG needs Chrome or Edge (override path with `DS_SKILLER_BROWSER`); without it only the HTML is written.

## Input
One row per unit; group column with 2 arms; metric binary (conversion) or numeric (revenue).

## Output
- `abtest_dashboard.png / .html`
- `summary.json`

## Chaining
Significant but heterogeneous effect? -> `ds-uplift` to find who to target.

## Response Rules (token-friendly)
- Never print raw DataFrames. Read `summary.json` (or the <=10 stdout lines), not the CSVs.
- Show/link the PNG dashboard (fallback: the HTML).
- Reply in **<= 3 short bullets**: Ship/Hold/Stop decision, relative lift & 95% CI, SRM / power check.
- If the script fails, report the last error line and the likely fix (wrong column name, file path, missing model).
