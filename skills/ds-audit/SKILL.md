---
name: ds-audit
description: Subgroup fairness and slice error audit for a saved classifier: demographic parity, equalized odds (TPR/FPR gap), segment-level ROC-AUC/F1/accuracy. Use for /ds-audit, /fairness, 'segment bazında model hatası', 'adillik analizi'.
argument-hint: --data <path> --slice <segment_col> [--model <model_joblib>] [--format landscape|portrait] [--theme light|dark] [--out <dir>]
---

# /ds-audit — Fairness & Slice Audit

Run the bundled script. Do not write ad-hoc pandas/matplotlib code for this task.

```bash
uv run --project "{{DS_SKILLER_ROOT}}" python "{{DS_SKILLER_ROOT}}/skills/ds-audit/scripts/run.py" --data "<data_path>" --slice "<segment_col>" [--model "<model_joblib>"] [--format landscape|portrait] [--theme light|dark]
```

- The absolute repo path above is written by the installer. If you see a raw placeholder, you are inside the ds-skiller repo: use `.`.
- Data paths and `--out` are relative to the current working directory. Default output: `outputs/ds-audit/`.
- `--format portrait` = 9:16 Reels/Shorts PNG. `--theme dark` = graphite theme.
- PNG needs Chrome or Edge (override path with `DS_SKILLER_BROWSER`); without it only the HTML is written.

## Input
Raw tabular file that contains the model's target column AND the slice column (e.g. the original training CSV).

## Output
- `audit_dashboard.png / .html`
- `summary.json`

## Chaining
Runs after `ds-classification` (default model path).

## Response Rules (token-friendly)
- Never print raw DataFrames. Read `summary.json` (or the <=10 stdout lines), not the CSVs.
- Show/link the PNG dashboard (fallback: the HTML).
- Reply in **<= 3 short bullets**: demographic parity gap, equalized odds gap, worst segment.
- If the script fails, report the last error line and the likely fix (wrong column name, file path, missing model).
