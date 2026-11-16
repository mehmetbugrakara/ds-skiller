---
name: ds-predict
description: Score new data with a saved ds-skiller model bundle (classification or regression): schema alignment, score/prediction distribution, predictions.csv, and live evaluation when the target column is present. Use for /ds-predict, 'yeni veriyi skorla', 'modeli kullan', 'tahmin üret', 'production scoring'.
argument-hint: --data <path> [--model outputs/ds-classification/best_model.joblib] [--threshold 0.5] [--id <id_col>] [--format landscape|portrait] [--theme light|dark] [--out <dir>]
---

# /ds-predict — Batch Inference / Scoring

Run the bundled script. Do not write ad-hoc pandas/matplotlib code for this task.

```bash
uv run --project "{{DS_SKILLER_ROOT}}" python "{{DS_SKILLER_ROOT}}/skills/ds-predict/scripts/run.py" --data "<new_data>" [--model "<model_joblib>"] [--threshold 0.5] [--id "<id_col>"] [--format landscape|portrait] [--theme light|dark]
```

- The absolute repo path above is written by the installer. If you see a raw placeholder, you are inside the ds-skiller repo: use `.`.
- Data paths and `--out` are relative to the current working directory. Default output: `outputs/ds-predict/`.
- `--format portrait` = 9:16 Reels/Shorts PNG. `--theme dark` = graphite theme.
- PNG needs Chrome or Edge (override path with `DS_SKILLER_BROWSER`); without it only the HTML is written.

## Input
New rows with the same raw columns used in training (target optional). Missing trained features are filled with 0 and reported.

## Output
- `predictions.csv` (id, score, prediction[, y_true])
- `predict_dashboard.png / .html`
- `summary.json`

## Chaining
Model from `ds-classification` / `ds-regression` / `ds-tune`; threshold from `ds-metrics`; compare to training data with `ds-drift`.

## Response Rules (token-friendly)
- Never print raw DataFrames. Read `summary.json` (or the <=10 stdout lines), not the CSVs.
- Show/link the PNG dashboard (fallback: the HTML).
- Reply in **<= 4 short bullets**: rows scored & positive rate / mean prediction, live metric if labels present, schema warnings, CSV path.
- If the script fails, report the last error line and the likely fix (wrong column name, file path, missing model).
