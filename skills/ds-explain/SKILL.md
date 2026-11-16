---
name: ds-explain
description: SHAP explainability for a saved model bundle: global mean |SHAP|, directional beeswarm-style impact, top feature dependence and a local driver breakdown. Use for /ds-explain, /shap, 'model neden böyle tahmin etti', 'feature etkileri'.
argument-hint: [--model outputs/ds-classification/best_model.joblib] [--data outputs/ds-classification/test_predictions.csv] [--max-samples 300] [--format landscape|portrait] [--theme light|dark] [--out <dir>]
---

# /ds-explain — Model Explainability (SHAP)

Run the bundled script. Do not write ad-hoc pandas/matplotlib code for this task.

```bash
uv run --project "{{DS_SKILLER_ROOT}}" python "{{DS_SKILLER_ROOT}}/skills/ds-explain/scripts/run.py" --model "<model_joblib>" --data "<test_csv>" [--format landscape|portrait] [--theme light|dark]
```

- The absolute repo path above is written by the installer. If you see a raw placeholder, you are inside the ds-skiller repo: use `.`.
- Data paths and `--out` are relative to the current working directory. Default output: `outputs/ds-explain/`.
- `--format portrait` = 9:16 Reels/Shorts PNG. `--theme dark` = graphite theme.
- PNG needs Chrome or Edge (override path with `DS_SKILLER_BROWSER`); without it only the HTML is written.

## Input
A ds-skiller model bundle and a CSV containing the model's feature columns (test_predictions.csv works).

## Output
- `shap_dashboard.png / .html`
- `summary.json`

## Chaining
Runs after `ds-classification` / `ds-regression` / `ds-tune`.

## Response Rules (token-friendly)
- Never print raw DataFrames. Read `summary.json` (or the <=10 stdout lines), not the CSVs.
- Show/link the PNG dashboard (fallback: the HTML).
- Reply in **<= 3 short bullets**: top 3 global drivers and their direction of effect.
- If the script fails, report the last error line and the likely fix (wrong column name, file path, missing model).
