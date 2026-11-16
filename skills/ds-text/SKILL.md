---
name: ds-text
description: Free-text column profiling (Turkish + English): length distribution, top terms and bigrams, duplicates, and with a label column a TF-IDF LogReg baseline plus most discriminative terms. Use for /ds-text, 'yorum analizi', 'metin kolonu', 'müşteri yorumları', 'NLP ön analiz'.
argument-hint: --data <path> --text <text_col> [--target <label_col>] [--top 10] [--format landscape|portrait] [--theme light|dark] [--out <dir>]
---

# /ds-text — Text Profiling

Run the bundled script. Do not write ad-hoc pandas/matplotlib code for this task.

```bash
uv run --project "{{DS_SKILLER_ROOT}}" python "{{DS_SKILLER_ROOT}}/skills/ds-text/scripts/run.py" --data "<data_path>" --text "<text_col>" [--target "<label_col>"] [--top 10] [--format landscape|portrait] [--theme light|dark]
```

- The absolute repo path above is written by the installer. If you see a raw placeholder, you are inside the ds-skiller repo: use `.`.
- Data paths and `--out` are relative to the current working directory. Default output: `outputs/ds-text/`.
- `--format portrait` = 9:16 Reels/Shorts PNG. `--theme dark` = graphite theme.
- PNG needs Chrome or Edge (override path with `DS_SKILLER_BROWSER`); without it only the HTML is written.

## Input
Tabular file with a free-text column; optional label column (2-20 classes).

## Output
- `term_frequencies.csv`
- `text_dashboard.png / .html`
- `summary.json`

## Chaining
Dedupe with `ds-clean`; strong baseline F1 means text features are worth adding to models.

## Response Rules (token-friendly)
- Never print raw DataFrames. Read `summary.json` (or the <=10 stdout lines), not the CSVs.
- Show/link the PNG dashboard (fallback: the HTML).
- Reply in **<= 4 short bullets**: text count & median length, top terms, duplicate rate, baseline F1 and key discriminative terms (if --target).
- If the script fails, report the last error line and the likely fix (wrong column name, file path, missing model).
