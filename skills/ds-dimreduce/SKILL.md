---
name: ds-dimreduce
description: PCA dimensionality reduction and K-Means segmentation (auto K via silhouette, scree plot, 2D cluster map, cluster profile heatmap). Use for /ds-dimreduce, /pca, /cluster, 'müşteri segmentasyonu', 'kümeleme'.
argument-hint: --data <path> [--target <col_to_exclude>] [--k <clusters>] [--format landscape|portrait] [--theme light|dark] [--out <dir>]
---

# /ds-dimreduce — PCA & Clustering

Run the bundled script. Do not write ad-hoc pandas/matplotlib code for this task.

```bash
uv run --project "{{DS_SKILLER_ROOT}}" python "{{DS_SKILLER_ROOT}}/skills/ds-dimreduce/scripts/run.py" --data "<data_path>" [--target "<col>"] [--k <n>] [--format landscape|portrait] [--theme light|dark]
```

- The absolute repo path above is written by the installer. If you see a raw placeholder, you are inside the ds-skiller repo: use `.`.
- Data paths and `--out` are relative to the current working directory. Default output: `outputs/ds-dimreduce/`.
- `--format portrait` = 9:16 Reels/Shorts PNG. `--theme dark` = graphite theme.
- PNG needs Chrome or Edge (override path with `DS_SKILLER_BROWSER`); without it only the HTML is written.

## Input
Tabular file with numeric columns.

## Output
- `pca_cluster_dashboard.png / .html`
- `summary.json`

## Chaining
Use cluster profiles as features or as `ds-audit --slice`.

## Response Rules (token-friendly)
- Never print raw DataFrames. Read `summary.json` (or the <=10 stdout lines), not the CSVs.
- Show/link the PNG dashboard (fallback: the HTML).
- Reply in **<= 3 short bullets**: components for 90% variance, chosen K & silhouette, key cluster differentiators.
- If the script fails, report the last error line and the likely fix (wrong column name, file path, missing model).
