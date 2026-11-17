# ds-skiller

**Token-friendly, visual-first data science skills** for **Claude Code**, **Codex** and **Gemini / Antigravity**.

Every skill is a deterministic Python pipeline (`skills/<skill>/scripts/run.py`). The agent runs it instead of writing ad-hoc pandas code and gets back:

1. A self-contained **HTML dashboard** + **2x PNG** (`--format landscape` 16:9 or `--format portrait` 9:16 for Reels/Shorts, `--theme light|dark`).
2. A **<=10-line KPI summary** on stdout + `summary.json`.

The agent reads the summary, shows the PNG and answers in a few bullets, so a full analysis costs a handful of tokens.

## Skill Catalog (19)

| Command | What it produces |
|---|---|
| **Explore & prepare** | |
| `/ds-eda` | 8-panel EDA: target skew, Mutual Information, missingness, IQR/Z outliers, cardinality, correlation, hygiene/leakage audit |
| `/ds-clean` | Dedup, constant/ID drop, median/mode imputation, IQR capping, cleaned CSV + audit log |
| `/ds-features` | VIF, Mutual Information, Spearman, redundant pairs, Keep/Drop list |
| `/ds-anomaly` | Isolation Forest + LOF + Robust Z consensus outliers, PCA anomaly map, flagged CSV |
| `/ds-text` | Text column profile (TR/EN): lengths, top terms/bigrams, duplicates, TF-IDF baseline + discriminative terms |
| **Model** | |
| `/ds-classification` | Stratified CV benchmark (XGBoost, CatBoost, LightGBM, RF, LogReg), ROC, confusion matrix, saved model |
| `/ds-regression` | K-fold benchmark (XGBoost, CatBoost, LightGBM, RF, Ridge), R2/RMSE/MAE/MAPE, residuals, saved model |
| `/ds-tune` | Hyperparameter search for XGBoost/CatBoost/LightGBM + learning curve |
| `/ds-predict` | Score new data with a saved model: schema alignment, predictions.csv, live eval if labels exist |
| `/ds-pipeline` | One command: EDA → Clean → Features → Model → SHAP → Metrics + index dashboard |
| **Evaluate & monitor** | |
| `/ds-metrics` | PR-AUC, KS, Brier calibration, gains/lift, threshold table |
| `/ds-explain` | SHAP global importance, dependence, local breakdown |
| `/ds-audit` | Fairness & slice audit: demographic parity, equalized odds |
| `/ds-drift` | PSI, CSI, JS divergence, Wasserstein, KS, Chi-square |
| **Business analytics** | |
| `/ds-abtest` | SRM check, Z / Welch / Mann-Whitney, 95% CI, power, decision card |
| `/ds-uplift` | T-learner uplift, Qini curve, decile uplift, who-to-target scores |
| `/ds-cohort` | Retention matrix, retention curve, cohort sizes, LTV per user |
| `/ds-timeseries` | ADF, seasonal decomposition, backtest, Holt-Winters forecast with 95% CI |
| `/ds-dimreduce` | PCA scree, auto-K K-Means (silhouette), cluster profiles |

Typical chains:

```
ds-eda → ds-clean → ds-features → ds-classification → ds-explain / ds-metrics / ds-audit → ds-predict
ds-abtest → ds-uplift
ds-drift (training vs live) → retrain → ds-predict
```

## Install

Requirements: [uv](https://docs.astral.sh/uv/), Python 3.11+, and Chrome or Edge for PNG export (HTML is always written).

```bash
git clone https://github.com/mehmetbugrakara/ds-skiller.git && cd ds-skiller
uv sync
```

Install into your agents (Windows):

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1
```

macOS / Linux:

```bash
./install.sh
```

The installer writes each `SKILL.md` into `~/.claude/skills/` (plus `~/.codex/skills/` and `~/.gemini/antigravity/skills/` if those agents are installed) with this repo's absolute path filled in. The scripts are **not copied**: they always run from this repo, so `git pull` updates every agent. Re-run the installer if you move the repo.

- Custom target: `install.ps1 -Targets "D:\other\skills"` or `DS_TARGETS="/path/skills" ./install.sh`
- Remove: `install.ps1 -Uninstall` or `./install.sh --uninstall`

Once installed, the skills work from **any** project folder: data paths and `outputs/` are relative to where you call them.

## Quickstart (inside the repo)

```bash
uv run python examples/generate_samples.py
uv run python skills/ds-pipeline/scripts/run.py --data examples/data/churn_sample.csv --target churn
uv run python skills/ds-predict/scripts/run.py --data examples/data/churn_current_drifted.csv --id customer_id
uv run python skills/ds-eda/scripts/run.py --data examples/data/churn_sample.csv --target churn --format portrait --theme dark
```

Sample data (`examples/data/`, generated, git-ignored):

| File | Use with |
|---|---|
| `churn_sample.csv` | eda, clean, features, classification, explain, metrics, audit (`--slice contract_type`), anomaly, dimreduce, pipeline |
| `churn_current_drifted.csv` | drift (`--curr`), predict |
| `housing_regression.csv` | regression, tune |
| `abtest_checkout.csv` | abtest (`--group group --metric converted`) |
| `campaign_uplift.csv` | uplift (`--treatment treatment --outcome converted`) |
| `user_events.csv` | cohort (`--user user_id --date event_date --value revenue`) |
| `reviews_tr.csv` | text (`--text review --target sentiment`) |
| `daily_orders_ts.csv` | timeseries (`--date date --target daily_orders`) |

## Common flags

| Flag | Meaning |
|---|---|
| `--format landscape\|portrait` | 16:9 dashboard or 9:16 Reels/Shorts |
| `--theme light\|dark` | Warm paper or graphite theme (also `DS_SKILLER_THEME`) |
| `--out <dir>` | Output folder (default `outputs/<skill>/`) |

Environment: `DS_SKILLER_BROWSER` = path to a Chromium browser if Chrome/Edge are not auto-detected.

## Output contract

Every run writes `<out>/summary.json`:

```json
{"skill": "ds-predict", "kpis": {"rows": 800, "roc_auc": 0.81}, "visuals": ["...png", "...html"], "warnings": [], "extra": {}}
```

and prints at most ~10 lines (`[skill] OK`, `KPI:`, `VISUAL:`, `JSON:`, up to 3 `WARN:`). Model skills save a joblib bundle `{"model", "features", "target", "label_encoder"?}` that `ds-predict`, `ds-explain` and `ds-audit` consume.

## Project layout

```
ds_core/            shared IO, feature prep, HTML/SVG dashboard engine, output emitter
  assets/fonts/     Geist, Geist Mono, Instrument Serif (embedded into HTML, works offline)
skills/ds-*/        SKILL.md (agent instructions) + scripts/run.py
examples/           sample-data generator
tests/              smoke tests (pytest)
install.ps1/.sh     agent installers
```

## Adding a skill

1. `skills/ds-<name>/scripts/run.py`: argparse with `--format`, `--theme`, `--out`; call `set_theme`, `render_dashboard`, `emit_result`.
2. `skills/ds-<name>/SKILL.md`: copy an existing one (keep the `{{DS_SKILLER_ROOT}}` command form).
3. Add a smoke case to `tests/test_smoke.py`, run `uv run pytest`, then re-run the installer.

## Development

```bash
uv sync
uv run python examples/generate_samples.py
uv run pytest -q          # runs every skill end-to-end on sample data (~2-4 min)
uv run pytest -q -m "not slow"
```

## License

MIT, see [LICENSE](LICENSE).
