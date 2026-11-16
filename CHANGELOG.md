# Changelog

## 0.2.0 — 2026-11-16

### Added
- `ds-predict`: batch scoring with saved model bundles, schema alignment, live evaluation when labels exist.
- `ds-pipeline`: one-command EDA → Clean → Features → Model → SHAP → Metrics with an index dashboard.
- `ds-cohort`: retention matrix, retention curve, cohort sizes, LTV per user.
- `ds-uplift`: T-learner uplift modeling, Qini curve, decile uplift, per-row uplift scores.
- `ds-text`: Turkish/English text profiling, top terms/bigrams, TF-IDF baseline and discriminative terms.
- `--theme light|dark` flag on every skill.
- `DS_SKILLER_BROWSER` override and a warning when no browser is found for PNG export.
- `ds_core.prepare_inference_features` for consistent encoding at prediction time.
- Sample data for cohort, uplift and text; pytest smoke suite; MIT license.

### Changed
- Installers render `SKILL.md` with the repo's absolute path instead of copying scripts, so installed skills work from any folder and `git pull` updates all agents. Added uninstall and custom targets.
- Skills are invoked as `/ds-<name>` (the slash command equals the skill name).
- `ds_core` is now an installable package (`uv sync`).
- All `SKILL.md` files rewritten with input, output, chaining and error-handling sections.
- stdout and `summary.json` are UTF-8 (Turkish text renders correctly on Windows).

## 0.1.0

- Initial 14 skills, `ds_core` dashboard engine, installers.
