# ds-skiller Agent Rules (Token-Friendly & Visual-First)

1. **Always execute the skill's bundled Python script** (`uv run python skills/<skill>/scripts/run.py ...` inside this repo; installed skills use `uv run --project <repo> ...`) instead of writing ad-hoc pandas/matplotlib code.
2. **Never dump raw DataFrames or long tables to stdout.** Scripts emit `summary.json` + a <=10-line KPI summary + HTML/PNG dashboards.
3. **Reply concisely (<= 5 bullet points)**:
   - Link/display the generated PNG dashboard (`outputs/<skill>/*.png`).
   - State key KPIs and actionable warnings only.
4. **Reels / Vertical Video Mode**: pass `--format portrait` for 9:16 dashboards (Instagram Reels / YouTube Shorts). `--theme dark` for the graphite theme.
5. In `SKILL.md` files, `{{DS_SKILLER_ROOT}}` means this repo root (`.`); the installer replaces it with the absolute path.
6. Development: after changing a skill run `uv run pytest -q`; after adding/renaming a skill re-run `install.ps1` / `install.sh`.
