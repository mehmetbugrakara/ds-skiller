"""End-to-end orchestrator: EDA -> Clean -> Features -> Model -> SHAP -> Metrics, one index dashboard.

Each step runs the skill's own run.py in a subprocess (same Python), reads its summary.json,
and the pipeline renders an index dashboard with the KPIs of every step.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ds_core import TOKENS, emit_result, html_table, load_dataset, render_dashboard, set_theme

SKILLS = ROOT / "skills"


def _detect_task(df: pd.DataFrame, target: str) -> str:
    y = df[target].dropna()
    if not pd.api.types.is_numeric_dtype(y) or y.nunique() <= min(15, max(2, len(y) // 50)):
        return "classification"
    return "regression"


def _run(skill: str, args: list[str], out_root: Path, fmt: str, theme: str | None) -> dict:
    out = out_root / skill
    cmd = [sys.executable, str(SKILLS / skill / "scripts" / "run.py"), *args, "--format", fmt, "--out", str(out)]
    if theme:
        cmd += ["--theme", theme]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    secs = round(time.time() - t0, 1)
    summ = out / "summary.json"
    if proc.returncode != 0 or not summ.exists():
        err = (proc.stderr or proc.stdout).strip().splitlines()
        return {"skill": skill, "status": "FAILED", "secs": secs, "error": err[-1] if err else "unknown", "out": str(out)}
    data = json.loads(summ.read_text(encoding="utf-8"))
    return {"skill": skill, "status": "OK", "secs": secs, "kpis": data.get("kpis", {}), "warnings": data.get("warnings", []),
            "visuals": data.get("visuals", []), "out": str(out)}


def _kpi_text(k: dict, n: int = 3) -> str:
    items = [(a, b) for a, b in k.items() if not isinstance(b, (list, dict)) and not str(b).endswith((".csv", ".joblib"))]
    return " · ".join(f"{a}={b}" for a, b in items[:n])


def run_pipeline(data_path: str, target: str, task: str, skip: list[str], cv: int, fmt: str, theme: str | None, out_dir: str) -> None:
    df = load_dataset(data_path)
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in dataset. Available: {list(df.columns)[:12]}")
    task = _detect_task(df, target) if task == "auto" else task
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    steps: list[dict] = []

    def step(skill: str, args: list[str]) -> dict:
        if skill in skip:
            r = {"skill": skill, "status": "SKIPPED", "secs": 0.0, "out": ""}
        else:
            print(f"... {skill}", file=sys.stderr, flush=True)
            r = _run(skill, args, out_root, fmt, theme)
        steps.append(r)
        return r

    step("ds-eda", ["--data", data_path, "--target", target])
    clean = step("ds-clean", ["--data", data_path, "--target", target])
    data_used = data_path
    if clean["status"] == "OK":
        cleaned = sorted(Path(clean["out"]).glob("*_cleaned.*"))
        if cleaned:
            data_used = str(cleaned[0])
    step("ds-features", ["--data", data_used, "--target", target])

    model_skill = "ds-classification" if task == "classification" else "ds-regression"
    model = step(model_skill, ["--data", data_used, "--target", target, "--cv", str(cv)])
    if model["status"] == "OK":
        mdir = Path(model["out"])
        step("ds-explain", ["--model", str(mdir / "best_model.joblib"), "--data", str(mdir / "test_predictions.csv")])
        if task == "classification":
            preds = pd.read_csv(mdir / "test_predictions.csv", nrows=5)
            if "y_prob" in preds.columns:
                step("ds-metrics", ["--preds", str(mdir / "test_predictions.csv")])

    ok = sum(s["status"] == "OK" for s in steps)
    ran = sum(s["status"] != "SKIPPED" for s in steps)
    failed = [s for s in steps if s["status"] == "FAILED"]
    total_secs = sum(s["secs"] for s in steps)
    model_k = model.get("kpis", {}) if model["status"] == "OK" else {}
    headline = ("roc_auc", "ROC-AUC") if task == "classification" else ("r2", "Test R²")

    tbl = pd.DataFrame([
        {"Step": s["skill"], "Status": s["status"], "Secs": s["secs"],
         "Key_KPIs": _kpi_text(s.get("kpis", {})) if s["status"] == "OK" else s.get("error", "")[:70]}
        for s in steps
    ])
    warn_rows = [{"Step": s["skill"], "Warning": w[:110]} for s in steps for w in s.get("warnings", [])[:2]]
    links = [{"Step": s["skill"], "Dashboard": Path(v).relative_to(out_root).as_posix() if Path(v).is_relative_to(out_root) else v}
             for s in steps for v in s.get("visuals", []) if str(v).endswith(".html")]

    k_list = [
        ("Task", task.title(), TOKENS["primary"]),
        ("Steps OK", f"{ok}/{ran}", TOKENS["success"] if not failed else TOKENS["danger"]),
        ("Best Model", str(model_k.get("best", "-")), TOKENS["purple"]),
        (headline[1], f"{model_k.get(headline[0], '-')}", TOKENS["teal"]),
        ("Runtime", f"{total_secs:.0f}s", TOKENS["muted"]),
    ]
    panels = [
        {"title": "Pipeline Steps", "badge": f"{len(steps)} steps", "html": html_table(tbl, status_col="Status", max_rows=12), "span": 2},
        {"title": "Warnings Across Steps", "badge": f"{len(warn_rows)}", "html": html_table(pd.DataFrame(warn_rows), max_rows=8) if warn_rows else '<div class="empty">No warnings</div>'},
        {"title": "Step Dashboards", "badge": "open the HTML files", "html": html_table(pd.DataFrame(links), max_rows=8) if links else '<div class="empty">None</div>'},
    ]
    png_path, html_path = render_dashboard(
        title="End-to-End DS Pipeline (Report Index)",
        subtitle=f"Dataset: {Path(data_path).name} | Target: {target} | EDA → Clean → Features → Model → SHAP → Metrics",
        kpis=k_list,
        panels=panels,
        out_dir=out_dir,
        file_stem="pipeline_dashboard",
        fmt=fmt,
    )
    emit_result(
        skill="ds-pipeline",
        out_dir=out_dir,
        kpis={"task": task, "steps_ok": f"{ok}/{ran}", "best": model_k.get("best"), headline[0]: model_k.get(headline[0]), "model": model_k.get("model")},
        visuals=[png_path, html_path],
        warnings=[f"{s['skill']} failed: {s.get('error', '')}" for s in failed] + [w["Warning"] for w in warn_rows][:3],
        extra={"steps": steps},
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run the full ds-skiller pipeline on one dataset.")
    p.add_argument("--data", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--task", default="auto", choices=["auto", "classification", "regression"])
    p.add_argument("--skip", nargs="*", default=[], help="Skill names to skip, e.g. ds-explain ds-features")
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--format", default="landscape", choices=["landscape", "portrait"])
    p.add_argument("--theme", default=None, choices=["light", "dark"])
    p.add_argument("--out", default="outputs/ds-pipeline")
    args = p.parse_args()
    set_theme(args.theme)
    run_pipeline(args.data, args.target, args.task, args.skip, args.cv, args.format, args.theme, args.out)
