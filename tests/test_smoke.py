"""End-to-end smoke tests: every skill runs on sample data and writes a valid summary.json + HTML."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data"


@pytest.fixture(scope="session", autouse=True)
def sample_data():
    if not (DATA / "reviews_tr.csv").exists():
        subprocess.run([sys.executable, str(ROOT / "examples" / "generate_samples.py")], check=True)


@pytest.fixture(scope="session")
def out_root(tmp_path_factory):
    return tmp_path_factory.mktemp("outputs")


def run_skill(skill: str, args: list[str], out: Path) -> dict:
    cmd = [sys.executable, str(ROOT / "skills" / skill / "scripts" / "run.py"), *args, "--out", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT)
    assert proc.returncode == 0, f"{skill} failed:\n{proc.stderr[-2000:]}"
    assert len(proc.stdout.strip().splitlines()) <= 10, "stdout must stay <= 10 lines"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["skill"] == skill
    assert any(v.endswith(".html") and Path(v).exists() for v in summary["visuals"])
    return summary


D = lambda name: str(DATA / name)  # noqa: E731

SIMPLE = {
    "ds-eda": ["--data", D("churn_sample.csv"), "--target", "churn"],
    "ds-clean": ["--data", D("churn_sample.csv"), "--target", "churn"],
    "ds-features": ["--data", D("churn_sample.csv"), "--target", "churn"],
    "ds-anomaly": ["--data", D("churn_sample.csv"), "--target", "churn"],
    "ds-dimreduce": ["--data", D("churn_sample.csv"), "--target", "churn"],
    "ds-drift": ["--ref", D("churn_sample.csv"), "--curr", D("churn_current_drifted.csv"), "--target", "churn"],
    "ds-abtest": ["--data", D("abtest_checkout.csv"), "--group", "group", "--metric", "converted"],
    "ds-timeseries": ["--data", D("daily_orders_ts.csv"), "--date", "date", "--target", "daily_orders"],
    "ds-cohort": ["--data", D("user_events.csv"), "--user", "user_id", "--date", "event_date", "--value", "revenue"],
    "ds-uplift": ["--data", D("campaign_uplift.csv"), "--treatment", "treatment", "--outcome", "converted"],
    "ds-text": ["--data", D("reviews_tr.csv"), "--text", "review", "--target", "sentiment"],
}


@pytest.mark.parametrize("skill", sorted(SIMPLE))
def test_standalone_skill(skill, out_root):
    run_skill(skill, SIMPLE[skill], out_root / skill)


@pytest.mark.slow
def test_classification_chain(out_root):
    clf = out_root / "clf"
    s = run_skill("ds-classification", ["--data", D("churn_sample.csv"), "--target", "churn", "--cv", "3"], clf)
    assert s["kpis"]["roc_auc"] > 0.6
    model, preds = str(clf / "best_model.joblib"), str(clf / "test_predictions.csv")
    run_skill("ds-explain", ["--model", model, "--data", preds, "--max-samples", "100"], out_root / "explain")
    run_skill("ds-metrics", ["--preds", preds], out_root / "metrics")
    run_skill("ds-audit", ["--data", D("churn_sample.csv"), "--model", model, "--slice", "contract_type"], out_root / "audit")
    p = run_skill("ds-predict", ["--model", model, "--data", D("churn_current_drifted.csv"), "--id", "customer_id"], out_root / "predict")
    assert p["kpis"]["rows"] == 800 and "roc_auc" in p["kpis"]


@pytest.mark.slow
def test_regression_and_predict(out_root):
    reg = out_root / "reg"
    s = run_skill("ds-regression", ["--data", D("housing_regression.csv"), "--target", "price_usd", "--cv", "3"], reg)
    assert s["kpis"]
    p = run_skill("ds-predict", ["--model", str(reg / "best_model.joblib"), "--data", D("housing_regression.csv")], out_root / "predict_reg")
    assert p["kpis"]["r2"] > 0.8


@pytest.mark.slow
def test_tune(out_root):
    run_skill("ds-tune", ["--data", D("housing_regression.csv"), "--target", "price_usd", "--algo", "lightgbm", "--trials", "3", "--cv", "3"], out_root / "tune")


@pytest.mark.slow
def test_pipeline(out_root):
    s = run_skill("ds-pipeline", ["--data", D("churn_sample.csv"), "--target", "churn", "--cv", "3", "--skip", "ds-explain"], out_root / "pipeline")
    assert s["kpis"]["steps_ok"] == "5/5"


def test_every_skill_is_covered_and_documented():
    skills = sorted(p.parent.parent.name for p in (ROOT / "skills").glob("ds-*/scripts/run.py"))
    covered = set(SIMPLE) | {"ds-classification", "ds-explain", "ds-metrics", "ds-audit", "ds-predict", "ds-regression", "ds-tune", "ds-pipeline"}
    assert set(skills) == covered, f"untested skills: {set(skills) - covered}"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for s in skills:
        md = (ROOT / "skills" / s / "SKILL.md").read_text(encoding="utf-8")
        assert re.search(rf"^name: {s}$", md, re.M), f"{s}: frontmatter name mismatch"
        assert "{{DS_SKILLER_ROOT}}" in md, f"{s}: SKILL.md must use the installer placeholder"
        assert f"`/{s}`" in readme, f"{s}: missing from README catalog"
        script = (ROOT / "skills" / s / "scripts" / "run.py").read_text(encoding="utf-8")
        for flag in ("--format", "--theme", "--out"):
            assert flag in script, f"{s}: missing common flag {flag}"
