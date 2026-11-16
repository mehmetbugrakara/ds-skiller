"""Token-friendly stdout + JSON output emitter."""

import json
import sys
from pathlib import Path


def emit_result(
    skill: str,
    out_dir: str | Path,
    kpis: dict,
    visuals: list[str],
    warnings: list[str] | None = None,
    extra: dict | None = None,
) -> None:
    """Write summary.json and print a compact <=10-line summary to stdout."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "skill": skill,
        "kpis": kpis,
        "visuals": [str(v) for v in visuals if v],
        "warnings": warnings or [],
        "extra": extra or {},
    }
    json_path = out / "summary.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False), encoding="utf-8")

    kpi_str = " | ".join(f"{k}={v}" for k, v in kpis.items())
    print(f"[{skill}] OK")
    print(f"KPI: {kpi_str}")
    for v in [v for v in visuals if v]:
        print(f"VISUAL: {v}")
    print(f"JSON: {json_path}")
    for w in (warnings or [])[:3]:
        print(f"WARN: {w}")
