"""Cohort retention & LTV analysis from an event log (user, date[, value])."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ds_core import TOKENS, emit_result, html_heatmap, html_table, load_dataset, render_dashboard, set_theme, svg_barv, svg_line_chart

FREQ = {"month": ("M", "M"), "week": ("W", "W")}


def run_cohort(data_path: str, user: str, date: str, value: str | None, period: str, max_periods: int, fmt: str, out_dir: str) -> None:
    df = load_dataset(data_path)
    for c in [user, date] + ([value] if value else []):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not in dataset. Available: {list(df.columns)[:12]}")
    df = df[[user, date] + ([value] if value else [])].copy()
    df[date] = pd.to_datetime(df[date], errors="coerce")
    df = df.dropna(subset=[user, date])
    if df.empty:
        raise ValueError("No valid rows after parsing dates.")

    code = FREQ[period][0]
    df["_period"] = df[date].dt.to_period(code)
    df["_cohort"] = df.groupby(user)["_period"].transform("min")
    df["_age"] = (df["_period"] - df["_cohort"]).apply(lambda x: x.n).astype(int)
    df = df[df["_age"] <= max_periods]

    active = df.groupby(["_cohort", "_age"])[user].nunique().unstack(fill_value=0)
    sizes = active[0]
    retention = active.div(sizes, axis=0)
    # Mask cells that are in the future relative to the last observed period
    last = df["_period"].max()
    for coh in retention.index:
        horizon = (last - coh).n
        retention.loc[coh, retention.columns > horizon] = np.nan

    ret_pct = (retention * 100).round(1)
    ret_pct.index = [str(i) for i in ret_pct.index]
    ret_pct.columns = [f"{'M' if period == 'month' else 'W'}{c}" for c in ret_pct.columns]

    # Weighted average retention curve (by cohort size, only observed cells)
    curve = []
    for c in retention.columns:
        col = retention[c]
        ok = col.notna()
        curve.append(float(np.average(col[ok], weights=sizes[ok])) * 100 if ok.any() else np.nan)
    curve = pd.Series(curve, index=retention.columns).dropna()

    p1 = curve.get(1, np.nan)
    p3 = curve.get(3, np.nan)
    tail = curve.iloc[-1] if len(curve) else np.nan

    kpis_out = {
        "users": int(df[user].nunique()),
        "cohorts": int(len(sizes)),
        f"{period}1_retention_pct": round(float(p1), 2) if not np.isnan(p1) else None,
        f"{period}3_retention_pct": round(float(p3), 2) if not np.isnan(p3) else None,
    }
    lbl = "M" if period == "month" else "W"
    k_list = [
        ("Users", f"{df[user].nunique():,}", TOKENS["primary"]),
        ("Cohorts", str(len(sizes)), TOKENS["muted"]),
        (f"{lbl}1 Retention", f"{p1:.1f}%" if not np.isnan(p1) else "-", TOKENS["success"]),
        (f"{lbl}3 Retention", f"{p3:.1f}%" if not np.isnan(p3) else "-", TOKENS["warning"]),
        (f"{lbl}{curve.index[-1] if len(curve) else '-'} Tail", f"{tail:.1f}%" if not np.isnan(tail) else "-", TOKENS["danger"]),
    ]

    # Best / worst cohort at period 1
    if 1 in retention.columns and retention[1].notna().sum() >= 2:
        r1 = retention[1].dropna()
        best_c, worst_c = str(r1.idxmax()), str(r1.idxmin())
        kpis_out["best_cohort"], kpis_out["worst_cohort"] = best_c, worst_c

    panels = [
        {
            "title": f"Retention Matrix (% active, by {period} since first activity)",
            "badge": f"{len(sizes)} cohorts",
            "html": html_heatmap(ret_pct.iloc[-12:, : min(ret_pct.shape[1], 10)].fillna(0), fmt_str=".0f", diverging=False),
            "span": 2,
        },
        {
            "title": "Average Retention Curve (size-weighted)",
            "badge": "%",
            "html": svg_line_chart(
                [{"name": "Retention %", "x": [int(i) for i in curve.index], "y": curve.round(2).tolist(), "color": TOKENS["primary"], "fill": True}],
                x_label=f"{period.title()}s since first activity",
            ),
        },
        {
            "title": "Cohort Sizes (new users)",
            "badge": "Count",
            "html": svg_barv([str(i) for i in sizes.index[-12:]], sizes.values[-12:].astype(float).tolist()),
        },
    ]

    warnings = []
    if value:
        df[value] = pd.to_numeric(df[value], errors="coerce").fillna(0)
        rev = df.groupby(["_cohort", "_age"])[value].sum().unstack(fill_value=0).cumsum(axis=1)
        ltv = rev.div(sizes, axis=0)
        ltv_curve = []
        for c in ltv.columns:
            ok = retention[c].notna() if c in retention.columns else pd.Series(False, index=ltv.index)
            ltv_curve.append(float(np.average(ltv.loc[ok, c], weights=sizes[ok])) if ok.any() else np.nan)
        ltv_curve = pd.Series(ltv_curve, index=ltv.columns).dropna()
        kpis_out["ltv_last"] = round(float(ltv_curve.iloc[-1]), 2) if len(ltv_curve) else None
        k_list.append(("Cum. LTV / User", f"{ltv_curve.iloc[-1]:,.1f}" if len(ltv_curve) else "-", TOKENS["purple"]))
        panels.append({
            "title": f"Cumulative {value} per User (LTV curve)",
            "badge": "size-weighted",
            "html": svg_line_chart(
                [{"name": "LTV", "x": [int(i) for i in ltv_curve.index], "y": ltv_curve.round(2).tolist(), "color": TOKENS["purple"]}],
                x_label=f"{period.title()}s since first activity",
            ),
        })
    if len(sizes) < 3:
        warnings.append("Fewer than 3 cohorts: retention comparison is weak.")
    if not np.isnan(p1) and p1 < 20:
        warnings.append(f"{lbl}1 retention {p1:.1f}% is low: check onboarding / activation.")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ret_pct.to_csv(out / "retention_matrix.csv")
    summary_tbl = pd.DataFrame({"Cohort": [str(i) for i in sizes.index], "Users": sizes.values})
    for c in [1, 3, 6]:
        if c in retention.columns:
            summary_tbl[f"{lbl}{c}_%"] = (retention[c].values * 100).round(1)
    panels.append({"title": "Latest Cohorts", "badge": "retention_matrix.csv", "html": html_table(summary_tbl.tail(8).fillna("-")), "span": 2})

    png_path, html_path = render_dashboard(
        title="Cohort Retention & LTV Analysis",
        subtitle=f"Dataset: {Path(data_path).name} | User: {user} | Date: {date} | Period: {period}",
        kpis=k_list,
        panels=panels,
        out_dir=out_dir,
        file_stem="cohort_dashboard",
        fmt=fmt,
    )
    kpis_out["csv"] = str(out / "retention_matrix.csv")
    emit_result(skill="ds-cohort", out_dir=out_dir, kpis=kpis_out, visuals=[png_path, html_path], warnings=warnings)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Cohort retention & LTV from an event log.")
    p.add_argument("--data", required=True)
    p.add_argument("--user", required=True, help="User / customer id column")
    p.add_argument("--date", required=True, help="Event date column")
    p.add_argument("--value", default=None, help="Optional revenue column for LTV")
    p.add_argument("--period", default="month", choices=["month", "week"])
    p.add_argument("--max-periods", type=int, default=12)
    p.add_argument("--format", default="landscape", choices=["landscape", "portrait"])
    p.add_argument("--theme", default=None, choices=["light", "dark"])
    p.add_argument("--out", default="outputs/ds-cohort")
    args = p.parse_args()
    set_theme(args.theme)
    run_cohort(args.data, args.user, args.date, args.value, args.period, args.max_periods, args.format, args.out)
