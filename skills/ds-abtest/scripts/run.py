"""A/B Test & Hypothesis Testing Impeccable HTML/PNG dashboard."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ds_core import set_theme, TOKENS, emit_result, html_table, load_dataset, render_dashboard, svg_barv, svg_line_chart


def run_abtest(
    data_path: str,
    group_col: str,
    metric_col: str,
    control: str | None,
    variant: str | None,
    fmt: str,
    out_dir: str,
) -> None:
    df = load_dataset(data_path).dropna(subset=[group_col, metric_col]).copy()
    groups = df[group_col].astype(str).unique().tolist()
    if len(groups) < 2:
        raise ValueError(f"Group column '{group_col}' needs at least 2 groups.")
    ctrl_name = control or sorted(groups)[0]
    var_name = variant or [g for g in sorted(groups) if g != ctrl_name][0]

    a = pd.to_numeric(df.loc[df[group_col].astype(str) == ctrl_name, metric_col], errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(df.loc[df[group_col].astype(str) == var_name, metric_col], errors="coerce").dropna().to_numpy()

    n_a, n_b = len(a), len(b)
    _, srm_p = stats.chisquare([n_a, n_b], f_exp=[(n_a + n_b) / 2, (n_a + n_b) / 2])
    srm_ok = srm_p >= 0.001

    mean_a, mean_b = float(a.mean()), float(b.mean())
    abs_diff = mean_b - mean_a
    rel_lift = (abs_diff / mean_a * 100) if mean_a != 0 else 0.0

    is_binary = set(np.unique(np.concatenate([a, b]))).issubset({0, 1})
    if is_binary:
        _, p_val = proportions_ztest([b.sum(), a.sum()], [n_b, n_a])
        se = np.sqrt(mean_a * (1 - mean_a) / n_a + mean_b * (1 - mean_b) / n_b)
        ci_low, ci_high = abs_diff - 1.96 * se, abs_diff + 1.96 * se
        eff = proportion_effectsize(mean_b, mean_a)
        power = float(NormalIndPower().solve_power(effect_size=abs(eff), nobs1=n_a, ratio=n_b / n_a, alpha=0.05))
        test_name = "Two-Prop Z-Test"
    else:
        _, p_val = stats.ttest_ind(b, a, equal_var=False)
        _, mw_p = stats.mannwhitneyu(b, a, alternative="two-sided")
        se = np.sqrt(a.var(ddof=1) / n_a + b.var(ddof=1) / n_b)
        ci_low, ci_high = abs_diff - 1.96 * se, abs_diff + 1.96 * se
        pooled_sd = np.sqrt(((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1)) / max(n_a + n_b - 2, 1))
        d = abs_diff / max(pooled_sd, 1e-9)
        power = float(TTestIndPower().solve_power(effect_size=abs(d), nobs1=n_a, ratio=n_b / n_a, alpha=0.05))
        test_name = f"Welch t (MW p={mw_p:.4f})"

    if not srm_ok:
        decision, dec_color = "INVALID (SRM)", TOKENS["danger"]
    elif p_val < 0.05 and abs_diff > 0:
        decision, dec_color = "SHIP VARIANT", TOKENS["success"]
    elif p_val < 0.05 and abs_diff < 0:
        decision, dec_color = "KEEP CONTROL", TOKENS["danger"]
    else:
        decision, dec_color = "INCONCLUSIVE", TOKENS["warning"]

    bar_svg = svg_barv([ctrl_name, var_name], [mean_a, mean_b], colors=[TOKENS["muted"], dec_color], fmt_str="{:.4f}")

    # Lift Normal Distribution Curve
    xs = np.linspace(abs_diff - 4 * max(se, 1e-6), abs_diff + 4 * max(se, 1e-6), 80)
    ys = stats.norm.pdf(xs, loc=abs_diff, scale=max(se, 1e-6))
    dist_svg = svg_line_chart(
        [{"name": f"Lift PDF (95% CI [{ci_low:+.4f}, {ci_high:+.4f}])", "x": xs.tolist(), "y": ys.tolist(), "color": dec_color, "fill": True}],
        x_label=f"Absolute Lift ({var_name} - {ctrl_name})",
    )

    tbl_df = pd.DataFrame([
        {"Group": ctrl_name, "Samples": f"{n_a:,}", "Mean": f"{mean_a:.4f}", "Rel_Lift": "-", "CI_95": "-", "p_val": "-", "Status": "OK"},
        {"Group": var_name, "Samples": f"{n_b:,}", "Mean": f"{mean_b:.4f}", "Rel_Lift": f"{rel_lift:+.2f}%", "CI_95": f"[{ci_low:+.4f}, {ci_high:+.4f}]", "p_val": f"{p_val:.4f}", "Status": "SHIP" if decision == "SHIP VARIANT" else "WARN"},
    ])

    kpis = [
        ("Decision", decision, dec_color),
        ("Relative Lift", f"{rel_lift:+.2f}%", TOKENS["success"] if rel_lift > 0 else TOKENS["danger"]),
        ("p-value", f"{p_val:.4f}", TOKENS["success"] if p_val < 0.05 else TOKENS["warning"]),
        ("Stat Power", f"{power * 100:.1f}%", TOKENS["primary"]),
        ("SRM Check", f"PASS (p={srm_p:.3f})" if srm_ok else f"FAIL (p={srm_p:.4f})", TOKENS["success"] if srm_ok else TOKENS["danger"]),
    ]

    panels = [
        {"title": f"Group Mean Comparison ({metric_col})", "badge": test_name, "html": bar_svg},
        {"title": "Lift Sampling Distribution & 95% CI", "badge": f"Diff={abs_diff:+.4f}", "html": dist_svg},
        {"title": "Experiment Results Summary", "badge": f"N={n_a + n_b:,}", "html": html_table(tbl_df, highlight_row=1 if decision == "SHIP VARIANT" else None, status_col="Status"), "span": 2},
    ]

    png_path, html_path = render_dashboard(
        title="A/B Test & Hypothesis Testing Decision Card",
        subtitle=f"{ctrl_name} (n={n_a:,}) vs {var_name} (n={n_b:,}) | Metric: {metric_col}",
        kpis=kpis,
        panels=panels,
        out_dir=out_dir,
        file_stem="abtest_dashboard",
        fmt=fmt,
    )
    warnings = ["SRM failed: check randomization pipeline!"] if not srm_ok else []
    if power < 0.8:
        warnings.append(f"Underpowered ({power * 100:.1f}% < 80%): collect more samples.")
    emit_result(
        skill="ds-abtest",
        out_dir=out_dir,
        kpis={"decision": decision, "rel_lift_pct": round(rel_lift, 2), "p_val": round(float(p_val), 4), "power_pct": round(power * 100, 1)},
        visuals=[png_path, html_path],
        warnings=warnings,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--group", required=True)
    p.add_argument("--metric", required=True)
    p.add_argument("--control", default=None)
    p.add_argument("--variant", default=None)
    p.add_argument("--format", default="landscape", choices=["landscape", "portrait"])
    p.add_argument("--theme", default=None, choices=["light", "dark"])
    p.add_argument("--out", default="outputs/ds-abtest")
    args = p.parse_args()
    set_theme(args.theme)
    run_abtest(args.data, args.group, args.metric, args.control, args.variant, args.format, args.out)
