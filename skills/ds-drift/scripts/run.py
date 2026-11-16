"""Production Data Drift, PSI, bin-level CSI, JS-Divergence & Wasserstein dashboard."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ds_core import set_theme
from ds_core import (
    TOKENS,
    emit_result,
    html_table,
    load_dataset,
    render_dashboard,
    svg_barh,
    svg_line_chart,
)


def calc_psi_and_csi(ref: np.ndarray, curr: np.ndarray, bins: int = 10) -> tuple[float, pd.DataFrame]:
    ref = ref[~np.isnan(ref)]
    curr = curr[~np.isnan(curr)]
    if len(ref) == 0 or len(curr) == 0:
        return 0.0, pd.DataFrame()
    edges = np.unique(np.percentile(ref, np.linspace(0, 100, bins + 1)))
    if len(edges) < 3:
        return 0.0, pd.DataFrame()
    edges[0], edges[-1] = -np.inf, np.inf
    ref_cnt = np.clip(np.histogram(ref, bins=edges)[0] / len(ref), 1e-4, None)
    cur_cnt = np.clip(np.histogram(curr, bins=edges)[0] / len(curr), 1e-4, None)
    csi = (cur_cnt - ref_cnt) * np.log(cur_cnt / ref_cnt)
    psi = float(np.sum(csi))

    bin_labels = [f"[{edges[i]:.1f}, {edges[i + 1]:.1f})" if np.isfinite(edges[i]) and np.isfinite(edges[i + 1])
                  else (f"< {edges[i + 1]:.1f}" if not np.isfinite(edges[i]) else f">= {edges[i]:.1f}")
                  for i in range(len(edges) - 1)]
    csi_df = pd.DataFrame({
        "Bin": bin_labels,
        "Ref%": (ref_cnt * 100).round(1),
        "Curr%": (cur_cnt * 100).round(1),
        "Delta%": ((cur_cnt - ref_cnt) * 100).round(1),
        "CSI": csi.round(4),
    })
    return psi, csi_df


def run_drift(ref_path: str, curr_path: str, target: str | None, fmt: str, out_dir: str) -> None:
    df_ref = load_dataset(ref_path)
    df_curr = load_dataset(curr_path)
    common_cols = [c for c in df_ref.columns if c in df_curr.columns]

    rows = []
    top_num_info = None
    max_num_psi = -1.0

    for c in common_cols:
        if pd.api.types.is_numeric_dtype(df_ref[c]) and pd.api.types.is_numeric_dtype(df_curr[c]):
            r = df_ref[c].dropna().to_numpy(dtype=float)
            cu = df_curr[c].dropna().to_numpy(dtype=float)
            if len(r) < 5 or len(cu) < 5:
                continue
            psi, csi_df = calc_psi_and_csi(r, cu)
            _, p_val = ks_2samp(r, cu)
            # Normalized Wasserstein & JS Divergence
            std_r = max(float(np.std(r)), 1e-9)
            w_dist = float(wasserstein_distance(r, cu) / std_r)
            edges = np.histogram_bin_edges(np.concatenate([r, cu]), bins=15)
            p_hist = np.clip(np.histogram(r, bins=edges)[0] / len(r), 1e-6, None)
            q_hist = np.clip(np.histogram(cu, bins=edges)[0] / len(cu), 1e-6, None)
            js_div = float(jensenshannon(p_hist, q_hist))
            status = "HIGH" if psi >= 0.2 else ("MODERATE" if psi >= 0.1 else "STABLE")
            rows.append({
                "Feature": c[:20] + ("*" if c == target else ""),
                "Type": "Num",
                "PSI": round(psi, 3),
                "JS_Div": round(js_div, 3),
                "Wass_Std": round(w_dist, 2),
                "p_val": round(float(p_val), 4),
                "Status": status,
            })
            if psi > max_num_psi:
                max_num_psi = psi
                top_num_info = (c, psi, r, cu, csi_df)
        else:
            vc_r = df_ref[c].fillna("MISSING").astype(str).value_counts()
            vc_c = df_curr[c].fillna("MISSING").astype(str).value_counts()
            cats = sorted(set(vc_r.index) | set(vc_c.index))
            if len(cats) < 2 or len(cats) > 50:
                continue
            tab = np.array([[vc_r.get(k, 0) for k in cats], [vc_c.get(k, 0) for k in cats]])
            _, p_val, _, _ = chi2_contingency(tab)
            r_p = np.clip(tab[0] / max(tab[0].sum(), 1), 1e-4, None)
            c_p = np.clip(tab[1] / max(tab[1].sum(), 1), 1e-4, None)
            psi = float(np.sum((c_p - r_p) * np.log(c_p / r_p)))
            js_div = float(jensenshannon(r_p, c_p))
            status = "HIGH" if psi >= 0.2 else ("MODERATE" if psi >= 0.1 else "STABLE")
            rows.append({
                "Feature": c[:20] + ("*" if c == target else ""),
                "Type": "Cat",
                "PSI": round(psi, 3),
                "JS_Div": round(js_div, 3),
                "Wass_Std": "-",
                "p_val": round(float(p_val), 4),
                "Status": status,
            })

    drift_df = pd.DataFrame(rows).sort_values("PSI", ascending=False).reset_index(drop=True)
    n_high = int((drift_df["Status"] == "HIGH").sum()) if not drift_df.empty else 0
    n_mod = int((drift_df["Status"] == "MODERATE").sum()) if not drift_df.empty else 0
    max_psi = float(drift_df["PSI"].max()) if not drift_df.empty else 0.0
    max_js = float(drift_df["JS_Div"].max()) if not drift_df.empty else 0.0

    kpis = [
        ("Checked Features", f"{len(drift_df)}", TOKENS["primary"]),
        ("High Drift (PSI>=0.2)", f"{n_high}", TOKENS["danger"] if n_high > 0 else TOKENS["success"]),
        ("Moderate (0.1-0.2)", f"{n_mod}", TOKENS["warning"] if n_mod > 0 else TOKENS["success"]),
        ("Max PSI / JS Div", f"{max_psi:.3f} / {max_js:.3f}", TOKENS["danger"] if max_psi >= 0.2 else TOKENS["primary"]),
        ("Ref / Curr Rows", f"{len(df_ref):,} / {len(df_curr):,}", TOKENS["muted"]),
    ]

    top_d = drift_df.head(8)
    bar_colors = [
        TOKENS["danger"] if s == "HIGH" else (TOKENS["warning"] if s == "MODERATE" else TOKENS["success"])
        for s in top_d["Status"]
    ]
    psi_svg = svg_barh(
        top_d["Feature"].tolist(),
        top_d["PSI"].tolist(),
        colors=bar_colors,
        thresholds=[(0.1, TOKENS["warning"], "0.10 Mod"), (0.2, TOKENS["danger"], "0.20 High")],
    )

    panels = [
        {"title": "Feature Drift Ranking (PSI)", "badge": "0.10 Mod / 0.20 High", "html": psi_svg},
        {"title": "Drift Audit Matrix (PSI + JS + Wasserstein)", "badge": f"{len(drift_df)} cols", "html": html_table(drift_df, status_col="Status", max_rows=8)},
    ]

    if top_num_info is not None:
        feat_name, f_psi, r_arr, cu_arr, csi_df = top_num_info
        edges = np.histogram_bin_edges(np.concatenate([r_arr, cu_arr]), bins=20)
        centers = ((edges[:-1] + edges[1:]) / 2).tolist()
        r_density = (np.histogram(r_arr, bins=edges)[0] / len(r_arr) * 100).tolist()
        c_density = (np.histogram(cu_arr, bins=edges)[0] / len(cu_arr) * 100).tolist()
        shift_svg = svg_line_chart(
            [
                {"name": "Reference (%)", "x": centers, "y": r_density, "color": TOKENS["primary"], "fill": True},
                {"name": "Current (%)", "x": centers, "y": c_density, "color": TOKENS["danger"], "fill": True},
            ],
            x_label=feat_name,
        )
        panels.append({"title": f"Top Distribution Shift: {feat_name}", "badge": f"PSI={f_psi:.3f}", "html": shift_svg})
        panels.append({"title": f"Bin-Level CSI Breakdown: {feat_name}", "badge": "Characteristic Stability", "html": html_table(csi_df.sort_values("CSI", ascending=False), max_rows=7)})

    png_path, html_path = render_dashboard(
        title="Data Drift, PSI & CSI Monitoring",
        subtitle=f"Ref: {Path(ref_path).name} vs Curr: {Path(curr_path).name}",
        kpis=kpis,
        panels=panels,
        out_dir=out_dir,
        file_stem="drift_dashboard",
        fmt=fmt,
    )
    warnings = [f"Retrain recommended: {n_high} feature(s) with PSI >= 0.2"] if n_high > 0 else []
    emit_result(
        skill="ds-drift",
        out_dir=out_dir,
        kpis={"high_drift": n_high, "moderate_drift": n_mod, "max_psi": round(max_psi, 3), "max_js": round(max_js, 3)},
        visuals=[png_path, html_path],
        warnings=warnings,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("--curr", required=True)
    p.add_argument("--target", default=None)
    p.add_argument("--format", default="landscape", choices=["landscape", "portrait"])
    p.add_argument("--theme", default=None, choices=["light", "dark"])
    p.add_argument("--out", default="outputs/ds-drift")
    args = p.parse_args()
    set_theme(args.theme)
    run_drift(args.ref, args.curr, args.target, args.format, args.out)
