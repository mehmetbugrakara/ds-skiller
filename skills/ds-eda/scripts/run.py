"""Comprehensive 8-panel Senior Data Scientist Exploratory Data Analysis (EDA) dashboard."""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ds_core import set_theme
from ds_core import (
    TOKENS,
    emit_result,
    html_heatmap,
    html_table,
    load_dataset,
    prepare_features,
    render_dashboard,
    split_feature_types,
    svg_barh,
    svg_barv,
)


def _shannon_entropy(series: pd.Series) -> float:
    probs = series.value_counts(normalize=True, dropna=True).to_numpy()
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log2(probs)).sum())


def run_eda(data_path: str, target: str | None, fmt: str, out_dir: str) -> None:
    df = load_dataset(data_path)
    n_rows, n_cols = df.shape
    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    dup_rows = int(df.duplicated().sum())
    dup_pct = dup_rows / max(n_rows, 1) * 100.0
    miss_pct = float(df.isna().mean().mean() * 100)
    types = split_feature_types(df, target=target)
    num_cols = types["numeric"]
    cat_cols = types["categorical"]
    dt_cols = types["datetime"]

    warnings: list[str] = []
    if dup_rows > 0:
        warnings.append(f"{dup_rows} duplicate rows ({dup_pct:.1f}%)")
    if types["constant"]:
        warnings.append(f"Constant cols: {', '.join(types['constant'][:5])}")
    if types["id_like"]:
        warnings.append(f"ID-like cols (drop before modeling): {', '.join(types['id_like'][:5])}")

    # Panel 1: Target distribution or Column Breakdown
    if target and target in df.columns:
        t = df[target].dropna()
        if pd.api.types.is_numeric_dtype(t) and t.nunique() > 12:
            counts, edges = np.histogram(t, bins=10)
            lbls = [f"{edges[i]:.0f}-{edges[i + 1]:.0f}" for i in range(len(counts))]
            p1_html = svg_barv(lbls, counts.tolist(), colors=[TOKENS["primary"]] * len(counts))
            p1_title = f"Target Distribution: {target}"
            p1_badge = f"skew={t.skew():.2f} · kurt={t.kurtosis():.2f}"
        else:
            vc = t.value_counts().head(8)
            min_pct = vc.min() / max(len(t), 1) * 100
            ent = _shannon_entropy(t)
            p1_html = svg_barv(vc.index.astype(str).tolist(), vc.values.tolist())
            p1_title = f"Target Balance: {target}"
            p1_badge = f"min={min_pct:.1f}% · H={ent:.2f}b"
    else:
        counts = {k: len(v) for k, v in types.items() if len(v) > 0}
        p1_html = svg_barv(list(counts.keys()), list(counts.values()))
        p1_title, p1_badge = "Column Types Breakdown", f"{n_cols} cols"

    # Panel 2: Target Signal Ranking (Mutual Information) or Variation Ranking
    if target and target in df.columns:
        try:
            X_mi, y_mi, _ = prepare_features(df, target=target)
            if X_mi.shape[1] > 0:
                is_clf = (not pd.api.types.is_numeric_dtype(y_mi)) or y_mi.nunique() <= 15
                if is_clf:
                    mi_vals = mutual_info_classif(X_mi, pd.factorize(y_mi)[0], random_state=42)
                else:
                    mi_vals = mutual_info_regression(X_mi, pd.to_numeric(y_mi, errors="coerce").fillna(0).to_numpy(), random_state=42)
                mi_s = pd.Series(mi_vals, index=X_mi.columns).sort_values(ascending=False).head(10)
                p2_html = svg_barh(mi_s.index.tolist(), mi_s.values.tolist(), colors=[TOKENS["primary"]] * len(mi_s), fmt_str="{:.3f}")
                p2_title, p2_badge = "Target Signal Ranking (Mutual Info)", f"Top {len(mi_s)}"
            else:
                p2_html = '<div class="empty">No usable predictor features</div>'
                p2_title, p2_badge = "Target Signal Ranking", "N/A"
        except Exception:
            p2_html = '<div class="empty">Signal ranking unavailable</div>'
            p2_title, p2_badge = "Target Signal Ranking", "N/A"
    else:
        cov_rows = {}
        for c in num_cols:
            s = df[c].dropna()
            if not s.empty and abs(s.mean()) > 1e-9:
                cov_rows[c] = abs(float(s.std() / s.mean()))
        if cov_rows:
            cov_s = pd.Series(cov_rows).sort_values(ascending=False).head(10)
            p2_html = svg_barh(cov_s.index.tolist(), cov_s.values.tolist(), colors=[TOKENS["purple"]] * len(cov_s), fmt_str="{:.2f}")
            p2_title, p2_badge = "Feature Dispersion (|Std / Mean|)", f"Top {len(cov_s)}"
        else:
            p2_html = '<div class="empty">Provide --target for Mutual Information</div>'
            p2_title, p2_badge = "Feature Dispersion", "No Target"

    # Panel 3: Missingness & Completeness Profile
    miss_series = (df.isna().mean() * 100).sort_values(ascending=False)
    miss_top = miss_series[miss_series > 0].head(10)
    if not miss_top.empty:
        miss_cols_col = [TOKENS["danger"] if v >= 20 else (TOKENS["warning"] if v >= 5 else TOKENS["primary"]) for v in miss_top.values]
        p3_html = svg_barh(
            miss_top.index.tolist(),
            miss_top.values.tolist(),
            colors=miss_cols_col,
            thresholds=[(5.0, TOKENS["warning"], "5% Warn"), (20.0, TOKENS["danger"], "20% High")],
            fmt_str="{:.1f}%",
        )
        p3_badge = f"{len(miss_series[miss_series > 0])} cols affected"
    else:
        p3_html = '<div class="empty" style="color:var(--pos);font-weight:500;">100% Complete — Zero Missing Values</div>'
        p3_badge = "0 missing"

    # Panel 4 & 5: Numeric Diagnostics (IQR vs |Z|>3, Quantiles, Skew, Kurtosis, Zero%)
    num_rows = []
    outlier_rank = []
    skewed_cols = []
    outlier_col_cnt = 0
    for c in num_cols:
        s = df[c].dropna()
        if s.empty:
            continue
        q1, med, q3 = float(s.quantile(0.25)), float(s.median()), float(s.quantile(0.75))
        iqr = q3 - q1
        iqr_out = float(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean() * 100) if iqr > 0 else 0.0
        std_v = float(s.std())
        z_out = float((((s - s.mean()).abs() / std_v) > 3.0).mean() * 100) if std_v > 1e-12 else 0.0
        skew_v = float(s.skew()) if len(s) > 2 else 0.0
        kurt_v = float(s.kurtosis()) if len(s) > 3 else 0.0
        zero_pct = float((s == 0).mean() * 100)
        null_pct = float(df[c].isna().mean() * 100)

        if iqr_out > 1.0:
            outlier_col_cnt += 1
        if abs(skew_v) > 1.5:
            skewed_cols.append(c)

        status = "OUTLIERS" if iqr_out > 5.0 else ("SKEWED" if abs(skew_v) > 1.5 else "OK")
        outlier_rank.append({"Feature": c, "IQR_Out": iqr_out, "Z3_Out": z_out})
        num_rows.append({
            "Feature": c[:18],
            "Mean": round(float(s.mean()), 2),
            "Std": round(std_v, 2),
            "Min": round(float(s.min()), 2),
            "P25": round(q1, 2),
            "Median": round(med, 2),
            "P75": round(q3, 2),
            "Max": round(float(s.max()), 2),
            "Skew": round(skew_v, 2),
            "Kurt": round(kurt_v, 2),
            "Zero%": f"{zero_pct:.1f}%",
            "Null%": f"{null_pct:.1f}%",
            "Outlier%": f"{iqr_out:.1f}%",
            "Status": status,
        })

    num_diag_df = pd.DataFrame(num_rows)
    out_df = pd.DataFrame(outlier_rank).sort_values("IQR_Out", ascending=False).head(10) if outlier_rank else pd.DataFrame()
    if not out_df.empty and out_df["IQR_Out"].max() > 0:
        out_labels = [f"{r['Feature'][:14]} (Z:{r['Z3_Out']:.1f}%)" for _, r in out_df.iterrows()]
        out_cols = [TOKENS["danger"] if v > 5 else (TOKENS["warning"] if v > 1 else TOKENS["primary"]) for v in out_df["IQR_Out"]]
        p4_html = svg_barh(
            out_labels,
            out_df["IQR_Out"].tolist(),
            colors=out_cols,
            thresholds=[(5.0, TOKENS["danger"], "5% High")],
            fmt_str="{:.1f}%",
        )
        p4_badge = f"Max {out_df['IQR_Out'].max():.1f}% IQR"
    else:
        p4_html = '<div class="empty" style="color:var(--pos);font-weight:500;">No IQR Outliers Detected</div>'
        p4_badge = "0% outliers"

    # Panel 6: Categorical Cardinality & Dominance Audit
    cat_rows = []
    for c in cat_cols[:10]:
        s = df[c].dropna().astype(str)
        if s.empty:
            continue
        vc = s.value_counts(normalize=True)
        n_uniq = int(s.nunique())
        uniq_pct = n_uniq / max(len(s), 1) * 100
        top_val = str(vc.index[0])[:14]
        top_share = float(vc.iloc[0] * 100)
        rare_cnt = int((vc < 0.01).sum())
        ent = _shannon_entropy(s)
        c_status = "HIGH_CARD" if n_uniq > 50 else ("DOMINATED" if top_share > 90 else "OK")
        cat_rows.append({
            "Feature": c[:16],
            "Unique": n_uniq,
            "Uniq%": f"{uniq_pct:.1f}%",
            "Top_Value": top_val,
            "Top%": f"{top_share:.1f}%",
            "Rare(<1%)": rare_cnt,
            "Entropy": round(ent, 2),
            "Status": c_status,
        })
    cat_df = pd.DataFrame(cat_rows)
    p6_html = html_table(cat_df, status_col="Status", max_rows=8) if not cat_df.empty else '<div class="empty">No categorical columns detected</div>'

    # Panel 7: Correlation Matrix + Leakage & Multicollinearity Detection
    num_with_target = num_cols + ([target] if target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]) else [])
    corr_df = df[num_with_target].corr(numeric_only=True) if len(num_with_target) >= 2 else pd.DataFrame()
    leak_list: list[str] = []
    if target and target in corr_df.columns:
        target_corr = corr_df[target].drop(index=target).abs()
        leak_list = target_corr[target_corr > 0.95].index.tolist()
        if leak_list:
            warnings.append(f"Possible leakage (|r|>0.95 with {target}): {', '.join(leak_list)}")

    pred_corr = df[num_cols].corr(numeric_only=True).abs() if len(num_cols) >= 2 else pd.DataFrame()
    collinear_pairs: list[str] = []
    if not pred_corr.empty:
        pcols = list(pred_corr.columns)
        for i in range(len(pcols)):
            for j in range(i + 1, len(pcols)):
                r_val = float(pred_corr.iloc[i, j])
                if not math.isnan(r_val) and r_val >= 0.85:
                    collinear_pairs.append(f"{pcols[i]}~{pcols[j]} ({r_val:.2f})")

    p7_html = html_heatmap(corr_df.iloc[:8, :8], fmt_str=".2f", diverging=True) if not corr_df.empty else '<div class="empty">Insufficient numeric columns</div>'

    # Panel 8: Automated Data Hygiene, Collinearity & Leakage Audit
    audit_checks = [
        {
            "Check": "Duplicate Rows",
            "Observation": f"{dup_rows:,} rows ({dup_pct:.2f}%)",
            "Rule": "> 0% triggers deduplication",
            "Status": "HIGH" if dup_pct > 5 else ("WARN" if dup_rows > 0 else "PASS"),
        },
        {
            "Check": "Constant Columns",
            "Observation": ", ".join(types["constant"][:4]) or "None",
            "Rule": "nunique <= 1 (zero variance)",
            "Status": "WARN" if types["constant"] else "PASS",
        },
        {
            "Check": "ID-Like Columns",
            "Observation": ", ".join(types["id_like"][:4]) or "None",
            "Rule": "Drop before ML training",
            "Status": "WARN" if types["id_like"] else "PASS",
        },
        {
            "Check": "Target Leakage (|r|>0.95)",
            "Observation": ", ".join(leak_list[:4]) or ("None" if target else "No target"),
            "Rule": "Check post-outcome features",
            "Status": "HIGH" if leak_list else "PASS",
        },
        {
            "Check": "Multicollinearity (|r|>=0.85)",
            "Observation": ", ".join(collinear_pairs[:3]) or "None",
            "Rule": "Run /features (VIF) to prune",
            "Status": "WARN" if collinear_pairs else "PASS",
        },
        {
            "Check": "High Skewness (|skew|>1.5)",
            "Observation": ", ".join(skewed_cols[:4]) or "None",
            "Rule": "Log / Yeo-Johnson candidate",
            "Status": "WARN" if skewed_cols else "PASS",
        },
        {
            "Check": "Missing Data Severity",
            "Observation": f"{miss_pct:.2f}% overall ({len(miss_series[miss_series > 0])} cols)",
            "Rule": ">5% warn, >20% critical",
            "Status": "HIGH" if miss_pct > 20 else ("WARN" if miss_pct > 5 or len(miss_top) > 0 else "PASS"),
        },
    ]
    audit_df = pd.DataFrame(audit_checks)
    n_pass = int((audit_df["Status"] == "PASS").sum())
    hygiene_score = round(n_pass / len(audit_df) * 100)

    kpis = [
        ("Rows", f"{n_rows:,}", TOKENS["primary"]),
        ("Columns", f"{n_cols} ({len(num_cols)}n/{len(cat_cols)}c/{len(dt_cols)}d)", TOKENS["purple"]),
        ("Missing Cells", f"{miss_pct:.2f}%", TOKENS["warning"] if miss_pct > 5 else TOKENS["success"]),
        ("Duplicates", f"{dup_rows:,} ({dup_pct:.1f}%)", TOKENS["danger"] if dup_rows > 0 else TOKENS["success"]),
        ("Outlier Cols (IQR)", f"{outlier_col_cnt} / {len(num_cols)}", TOKENS["warning"] if outlier_col_cnt > 0 else TOKENS["success"]),
        ("Memory / Hygiene", f"{mem_mb:.1f}MB · {hygiene_score}%", TOKENS["success"] if hygiene_score >= 80 else TOKENS["warning"]),
    ]

    panels = [
        {"title": p1_title, "badge": p1_badge, "html": p1_html},
        {"title": p2_title, "badge": p2_badge, "html": p2_html},
        {"title": "Missingness & Completeness Profile", "badge": p3_badge, "html": p3_html},
        {"title": "Outlier Intensity (IQR% vs |Z|>3%)", "badge": p4_badge, "html": p4_html},
        {"title": "Extended Numeric Feature Diagnostics", "badge": f"{len(num_diag_df)} features", "html": html_table(num_diag_df, status_col="Status", max_rows=10), "span": 2},
        {"title": "Categorical Cardinality & Dominance", "badge": f"{len(cat_cols)} cat cols", "html": p6_html},
        {"title": "Correlation Matrix (Top 8)", "badge": "Pearson r", "html": p7_html},
        {"title": "Automated Data Hygiene, Collinearity & Leakage Audit", "badge": f"{n_pass}/{len(audit_df)} PASS", "html": html_table(audit_df, status_col="Status", max_rows=8), "span": 2},
    ]

    png_path, html_path = render_dashboard(
        title="Exploratory Data Analysis (EDA)",
        subtitle=f"Dataset: {Path(data_path).name}" + (f" | Target: {target}" if target else "") + f" | Hygiene Score: {hygiene_score}%",
        kpis=kpis,
        panels=panels,
        out_dir=out_dir,
        file_stem="eda_dashboard",
        fmt=fmt,
    )
    emit_result(
        skill="ds-eda",
        out_dir=out_dir,
        kpis={
            "rows": n_rows,
            "cols": n_cols,
            "missing_pct": round(miss_pct, 2),
            "dups": dup_rows,
            "outlier_cols": outlier_col_cnt,
            "hygiene_pct": hygiene_score,
        },
        visuals=[png_path, html_path],
        warnings=warnings,
        extra={"types": types, "skewed": skewed_cols, "collinear": collinear_pairs},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--target", default=None)
    p.add_argument("--format", default="landscape", choices=["landscape", "portrait"])
    p.add_argument("--theme", default=None, choices=["light", "dark"])
    p.add_argument("--out", default="outputs/ds-eda")
    args = p.parse_args()
    set_theme(args.theme)
    run_eda(args.data, args.target, args.format, args.out)
