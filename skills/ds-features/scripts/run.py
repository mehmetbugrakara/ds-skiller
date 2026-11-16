"""Feature Selection & Multicollinearity Audit (VIF, Mutual Information, Spearman)."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ds_core import set_theme, TOKENS, emit_result, html_table, load_dataset, prepare_features, render_dashboard, svg_barh


def run_features(data_path: str, target: str, fmt: str, out_dir: str) -> None:
    df = load_dataset(data_path)
    X, y_raw, meta = prepare_features(df, target=target)

    is_clf = (not pd.api.types.is_numeric_dtype(y_raw)) or y_raw.nunique() <= 15
    if is_clf:
        y_num = pd.factorize(y_raw)[0]
        mi = mutual_info_classif(X, y_num, random_state=42)
    else:
        y_num = pd.to_numeric(y_raw, errors="coerce").fillna(0).to_numpy()
        mi = mutual_info_regression(X, y_num, random_state=42)

    # VIF calculation on numeric-compatible standardized matrix (cap to 25 features for speed)
    X_vif = X.iloc[:, :25].astype(float)
    X_mat = X_vif.to_numpy()
    vifs = {}
    if X_mat.shape[1] >= 2:
        X_std = (X_mat - X_mat.mean(axis=0)) / np.where(X_mat.std(axis=0) == 0, 1.0, X_mat.std(axis=0))
        for i, col in enumerate(X_vif.columns):
            try:
                vifs[col] = float(variance_inflation_factor(X_std, i))
            except Exception:
                vifs[col] = 1.0

    # Spearman correlation with target
    spearman = {}
    y_s = pd.Series(y_num, index=X.index)
    for col in X.columns:
        spearman[col] = float(abs(X[col].corr(y_s, method="spearman")))

    # Redundant feature pairs (|r| > 0.85)
    corr_mat = X.corr().abs()
    redundant_pairs = []
    cols = list(corr_mat.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r_val = float(corr_mat.iloc[i, j])
            if r_val >= 0.85:
                redundant_pairs.append({"Feature_A": cols[i][:18], "Feature_B": cols[j][:18], "Abs_r": round(r_val, 3)})

    rows = []
    for col, m_val in zip(X.columns, mi):
        v_val = vifs.get(col, 1.0)
        sp_val = spearman.get(col, 0.0)
        if v_val >= 10:
            action = "DROP (VIF>=10)"
        elif m_val < 0.005 and sp_val < 0.03:
            action = "WARN (Low Signal)"
        else:
            action = "KEEP"
        rows.append({
            "Feature": col[:22],
            "MI_Score": round(float(m_val), 4),
            "Spearman_r": round(float(sp_val), 3),
            "VIF": round(float(v_val), 2),
            "Status": action,
        })

    feat_df = pd.DataFrame(rows).sort_values("MI_Score", ascending=False).reset_index(drop=True)
    n_high_vif = int((feat_df["VIF"] >= 5.0).sum())
    n_low_sig = int(feat_df["Status"].str.contains("WARN").sum())

    kpis = [
        ("Evaluated Features", f"{len(feat_df)}", TOKENS["primary"]),
        ("Top Signal (MI)", f"{feat_df.loc[0, 'Feature'][:16]} ({feat_df.loc[0, 'MI_Score']:.3f})", TOKENS["success"]),
        ("High VIF (>=5)", f"{n_high_vif}", TOKENS["danger"] if n_high_vif > 0 else TOKENS["success"]),
        ("Redundant Pairs", f"{len(redundant_pairs)}", TOKENS["warning"] if redundant_pairs else TOKENS["success"]),
        ("Low Signal", f"{n_low_sig}", TOKENS["warning"] if n_low_sig > 0 else TOKENS["success"]),
    ]

    top_mi = feat_df.head(10)
    mi_svg = svg_barh(top_mi["Feature"].tolist(), top_mi["MI_Score"].tolist(),
                      colors=[TOKENS["primary"]] * len(top_mi), fmt_str="{:.4f}")

    vif_sorted = feat_df.sort_values("VIF", ascending=False).head(10)
    vif_cols = [TOKENS["danger"] if v >= 10 else (TOKENS["warning"] if v >= 5 else TOKENS["success"]) for v in vif_sorted["VIF"]]
    vif_svg = svg_barh(vif_sorted["Feature"].tolist(), vif_sorted["VIF"].tolist(),
                       colors=vif_cols, thresholds=[(5.0, TOKENS["warning"], "VIF=5"), (10.0, TOKENS["danger"], "VIF=10")],
                       fmt_str="{:.2f}")

    panels = [
        {"title": "Mutual Information with Target (Top 10)", "badge": "Non-linear signal", "html": mi_svg},
        {"title": "Multicollinearity Audit (VIF)", "badge": "VIF >= 5 Check", "html": vif_svg},
        {"title": "Feature Selection Scorecard", "badge": f"{len(feat_df)} features", "html": html_table(feat_df, status_col="Status", max_rows=9), "span": 2},
    ]

    png_path, html_path = render_dashboard(
        title="Feature Selection & Multicollinearity Audit",
        subtitle=f"Dataset: {Path(data_path).name} | Target: {target} | Dropped ID/Const: {len(meta['dropped'])}",
        kpis=kpis,
        panels=panels,
        out_dir=out_dir,
        file_stem="features_dashboard",
        fmt=fmt,
    )
    emit_result(
        skill="ds-features",
        out_dir=out_dir,
        kpis={"features": len(feat_df), "top_mi": feat_df.loc[0, "Feature"], "high_vif": n_high_vif, "redundant": len(redundant_pairs)},
        visuals=[png_path, html_path],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--format", default="landscape", choices=["landscape", "portrait"])
    p.add_argument("--theme", default=None, choices=["light", "dark"])
    p.add_argument("--out", default="outputs/ds-features")
    args = p.parse_args()
    set_theme(args.theme)
    run_features(args.data, args.target, args.format, args.out)
