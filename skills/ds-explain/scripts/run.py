"""SHAP model explainability Impeccable HTML/PNG dashboard."""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ds_core import set_theme, TOKENS, emit_result, html_table, load_dataset, render_dashboard, svg_barh, svg_scatter


def run_explain(model_path: str, data_path: str, max_samples: int, fmt: str, out_dir: str) -> None:
    bundle = joblib.load(model_path)
    mdl = bundle["model"]
    features = bundle["features"]

    df = load_dataset(data_path)
    X = df[features].copy()
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=42)

    est = mdl.named_steps[list(mdl.named_steps.keys())[-1]] if hasattr(mdl, "named_steps") else mdl
    X_trans = pd.DataFrame(mdl[:-1].transform(X), columns=features, index=X.index) if (hasattr(mdl, "named_steps") and len(mdl.named_steps) > 1) else X

    if any(k in type(est).__name__ for k in ("LGBM", "RandomForest", "HistGradient", "XGB", "GradientBoosting", "CatBoost")):
        sv = shap.TreeExplainer(est).shap_values(X_trans)
    elif "LogisticRegression" in type(est).__name__ or "Ridge" in type(est).__name__:
        sv = shap.LinearExplainer(est, X_trans).shap_values(X_trans)
    else:
        bg = shap.sample(X_trans, min(50, len(X_trans)))
        sv = shap.KernelExplainer(est.predict, bg).shap_values(X_trans.iloc[:80])
        X, X_trans = X.iloc[:80], X_trans.iloc[:80]

    if isinstance(sv, list):
        shap_vals = sv[1] if len(sv) == 2 else np.mean(np.abs(np.stack(sv, axis=0)), axis=0)
    elif sv.ndim == 3:
        shap_vals = sv[:, :, 1] if sv.shape[2] == 2 else np.mean(np.abs(sv), axis=2)
    else:
        shap_vals = sv

    mean_abs = np.abs(shap_vals).mean(axis=0)
    mean_signed = shap_vals.mean(axis=0)
    imp_df = pd.DataFrame({
        "Feature": features,
        "Mean_Abs_SHAP": np.round(mean_abs, 4),
        "Mean_Signed": np.round(mean_signed, 4),
        "Direction": ["Positive (+)" if v >= 0 else "Negative (-)" for v in mean_signed],
    }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

    top_feats = imp_df["Feature"].head(10).tolist()
    top1 = top_feats[0]
    top1_idx = features.index(top1)

    sample_idx = int(np.argmax(np.abs(shap_vals).sum(axis=1)))
    local_sv = pd.Series(shap_vals[sample_idx], index=features)
    local_top = local_sv.reindex(local_sv.abs().sort_values(ascending=False).head(8).index)

    global_svg = svg_barh(imp_df["Feature"].head(8).tolist(), imp_df["Mean_Abs_SHAP"].head(8).tolist(),
                          colors=[TOKENS["primary"]] * min(8, len(imp_df)), fmt_str="{:.4f}")
    dep_svg = svg_scatter([{"name": top1, "x": pd.to_numeric(X[top1], errors="coerce").fillna(0).tolist(), "y": shap_vals[:, top1_idx].tolist(), "color": TOKENS["success"]}],
                          x_label=top1, hline=0.0)
    loc_cols = [TOKENS["danger"] if v > 0 else TOKENS["primary"] for v in local_top.values]
    local_svg = svg_barh(local_top.index.tolist(), local_top.values.tolist(), colors=loc_cols, fmt_str="{:+.3f}")

    kpis = [
        ("Top Driver #1", top1, TOKENS["primary"]),
        ("#1 Mean |SHAP|", f"{imp_df.loc[0, 'Mean_Abs_SHAP']:.4f}", TOKENS["success"]),
        ("Top Driver #2", top_feats[1] if len(top_feats) > 1 else "-", TOKENS["purple"]),
        ("Explained Rows", f"{len(X):,}", TOKENS["muted"]),
    ]

    panels = [
        {"title": "Global Feature Importance (Mean |SHAP|)", "badge": type(est).__name__, "html": global_svg},
        {"title": f"Dependence Plot: {top1}", "badge": "SHAP Impact", "html": dep_svg},
        {"title": f"Local Explanation (Sample #{sample_idx})", "badge": "Red=+Risk / Blue=-Risk", "html": local_svg},
        {"title": "SHAP Feature Impact Summary", "badge": f"{len(features)} features", "html": html_table(imp_df, max_rows=8)},
    ]

    png_path, html_path = render_dashboard(
        title="Model Explainability (SHAP Analysis)",
        subtitle=f"Model: {type(est).__name__} | Features: {len(features)}",
        kpis=kpis,
        panels=panels,
        out_dir=out_dir,
        file_stem="shap_dashboard",
        fmt=fmt,
    )
    emit_result(
        skill="ds-explain",
        out_dir=out_dir,
        kpis={"top1": top1, "top2": top_feats[1] if len(top_feats) > 1 else "-", "top3": top_feats[2] if len(top_feats) > 2 else "-"},
        visuals=[png_path, html_path],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="outputs/ds-classification/best_model.joblib")
    p.add_argument("--data", default="outputs/ds-classification/test_predictions.csv")
    p.add_argument("--max-samples", type=int, default=300)
    p.add_argument("--format", default="landscape", choices=["landscape", "portrait"])
    p.add_argument("--theme", default=None, choices=["light", "dark"])
    p.add_argument("--out", default="outputs/ds-explain")
    args = p.parse_args()
    set_theme(args.theme)
    run_explain(args.model, args.data, args.max_samples, args.format, args.out)
