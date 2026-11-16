"""Model Fairness & Subgroup Slice Error Audit."""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ds_core import set_theme, TOKENS, emit_result, html_table, load_dataset, prepare_features, render_dashboard, svg_barh


def run_audit(data_path: str, model_path: str, slice_col: str, fmt: str, out_dir: str) -> None:
    bundle = joblib.load(model_path)
    mdl = bundle["model"]
    features = bundle["features"]
    target = bundle["target"]
    le = bundle.get("label_encoder")

    df = load_dataset(data_path).dropna(subset=[target, slice_col]).copy()
    X_all, y_raw, _ = prepare_features(df, target=target)
    for f in features:
        if f not in X_all.columns:
            X_all[f] = 0.0
    X = X_all[features]
    y = le.transform(y_raw.astype(str)) if le is not None else pd.to_numeric(y_raw, errors="coerce").fillna(0).astype(int).to_numpy()

    y_pred = mdl.predict(X)
    y_prob = mdl.predict_proba(X)[:, 1] if hasattr(mdl, "predict_proba") else y_pred.astype(float)

    # Bin numeric slice col into quartiles if > 8 unique values
    seg = df.loc[X.index, slice_col]
    if pd.api.types.is_numeric_dtype(seg) and seg.nunique() > 8:
        seg = pd.qcut(seg, q=4, duplicates="drop").astype(str)
    else:
        seg = seg.astype(str)

    rows = []
    for g_name, idx in seg.groupby(seg).groups.items():
        pos = X.index.get_indexer(idx)
        yt, yp, ypr = y[pos], y_pred[pos], y_prob[pos]
        if len(yt) < 5:
            continue
        pos_rate = float(yp.mean())
        tpr = float(((yp == 1) & (yt == 1)).sum() / max((yt == 1).sum(), 1))
        fpr = float(((yp == 1) & (yt == 0)).sum() / max((yt == 0).sum(), 1))
        auc = float(roc_auc_score(yt, ypr)) if len(np.unique(yt)) == 2 else np.nan
        f1 = float(f1_score(yt, yp, zero_division=0))
        acc = float(accuracy_score(yt, yp))
        rows.append({
            "Segment": str(g_name)[:18],
            "N": len(yt),
            "Pred_Pos%": round(pos_rate * 100, 1),
            "TPR": round(tpr, 3),
            "FPR": round(fpr, 3),
            "ROC_AUC": round(auc, 3) if not np.isnan(auc) else "-",
            "F1": round(f1, 3),
            "Acc": round(acc, 3),
        })

    audit_df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    dp_gap = float(audit_df["Pred_Pos%"].max() - audit_df["Pred_Pos%"].min()) if not audit_df.empty else 0.0
    tpr_gap = float(audit_df["TPR"].max() - audit_df["TPR"].min()) if not audit_df.empty else 0.0
    fpr_gap = float(audit_df["FPR"].max() - audit_df["FPR"].min()) if not audit_df.empty else 0.0
    worst_seg = audit_df.iloc[-1]["Segment"] if not audit_df.empty else "-"
    fairness_ok = tpr_gap <= 0.15 and fpr_gap <= 0.15

    kpis = [
        ("Fairness Status", "PASS" if fairness_ok else "REVIEW GAP", TOKENS["success"] if fairness_ok else TOKENS["danger"]),
        ("Dem. Parity Gap", f"{dp_gap:.1f} pp", TOKENS["warning"] if dp_gap > 15 else TOKENS["primary"]),
        ("TPR Gap (Eq. Opp)", f"{tpr_gap:.3f}", TOKENS["danger"] if tpr_gap > 0.15 else TOKENS["success"]),
        ("FPR Gap", f"{fpr_gap:.3f}", TOKENS["danger"] if fpr_gap > 0.15 else TOKENS["success"]),
        ("Worst Slice (F1)", f"{worst_seg}", TOKENS["warning"]),
    ]

    f1_svg = svg_barh(audit_df["Segment"].tolist(), audit_df["F1"].tolist(),
                      colors=[TOKENS["primary"]] * len(audit_df), fmt_str="{:.3f}")
    tpr_svg = svg_barh(audit_df["Segment"].tolist(), audit_df["TPR"].tolist(),
                       colors=[TOKENS["success"] if v >= audit_df["TPR"].mean() else TOKENS["warning"] for v in audit_df["TPR"]],
                       fmt_str="{:.3f}")

    panels = [
        {"title": f"Subgroup F1 Score by {slice_col}", "badge": f"{len(audit_df)} slices", "html": f1_svg},
        {"title": f"True Positive Rate (Equal Opportunity) by {slice_col}", "badge": f"Gap={tpr_gap:.3f}", "html": tpr_svg},
        {"title": "Subgroup Fairness & Error Matrix", "badge": "Demographic Parity & Equalized Odds", "html": html_table(audit_df, highlight_row=len(audit_df) - 1), "span": 2},
    ]

    png_path, html_path = render_dashboard(
        title="Model Fairness & Subgroup Slice Audit",
        subtitle=f"Dataset: {Path(data_path).name} | Slice: {slice_col} | Model: {Path(model_path).name}",
        kpis=kpis,
        panels=panels,
        out_dir=out_dir,
        file_stem="audit_dashboard",
        fmt=fmt,
    )
    warnings = [f"TPR gap ({tpr_gap:.3f}) > 0.15 across '{slice_col}' slices"] if tpr_gap > 0.15 else []
    emit_result(
        skill="ds-audit",
        out_dir=out_dir,
        kpis={"dp_gap_pp": round(dp_gap, 1), "tpr_gap": round(tpr_gap, 3), "fpr_gap": round(fpr_gap, 3), "worst": worst_seg},
        visuals=[png_path, html_path],
        warnings=warnings,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="outputs/ds-classification/best_model.joblib")
    p.add_argument("--slice", required=True)
    p.add_argument("--format", default="landscape", choices=["landscape", "portrait"])
    p.add_argument("--theme", default=None, choices=["light", "dark"])
    p.add_argument("--out", default="outputs/ds-audit")
    args = p.parse_args()
    set_theme(args.theme)
    run_audit(args.data, args.model, args.slice, args.format, args.out)
