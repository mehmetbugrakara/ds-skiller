"""Deep classification evaluation: PR curve, calibration, cumulative gains, threshold table."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, log_loss, precision_recall_curve, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ds_core import set_theme, TOKENS, emit_result, html_table, load_dataset, render_dashboard, svg_line_chart


def run_metrics(preds_path: str, fmt: str, out_dir: str) -> None:
    df = load_dataset(preds_path)
    y_true = df["y_true"].values
    if "y_prob" not in df.columns:
        raise ValueError("Predictions file must contain 'y_prob' column.")

    y_prob = df["y_prob"].values
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ll = log_loss(y_true, y_prob)

    prec, rec, thresh = precision_recall_curve(y_true, y_prob)
    f1s = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    best_idx = int(np.argmax(f1s))
    opt_thresh = float(thresh[best_idx])
    opt_f1 = float(f1s[best_idx])

    pos_probs = np.sort(y_prob[y_true == 1])
    neg_probs = np.sort(y_prob[y_true == 0])
    all_pts = np.sort(np.unique(y_prob))
    cdf_pos = np.searchsorted(pos_probs, all_pts, side="right") / max(len(pos_probs), 1)
    cdf_neg = np.searchsorted(neg_probs, all_pts, side="right") / max(len(neg_probs), 1)
    ks_stat = float(np.max(np.abs(cdf_pos - cdf_neg)))

    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    pct_pop = np.arange(1, len(y_sorted) + 1) / len(y_sorted) * 100
    cum_gain = np.cumsum(y_sorted) / max(y_sorted.sum(), 1) * 100
    top20_idx = int(0.20 * len(y_sorted))
    gain_at_20 = float(cum_gain[top20_idx]) if top20_idx < len(cum_gain) else 0.0
    lift_at_20 = gain_at_20 / 20.0

    rows = []
    for t in sorted(set([0.3, 0.4, 0.5, 0.6, 0.7, round(opt_thresh, 2)])):
        yp = (y_prob >= t).astype(int)
        rows.append({
            "Threshold": f"{t:.2f}",
            "Precision": round(float(precision_score(y_true, yp, zero_division=0)), 3),
            "Recall": round(float(recall_score(y_true, yp, zero_division=0)), 3),
            "F1": round(float(f1_score(y_true, yp, zero_division=0)), 3),
            "Flagged%": f"{yp.mean() * 100:.1f}%",
            "Status": "BEST" if abs(t - opt_thresh) < 0.01 else "OK",
        })
    thresh_df = pd.DataFrame(rows)
    best_row_idx = int(thresh_df.index[thresh_df["Status"] == "BEST"][0]) if (thresh_df["Status"] == "BEST").any() else None

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")

    pr_svg = svg_line_chart([{"name": f"PR Curve (AUC={pr_auc:.3f})", "x": rec.tolist(), "y": prec.tolist(), "color": TOKENS["primary"], "fill": True}], x_label="Recall")
    gains_svg = svg_line_chart(
        [
            {"name": f"Model Gains ({gain_at_20:.0f}% @20%)", "x": pct_pop[::5].tolist(), "y": cum_gain[::5].tolist(), "color": TOKENS["success"], "fill": True},
            {"name": "Random Baseline", "x": [0.0, 100.0], "y": [0.0, 100.0], "color": TOKENS["muted"], "dashed": True},
        ],
        x_label="% Population Targeted",
    )
    cal_svg = svg_line_chart([{"name": f"Calibration (Brier={brier:.3f})", "x": prob_pred.tolist(), "y": prob_true.tolist(), "color": TOKENS["purple"]}], x_label="Mean Predicted Prob", diag_line=True)

    kpis = [
        ("PR-AUC", f"{pr_auc:.4f}", TOKENS["primary"]),
        ("KS Statistic", f"{ks_stat:.3f}", TOKENS["success"]),
        ("Brier / LogLoss", f"{brier:.3f} / {ll:.3f}", TOKENS["purple"]),
        ("Opt Threshold", f"{opt_thresh:.2f} (F1={opt_f1:.3f})", TOKENS["warning"]),
        ("Lift @ Top 20%", f"{lift_at_20:.2f}x ({gain_at_20:.0f}%)", TOKENS["success"]),
    ]

    panels = [
        {"title": "Precision-Recall Curve", "badge": f"PR-AUC={pr_auc:.3f}", "html": pr_svg},
        {"title": "Cumulative Gains / Lift Curve", "badge": f"Top 20% = {lift_at_20:.2f}x", "html": gains_svg},
        {"title": "Calibration Reliability Curve", "badge": f"Brier={brier:.4f}", "html": cal_svg},
        {"title": "Threshold Optimization Table", "badge": f"Opt t={opt_thresh:.2f}", "html": html_table(thresh_df, highlight_row=best_row_idx, status_col="Status")},
    ]

    png_path, html_path = render_dashboard(
        title="Model Evaluation & Business Impact Metrics",
        subtitle=f"Predictions: {Path(preds_path).name} | LogLoss={ll:.4f} | N={len(y_true):,}",
        kpis=kpis,
        panels=panels,
        out_dir=out_dir,
        file_stem="metrics_dashboard",
        fmt=fmt,
    )
    emit_result(
        skill="ds-metrics",
        out_dir=out_dir,
        kpis={"pr_auc": round(pr_auc, 4), "ks": round(ks_stat, 3), "opt_thresh": round(opt_thresh, 2), "lift_20": round(lift_at_20, 2)},
        visuals=[png_path, html_path],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--preds", default="outputs/ds-classification/test_predictions.csv")
    p.add_argument("--format", default="landscape", choices=["landscape", "portrait"])
    p.add_argument("--theme", default=None, choices=["light", "dark"])
    p.add_argument("--out", default="outputs/ds-metrics")
    args = p.parse_args()
    set_theme(args.theme)
    run_metrics(args.preds, args.format, args.out)
