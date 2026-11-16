"""Batch inference with a saved ds-skiller model bundle (+ live evaluation if labels are present)."""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ds_core import (
    TOKENS,
    emit_result,
    html_table,
    load_dataset,
    prepare_inference_features,
    render_dashboard,
    set_theme,
    svg_barv,
)


def _hist(values: np.ndarray, bins: int = 10) -> tuple[list[str], list[float]]:
    counts, edges = np.histogram(values, bins=bins)
    labels = [f"{edges[i]:.2g}" for i in range(len(counts))]
    return labels, counts.astype(float).tolist()


def run_predict(model_path: str, data_path: str, threshold: float, id_col: str | None, fmt: str, out_dir: str) -> None:
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path} (train one with ds-classification / ds-regression / ds-tune)")
    bundle = joblib.load(model_path)
    mdl, features, target = bundle["model"], bundle["features"], bundle.get("target")
    le = bundle.get("label_encoder")
    is_clf = le is not None or hasattr(mdl, "predict_proba")

    df = load_dataset(data_path)
    has_label = target is not None and target in df.columns
    feat_df = df.drop(columns=[target], errors="ignore")
    if id_col and id_col in feat_df.columns:
        feat_df = feat_df.drop(columns=[id_col])
    X, meta = prepare_inference_features(feat_df, features)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(index=df.index)
    if id_col and id_col in df.columns:
        res[id_col] = df[id_col]

    warnings: list[str] = []
    live: dict = {}
    if meta["missing_features"]:
        warnings.append(
            f"{len(meta['missing_features'])} trained feature(s) missing in data, filled with 0: {meta['missing_features'][:4]}"
        )

    dist_panel = None
    if is_clf:
        prob = mdl.predict_proba(X)
        classes = [str(c) for c in le.classes_] if le is not None else [str(c) for c in range(prob.shape[1])]
        binary = prob.shape[1] == 2
        if binary:
            score = prob[:, 1]
            pred_idx = (score >= threshold).astype(int)
        else:
            pred_idx = prob.argmax(axis=1)
            score = prob.max(axis=1)
        res["score"] = score.round(5)
        res["prediction"] = [classes[i] for i in pred_idx]
        dist = res["prediction"].value_counts()
        hist_lbl, hist_val = _hist(score)
        kpis = {
            "rows": len(res),
            "positive_rate": round(float(pred_idx.mean()), 4) if binary else "-",
            "mean_score": round(float(score.mean()), 4),
        }
        if has_label:
            y_true = df[target].astype(str)
            y_idx = np.array([classes.index(v) if v in classes else -1 for v in y_true])
            ok = y_idx >= 0
            live["accuracy"] = round(float(accuracy_score(y_idx[ok], pred_idx[ok])), 4)
            live["f1"] = round(float(f1_score(y_idx[ok], pred_idx[ok], average="binary" if binary else "macro")), 4)
            if binary and len(np.unique(y_idx[ok])) == 2:
                live["roc_auc"] = round(float(roc_auc_score(y_idx[ok], score[ok])), 4)
        k_list = [
            ("Rows Scored", f"{len(res):,}", TOKENS["primary"]),
            (
                "Positive Rate" if binary else "Classes",
                f"{pred_idx.mean() * 100:.1f}%" if binary else str(len(classes)),
                TOKENS["warning"],
            ),
            ("Mean Score", f"{score.mean():.3f}", TOKENS["purple"]),
            ("Threshold" if binary else "Top Class", f"{threshold:.2f}" if binary else str(dist.index[0]), TOKENS["teal"]),
        ]
        dist_panel = svg_barv([str(i) for i in dist.index[:8]], dist.values[:8].astype(float).tolist())
        score_title = "Score Distribution · P(positive)" if binary else "Max Class Probability"
    else:
        pred = np.ravel(mdl.predict(X)).astype(float)
        res["prediction"] = pred.round(4)
        hist_lbl, hist_val = _hist(pred)
        kpis = {"rows": len(res), "mean_pred": round(float(pred.mean()), 4)}
        if has_label:
            y_true = pd.to_numeric(df[target], errors="coerce").to_numpy()
            ok = ~np.isnan(y_true)
            live["r2"] = round(float(r2_score(y_true[ok], pred[ok])), 4)
            live["rmse"] = round(float(np.sqrt(mean_squared_error(y_true[ok], pred[ok]))), 4)
            live["mae"] = round(float(mean_absolute_error(y_true[ok], pred[ok])), 4)
        k_list = [
            ("Rows Scored", f"{len(res):,}", TOKENS["primary"]),
            ("Mean Prediction", f"{pred.mean():,.4g}", TOKENS["purple"]),
            ("P10 / P90", f"{np.percentile(pred, 10):,.3g} / {np.percentile(pred, 90):,.3g}", TOKENS["teal"]),
        ]
        score_title = "Prediction Distribution"

    if has_label:
        res["y_true"] = df[target].values
        name, val = next(iter(live.items()))
        k_list.append((f"Live {name.upper()}", f"{val:.4f}", TOKENS["success"]))
    else:
        k_list.append(("Live Eval", "no labels", TOKENS["muted"]))
    n_miss = len(meta["missing_features"])
    k_list.append(("Schema", "OK" if not n_miss else f"{n_miss} missing", TOKENS["success"] if not n_miss else TOKENS["danger"]))

    pred_csv = out / "predictions.csv"
    res.to_csv(pred_csv, index=False)

    panels = [{"title": score_title, "badge": f"n={len(res):,}", "html": svg_barv(hist_lbl, hist_val)}]
    if dist_panel:
        panels.append({"title": "Predicted Class Distribution", "badge": "Counts", "html": dist_panel})
    if live:
        panels.append({"title": "Live Evaluation (labels found in data)", "badge": str(target), "html": html_table(pd.DataFrame([live]))})
    panels.append({"title": "Sample Predictions", "badge": pred_csv.name, "html": html_table(res.head(8)), "span": 2})

    png_path, html_path = render_dashboard(
        title="Batch Inference & Scoring Report",
        subtitle=f"Model: {Path(model_path).name} ({type(mdl).__name__}) | Data: {Path(data_path).name} | {len(features)} features",
        kpis=k_list,
        panels=panels,
        out_dir=out_dir,
        file_stem="predict_dashboard",
        fmt=fmt,
    )
    kpis.update(live)
    kpis["csv"] = str(pred_csv)
    emit_result(skill="ds-predict", out_dir=out_dir, kpis=kpis, visuals=[png_path, html_path], warnings=warnings)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Score new data with a saved ds-skiller model bundle.")
    p.add_argument("--model", default="outputs/ds-classification/best_model.joblib")
    p.add_argument("--data", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--id", dest="id_col", default=None, help="ID column to carry into predictions.csv")
    p.add_argument("--format", default="landscape", choices=["landscape", "portrait"])
    p.add_argument("--theme", default=None, choices=["light", "dark"])
    p.add_argument("--out", default="outputs/ds-predict")
    args = p.parse_args()
    set_theme(args.theme)
    run_predict(args.model, args.data, args.threshold, args.id_col, args.format, args.out)
