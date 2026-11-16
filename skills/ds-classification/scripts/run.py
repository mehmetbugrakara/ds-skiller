"""Classification benchmark (XGBoost, CatBoost, LightGBM, RandomForest, LogReg)."""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

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
    svg_barh,
    svg_barv,
    svg_line_chart,
)


def _cv_score_explicit(factory, X: pd.DataFrame, y: np.ndarray, skf: StratifiedKFold, is_binary: bool) -> float:
    """Explicit StratifiedKFold loop (avoids sklearn>=1.6 __sklearn_tags__ issues on CatBoost)."""
    scores = []
    for tr_idx, va_idx in skf.split(X, y):
        m = factory()
        m.fit(X.iloc[tr_idx], y[tr_idx])
        if is_binary:
            prob = m.predict_proba(X.iloc[va_idx])[:, 1]
            scores.append(roc_auc_score(y[va_idx], prob))
        else:
            pred = m.predict(X.iloc[va_idx])
            if np.ndim(pred) > 1:
                pred = np.ravel(pred)
            scores.append(f1_score(y[va_idx], pred, average="macro"))
    return float(np.mean(scores))


def run_classification(data_path: str, target: str, cv_folds: int, fmt: str, out_dir: str) -> None:
    df = load_dataset(data_path)
    X, y_raw, meta = prepare_features(df, target=target)

    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))
    n_classes = len(le.classes_)
    is_binary = n_classes == 2

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    factories = {
        "XGBoost": lambda: XGBClassifier(
            n_estimators=250, learning_rate=0.05, max_depth=6, random_state=42,
            eval_metric="logloss", verbosity=0, n_jobs=-1,
        ),
        "CatBoost": lambda: CatBoostClassifier(
            iterations=250, learning_rate=0.05, depth=6, random_seed=42,
            verbose=0, allow_writing_files=False,
        ),
        "LightGBM": lambda: LGBMClassifier(
            n_estimators=250, learning_rate=0.05, random_state=42, verbose=-1, n_jobs=-1,
        ),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1,
        ),
        "LogReg": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=42),
        ),
    }

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    results, fitted, roc_series = [], {}, []
    for idx, (name, make_mdl) in enumerate(factories.items()):
        cv_mean = _cv_score_explicit(make_mdl, X_train, y_train, skf, is_binary)
        mdl = make_mdl()
        mdl.fit(X_train, y_train)
        fitted[name] = mdl

        y_pred = np.ravel(mdl.predict(X_test)).astype(int)
        y_prob = mdl.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="binary" if is_binary else "macro")
        if is_binary:
            auc = roc_auc_score(y_test, y_prob[:, 1])
            pr_auc = average_precision_score(y_test, y_prob[:, 1])
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            roc_series.append({"name": f"{name} ({auc:.3f})", "x": fpr.tolist(), "y": tpr.tolist(), "color": TOKENS["series"][idx % 6]})
        else:
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
            pr_auc = np.nan

        results.append({
            "Model": name,
            "CV_Mean": round(cv_mean, 4),
            "ROC_AUC": round(float(auc), 4),
            "PR_AUC": round(float(pr_auc), 4) if is_binary else "-",
            "F1": round(float(f1), 4),
            "Acc": round(float(acc), 4),
        })

    res_df = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    best_name = res_df.loc[0, "Model"]
    best_mdl = fitted[best_name]
    best_pred = np.ravel(best_mdl.predict(X_test)).astype(int)
    best_prob = best_mdl.predict_proba(X_test)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "best_model.joblib"
    joblib.dump({"model": best_mdl, "label_encoder": le, "features": list(X.columns), "target": target}, model_path)

    pred_df = X_test.copy()
    pred_df["y_true"] = y_test
    pred_df["y_pred"] = best_pred
    if is_binary:
        pred_df["y_prob"] = best_prob[:, 1]
    pred_df.to_csv(out / "test_predictions.csv", index=False)

    cm = confusion_matrix(y_test, best_pred)
    cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in le.classes_], columns=[f"Pred_{c}" for c in le.classes_])

    if hasattr(best_mdl, "get_feature_importance"):
        imp = pd.Series(best_mdl.get_feature_importance(), index=X.columns).sort_values(ascending=False).head(8)
    elif hasattr(best_mdl, "feature_importances_"):
        imp = pd.Series(best_mdl.feature_importances_, index=X.columns).sort_values(ascending=False).head(8)
    elif hasattr(best_mdl, "named_steps"):
        lr = best_mdl.named_steps[list(best_mdl.named_steps.keys())[-1]]
        coef = np.abs(lr.coef_)
        imp = pd.Series(coef.mean(axis=0) if coef.ndim > 1 else coef, index=X.columns).sort_values(ascending=False).head(8)
    else:
        imp = pd.Series(dtype=float)

    roc_or_bar = svg_line_chart(roc_series, x_label="False Positive Rate", diag_line=True) if is_binary else svg_barv(res_df["Model"].tolist(), res_df["F1"].tolist(), fmt_str="{:.3f}")
    imp_html = svg_barh(imp.index.tolist(), imp.values.tolist(), colors=[TOKENS["primary"]] * len(imp), fmt_str="{:.3f}") if not imp.empty else '<div class="empty">Run /shap</div>'

    warnings = []
    pos_rate = float(np.bincount(y).min() / len(y))
    if pos_rate < 0.15:
        warnings.append(f"Imbalanced classes (minority={pos_rate * 100:.1f}%): prefer PR-AUC & threshold tuning.")

    kpis = [
        ("Best Model", best_name, TOKENS["success"]),
        ("Test ROC-AUC", f"{res_df.loc[0, 'ROC_AUC']:.4f}", TOKENS["primary"]),
        ("Test F1", f"{res_df.loc[0, 'F1']:.4f}", TOKENS["purple"]),
        ("CV Score", f"{res_df.loc[0, 'CV_Mean']:.4f}", TOKENS["teal"]),
        ("Train / Test", f"{len(X_train):,} / {len(X_test):,}", TOKENS["muted"]),
    ]

    panels = [
        {"title": "Model Leaderboard (XGBoost · CatBoost · LightGBM · RF · LogReg)", "badge": f"{cv_folds}-Fold CV", "html": html_table(res_df, highlight_row=0)},
        {"title": "ROC Curves Comparison" if is_binary else "Macro F1 by Model", "badge": "Holdout Test", "html": roc_or_bar},
        {"title": f"Confusion Matrix ({best_name})", "badge": f"n={len(y_test)}", "html": html_heatmap(cm_df, fmt_str=".0f", diverging=False)},
        {"title": f"Top Features ({best_name})", "badge": "Importance / |Coef|", "html": imp_html},
    ]

    png_path, html_path = render_dashboard(
        title="Classification Benchmark & Diagnostics",
        subtitle=f"Dataset: {Path(data_path).name} | Target: {target} ({n_classes} classes, {meta['n_features']} features)",
        kpis=kpis,
        panels=panels,
        out_dir=out_dir,
        file_stem="classification_dashboard",
        fmt=fmt,
    )
    emit_result(
        skill="ds-classification",
        out_dir=out_dir,
        kpis={"best": best_name, "roc_auc": res_df.loc[0, "ROC_AUC"], "f1": res_df.loc[0, "F1"], "model": str(model_path)},
        visuals=[png_path, html_path],
        warnings=warnings,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--format", default="landscape", choices=["landscape", "portrait"])
    p.add_argument("--theme", default=None, choices=["light", "dark"])
    p.add_argument("--out", default="outputs/ds-classification")
    args = p.parse_args()
    set_theme(args.theme)
    run_classification(args.data, args.target, args.cv, args.format, args.out)
