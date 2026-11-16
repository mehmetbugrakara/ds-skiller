"""Multi-GBDT Hyperparameter Tuning (XGBoost, CatBoost, LightGBM) & Learning Curve diagnosis."""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import f1_score, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ds_core import set_theme, TOKENS, emit_result, html_table, load_dataset, prepare_features, render_dashboard, svg_barh, svg_line_chart


def _make_gbdt(algo: str, is_clf: bool, is_bin: bool, params: dict):
    if algo == "xgboost":
        if is_clf:
            return XGBClassifier(
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample"],
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
                n_jobs=-1,
            )
        return XGBRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample"],
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        )
    if algo == "catboost":
        depth = min(max(int(params["max_depth"]), 4), 8)
        if is_clf:
            return CatBoostClassifier(
                iterations=params["n_estimators"],
                learning_rate=params["learning_rate"],
                depth=depth,
                l2_leaf_reg=params["l2"],
                random_seed=42,
                verbose=0,
                allow_writing_files=False,
            )
        return CatBoostRegressor(
            iterations=params["n_estimators"],
            learning_rate=params["learning_rate"],
            depth=depth,
            l2_leaf_reg=params["l2"],
            random_seed=42,
            verbose=0,
            allow_writing_files=False,
        )
    # lightgbm
    if is_clf:
        return LGBMClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            num_leaves=params["num_leaves"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample"],
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
    return LGBMRegressor(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        num_leaves=params["num_leaves"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample"],
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )


def _eval_metric(mdl, X_part: pd.DataFrame, y_part: np.ndarray, is_clf: bool, is_bin: bool) -> float:
    if is_clf:
        if is_bin:
            return float(roc_auc_score(y_part, mdl.predict_proba(X_part)[:, 1]))
        pred = np.ravel(mdl.predict(X_part)).astype(int)
        return float(f1_score(y_part, pred, average="macro"))
    pred = np.ravel(mdl.predict(X_part))
    return float(r2_score(y_part, pred))


def _explicit_cv(algo: str, params: dict, X: pd.DataFrame, y: np.ndarray, splitter, is_clf: bool, is_bin: bool) -> tuple[float, float]:
    tr_scores, va_scores = [], []
    for tr_idx, va_idx in splitter.split(X, y if is_clf else None):
        mdl = _make_gbdt(algo, is_clf, is_bin, params)
        mdl.fit(X.iloc[tr_idx], y[tr_idx])
        tr_scores.append(_eval_metric(mdl, X.iloc[tr_idx], y[tr_idx], is_clf, is_bin))
        va_scores.append(_eval_metric(mdl, X.iloc[va_idx], y[va_idx], is_clf, is_bin))
    return float(np.mean(tr_scores)), float(np.mean(va_scores))


def run_tune(data_path: str, target: str, algo_choice: str, n_iter: int, cv: int, fmt: str, out_dir: str) -> None:
    df = load_dataset(data_path)
    X, y_raw, _ = prepare_features(df, target=target)

    is_clf = (not pd.api.types.is_numeric_dtype(y_raw)) or y_raw.nunique() <= 15
    if is_clf:
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
        is_bin = len(le.classes_) == 2
        scoring = "roc_auc" if is_bin else "f1_macro"
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    else:
        le = None
        y = pd.to_numeric(y_raw, errors="coerce").fillna(0).to_numpy()
        is_bin = False
        scoring = "r2"
        splitter = KFold(n_splits=cv, shuffle=True, random_state=42)

    algos = ["xgboost", "catboost", "lightgbm"] if algo_choice == "all" else [algo_choice]
    rng = np.random.default_rng(42)

    trials = []
    for t_idx in range(n_iter):
        algo = algos[t_idx % len(algos)]
        params = {
            "n_estimators": int(rng.choice([120, 200, 300, 400])),
            "learning_rate": float(rng.choice([0.03, 0.05, 0.08, 0.12])),
            "max_depth": int(rng.choice([4, 6, 8])),
            "num_leaves": int(rng.choice([15, 31, 48])),
            "subsample": float(rng.choice([0.75, 0.85, 1.0])),
            "colsample": float(rng.choice([0.75, 0.85, 1.0])),
            "l2": float(rng.choice([1.0, 3.0, 6.0])),
        }
        tr_sc, va_sc = _explicit_cv(algo, params, X, y, splitter, is_clf, is_bin)
        trials.append({
            "Trial": t_idx + 1,
            "Algo": algo.upper() if algo == "xgboost" else algo.title(),
            "CV_Score": round(va_sc, 4),
            "Train_Score": round(tr_sc, 4),
            "n_est": params["n_estimators"],
            "lr": params["learning_rate"],
            "depth": params["max_depth"],
            "_algo_key": algo,
            "_params": params,
        })

    cv_res = pd.DataFrame(trials).sort_values("CV_Score", ascending=False).reset_index(drop=True)
    cv_res.insert(0, "Rank", np.arange(1, len(cv_res) + 1))

    best_row = cv_res.iloc[0]
    best_algo_key = str(best_row["_algo_key"])
    best_algo_lbl = str(best_row["Algo"])
    best_params = dict(best_row["_params"])
    best_score = float(best_row["CV_Score"])
    best_train = float(best_row["Train_Score"])
    gap = best_train - best_score
    diagnosis = "High Variance (Overfit)" if gap > 0.12 else ("Underfit" if best_score < 0.65 else "Well Balanced")

    # Fit winning model on full dataset
    best_model = _make_gbdt(best_algo_key, is_clf, is_bin, best_params)
    best_model.fit(X, y)

    # Explicit Learning Curve for the winning GBDT model (safe across all 3 frameworks)
    fracs = np.linspace(0.25, 1.0, 5)
    lc_sizes, lc_tr, lc_va = [], [], []
    for frac in fracs:
        tr_fold, va_fold = [], []
        for tr_idx, va_idx in splitter.split(X, y if is_clf else None):
            k_n = max(25, int(len(tr_idx) * frac))
            sub_idx = tr_idx[:k_n]
            if is_clf and len(np.unique(y[sub_idx])) < 2:
                continue
            m = _make_gbdt(best_algo_key, is_clf, is_bin, best_params)
            m.fit(X.iloc[sub_idx], y[sub_idx])
            tr_fold.append(_eval_metric(m, X.iloc[sub_idx], y[sub_idx], is_clf, is_bin))
            va_fold.append(_eval_metric(m, X.iloc[va_idx], y[va_idx], is_clf, is_bin))
        if va_fold:
            lc_sizes.append(int(len(X) * (1 - 1 / cv) * frac))
            lc_tr.append(float(np.mean(tr_fold)))
            lc_va.append(float(np.mean(va_fold)))

    lc_svg = svg_line_chart(
        [
            {"name": f"Train ({best_algo_lbl})", "x": lc_sizes, "y": lc_tr, "color": TOKENS["primary"]},
            {"name": f"CV ({best_algo_lbl})", "x": lc_sizes, "y": lc_va, "color": TOKENS["success"]},
        ],
        x_label="Training Samples",
    )

    top_trials = cv_res.head(8)
    trial_labels = [f"{r['Algo']} (lr={r['lr']}, d={r['depth']})" for _, r in top_trials.iterrows()]
    trials_svg = svg_barh(
        trial_labels,
        top_trials["CV_Score"].tolist(),
        colors=[TOKENS["success"]] + [TOKENS["primary"]] * (len(top_trials) - 1),
        fmt_str="{:.4f}",
    )

    show_df = cv_res[["Rank", "Algo", "CV_Score", "Train_Score", "n_est", "lr", "depth"]].copy()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mdl_path = out / "tuned_model.joblib"
    joblib.dump({"model": best_model, "label_encoder": le, "features": list(X.columns), "target": target}, mdl_path)

    kpis = [
        ("Winner", best_algo_lbl, TOKENS["success"]),
        ("Best CV Score", f"{best_score:.4f} ({scoring})", TOKENS["primary"]),
        ("Train Score", f"{best_train:.4f}", TOKENS["purple"]),
        ("Overfit Gap", f"{gap:+.4f}", TOKENS["danger"] if gap > 0.12 else TOKENS["success"]),
        ("Diagnosis", diagnosis, TOKENS["warning"] if "Variance" in diagnosis or "Underfit" in diagnosis else TOKENS["success"]),
    ]

    panels = [
        {"title": f"Learning Curve ({best_algo_lbl} · Bias vs Variance)", "badge": diagnosis, "html": lc_svg},
        {"title": f"Top {len(top_trials)} GBDT Trials", "badge": scoring, "html": trials_svg},
        {"title": "Hyperparameter Leaderboard (XGBoost · CatBoost · LightGBM)", "badge": mdl_path.name, "html": html_table(show_df, highlight_row=0), "span": 2},
    ]

    png_path, html_path = render_dashboard(
        title="Hyperparameter Tuning & Learning Curve",
        subtitle=f"Dataset: {Path(data_path).name} | Target: {target} | Frameworks: {', '.join(algos).upper()}",
        kpis=kpis,
        panels=panels,
        out_dir=out_dir,
        file_stem="tune_dashboard",
        fmt=fmt,
    )
    emit_result(
        skill="ds-tune",
        out_dir=out_dir,
        kpis={"winner": best_algo_lbl, "best_cv": round(best_score, 4), "gap": round(gap, 4), "diagnosis": diagnosis, "model": str(mdl_path)},
        visuals=[png_path, html_path],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--algo", default="all", choices=["all", "xgboost", "lightgbm", "catboost"])
    p.add_argument("--trials", type=int, default=12)
    p.add_argument("--cv", type=int, default=4)
    p.add_argument("--format", default="landscape", choices=["landscape", "portrait"])
    p.add_argument("--theme", default=None, choices=["light", "dark"])
    p.add_argument("--out", default="outputs/ds-tune")
    args = p.parse_args()
    set_theme(args.theme)
    run_tune(args.data, args.target, args.algo, args.trials, args.cv, args.format, args.out)
