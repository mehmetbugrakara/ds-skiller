"""Regression benchmark (XGBoost, CatBoost, LightGBM, RandomForest, Ridge)."""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ds_core import set_theme
from ds_core import (
    TOKENS,
    emit_result,
    html_table,
    load_dataset,
    prepare_features,
    render_dashboard,
    svg_barh,
    svg_line_chart,
    svg_scatter,
)


def _cv_r2_explicit(factory, X: pd.DataFrame, y: pd.Series, kf: KFold) -> float:
    """Explicit KFold loop (avoids sklearn>=1.6 __sklearn_tags__ issues on CatBoost)."""
    scores = []
    for tr_idx, va_idx in kf.split(X):
        m = factory()
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = np.ravel(m.predict(X.iloc[va_idx]))
        scores.append(r2_score(y.iloc[va_idx], pred))
    return float(np.mean(scores))


def run_regression(data_path: str, target: str, cv_folds: int, fmt: str, out_dir: str) -> None:
    df = load_dataset(data_path)
    X, y_raw, meta = prepare_features(df, target=target)
    y = pd.to_numeric(y_raw, errors="coerce")
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    factories = {
        "XGBoost": lambda: XGBRegressor(
            n_estimators=250, learning_rate=0.05, max_depth=6, random_state=42, verbosity=0, n_jobs=-1,
        ),
        "CatBoost": lambda: CatBoostRegressor(
            iterations=250, learning_rate=0.05, depth=6, random_seed=42, verbose=0, allow_writing_files=False,
        ),
        "LightGBM": lambda: LGBMRegressor(
            n_estimators=250, learning_rate=0.05, random_state=42, verbose=-1, n_jobs=-1,
        ),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1,
        ),
        "Ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    }

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results, fitted = [], {}

    for name, make_mdl in factories.items():
        cv_r2 = _cv_r2_explicit(make_mdl, X_train, y_train, kf)
        mdl = make_mdl()
        mdl.fit(X_train, y_train)
        fitted[name] = mdl
        y_pred = np.ravel(mdl.predict(X_test))
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        nonzero = y_test != 0
        mape = mean_absolute_percentage_error(y_test[nonzero], y_pred[nonzero]) * 100 if nonzero.any() else np.nan
        results.append({
            "Model": name,
            "CV_R2": round(cv_r2, 4),
            "R2": round(float(r2), 4),
            "RMSE": round(float(rmse), 2),
            "MAE": round(float(mae), 2),
            "MAPE%": round(float(mape), 2) if not np.isnan(mape) else "-",
        })

    res_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
    best_name = res_df.loc[0, "Model"]
    best_mdl = fitted[best_name]
    best_pred = np.ravel(best_mdl.predict(X_test))
    residuals = y_test.values - best_pred

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "best_model.joblib"
    joblib.dump({"model": best_mdl, "features": list(X.columns), "target": target}, model_path)

    pred_df = X_test.copy()
    pred_df["y_true"] = y_test.values
    pred_df["y_pred"] = best_pred
    pred_df.to_csv(out / "test_predictions.csv", index=False)

    r_cnt, r_edges = np.histogram(residuals, bins=18)
    r_centers = ((r_edges[:-1] + r_edges[1:]) / 2).tolist()
    resid_svg = svg_line_chart(
        [{"name": "Residual Density", "x": r_centers, "y": r_cnt.tolist(), "color": TOKENS["warning"], "fill": True}],
        x_label="Residual (Actual - Predicted)",
    )

    scatter_svg = svg_scatter(
        [{"name": best_name, "x": y_test.tolist(), "y": best_pred.tolist(), "color": TOKENS["primary"]}],
        x_label="Actual", diag_line=True,
    )

    if hasattr(best_mdl, "get_feature_importance"):
        imp = pd.Series(best_mdl.get_feature_importance(), index=X.columns).sort_values(ascending=False).head(8)
    elif hasattr(best_mdl, "feature_importances_"):
        imp = pd.Series(best_mdl.feature_importances_, index=X.columns).sort_values(ascending=False).head(8)
    elif hasattr(best_mdl, "named_steps"):
        ridge = best_mdl.named_steps[list(best_mdl.named_steps.keys())[-1]]
        imp = pd.Series(np.abs(ridge.coef_), index=X.columns).sort_values(ascending=False).head(8)
    else:
        imp = pd.Series(dtype=float)

    imp_html = svg_barh(imp.index.tolist(), imp.values.tolist(), colors=[TOKENS["purple"]] * len(imp), fmt_str="{:.2f}") if not imp.empty else '<div class="empty">N/A</div>'

    warnings = ["Residual skew > 1.0: consider log-transforming target."] if abs(float(pd.Series(residuals).skew())) > 1.0 else []
    kpis = [
        ("Best Model", best_name, TOKENS["success"]),
        ("Test R2", f"{res_df.loc[0, 'R2']:.4f}", TOKENS["primary"]),
        ("Test RMSE", f"{res_df.loc[0, 'RMSE']:,}", TOKENS["purple"]),
        ("Test MAE", f"{res_df.loc[0, 'MAE']:,}", TOKENS["teal"]),
        ("Train / Test", f"{len(X_train):,} / {len(X_test):,}", TOKENS["muted"]),
    ]

    panels = [
        {"title": "Model Leaderboard (XGBoost · CatBoost · LightGBM · RF · Ridge)", "badge": f"{cv_folds}-Fold CV", "html": html_table(res_df, highlight_row=0)},
        {"title": f"Actual vs Predicted ({best_name})", "badge": "Ideal y=x", "html": scatter_svg},
        {"title": f"Residual Distribution (mean={residuals.mean():.1f})", "badge": "Error Spread", "html": resid_svg},
        {"title": f"Top Features ({best_name})", "badge": "Importance / |Coef|", "html": imp_html},
    ]

    png_path, html_path = render_dashboard(
        title="Regression Benchmark & Residual Diagnostics",
        subtitle=f"Dataset: {Path(data_path).name} | Target: {target} ({meta['n_features']} features)",
        kpis=kpis,
        panels=panels,
        out_dir=out_dir,
        file_stem="regression_dashboard",
        fmt=fmt,
    )
    emit_result(
        skill="ds-regression",
        out_dir=out_dir,
        kpis={"best": best_name, "r2": res_df.loc[0, "R2"], "rmse": res_df.loc[0, "RMSE"], "model": str(model_path)},
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
    p.add_argument("--out", default="outputs/ds-regression")
    args = p.parse_args()
    set_theme(args.theme)
    run_regression(args.data, args.target, args.cv, args.format, args.out)
