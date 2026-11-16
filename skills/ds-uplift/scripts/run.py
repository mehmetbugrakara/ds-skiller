"""Uplift / heterogeneous treatment effect modeling (T-Learner with LightGBM, Qini curve, decile uplift)."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ds_core import TOKENS, emit_result, html_table, load_dataset, prepare_features, render_dashboard, set_theme, svg_barh, svg_barv, svg_line_chart


def _qini(uplift: np.ndarray, y: np.ndarray, t: np.ndarray, n_points: int = 20) -> tuple[list[float], list[float], list[float], float]:
    order = np.argsort(-uplift)
    y, t = y[order], t[order]
    n = len(y)
    xs, qs, rand = [0.0], [0.0], [0.0]
    n_t_all, n_c_all = t.sum(), (1 - t).sum()
    total = y[t == 1].sum() - y[t == 0].sum() * n_t_all / max(n_c_all, 1)
    for k in np.linspace(n / n_points, n, n_points).astype(int):
        yt, tt = y[:k], t[:k]
        n_t, n_c = tt.sum(), (1 - tt).sum()
        q = yt[tt == 1].sum() - (yt[tt == 0].sum() * n_t / n_c if n_c > 0 else 0.0)
        xs.append(k / n)
        qs.append(float(q))
        rand.append(float(total * k / n))
    area = float(np.trapezoid(np.array(qs) - np.array(rand), xs)) if hasattr(np, "trapezoid") else float(np.trapz(np.array(qs) - np.array(rand), xs))
    return xs, qs, rand, area


def run_uplift(data_path: str, treatment: str, outcome: str, fmt: str, out_dir: str) -> None:
    df = load_dataset(data_path)
    for c in (treatment, outcome):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not in dataset.")
    df = df.dropna(subset=[treatment, outcome]).reset_index(drop=True)
    t_vals = sorted(df[treatment].unique().tolist())
    if len(t_vals) != 2:
        raise ValueError(f"Treatment must be binary, found {t_vals[:5]}")
    t = (df[treatment] == t_vals[1]).astype(int).to_numpy()

    y_raw = df[outcome]
    binary = y_raw.nunique() == 2
    y = (y_raw == sorted(y_raw.unique())[1]).astype(int).to_numpy() if binary else pd.to_numeric(y_raw, errors="coerce").fillna(0).to_numpy()

    X, _, meta = prepare_features(df.drop(columns=[treatment]), target=outcome)
    idx_tr, idx_te = train_test_split(np.arange(len(X)), test_size=0.3, random_state=42, stratify=t)

    def make():
        if binary:
            return LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=40, random_state=42, verbose=-1)
        return LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=40, random_state=42, verbose=-1)

    def pred(m, Xp):
        return m.predict_proba(Xp)[:, 1] if binary else m.predict(Xp)

    tr_t = idx_tr[t[idx_tr] == 1]
    tr_c = idx_tr[t[idx_tr] == 0]
    m_t, m_c = make().fit(X.iloc[tr_t], y[tr_t]), make().fit(X.iloc[tr_c], y[tr_c])

    X_te, y_te, t_te = X.iloc[idx_te], y[idx_te], t[idx_te]
    uplift_te = pred(m_t, X_te) - pred(m_c, X_te)

    ate = float(y[t == 1].mean() - y[t == 0].mean())
    xs, qs, rand, qini_area = _qini(uplift_te, y_te, t_te)

    # Decile table: observed uplift per predicted-uplift decile
    dec = pd.qcut(pd.Series(uplift_te).rank(method="first", ascending=False), 10, labels=False) + 1
    rows = []
    for d in range(1, 11):
        m = (dec == d).to_numpy()
        yt, tt = y_te[m], t_te[m]
        obs = (yt[tt == 1].mean() - yt[tt == 0].mean()) if tt.sum() and (1 - tt).sum() else np.nan
        rows.append({"Decile": d, "Pred_Uplift": round(float(uplift_te[m].mean()), 4), "Obs_Uplift": round(float(obs), 4) if not np.isnan(obs) else "-", "n": int(m.sum())})
    dec_df = pd.DataFrame(rows)

    # Score everyone and save
    uplift_all = pred(m_t, X) - pred(m_c, X)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scored = df.copy()
    scored["uplift_score"] = uplift_all.round(5)
    scored_csv = out / "uplift_scores.csv"
    scored.sort_values("uplift_score", ascending=False).to_csv(scored_csv, index=False)

    # Feature drivers of uplift: correlation of features with predicted uplift
    corr = X.apply(lambda c: np.corrcoef(c, uplift_all)[0, 1] if c.std() > 0 else 0.0).fillna(0)
    drv = corr.reindex(corr.abs().sort_values(ascending=False).index).head(8)

    top30 = np.argsort(-uplift_te)[: max(1, int(0.3 * len(uplift_te)))]
    yt, tt = y_te[top30], t_te[top30]
    top30_obs = float(yt[tt == 1].mean() - yt[tt == 0].mean()) if tt.sum() and (1 - tt).sum() else np.nan
    sleeping = float((uplift_all < 0).mean())

    fmt_eff = (lambda v: f"{v * 100:+.2f} pp") if binary else (lambda v: f"{v:+,.3g}")
    k_list = [
        ("ATE (overall)", fmt_eff(ate), TOKENS["primary"]),
        ("Top-30% Uplift", fmt_eff(top30_obs) if not np.isnan(top30_obs) else "-", TOKENS["success"]),
        ("Qini Area", f"{qini_area:,.2f}", TOKENS["purple"]),
        ("Negative Uplift", f"{sleeping * 100:.1f}%", TOKENS["danger"]),
        ("Treated / Control", f"{t.sum():,} / {(1 - t).sum():,}", TOKENS["muted"]),
    ]
    warnings = []
    if qini_area <= 0:
        warnings.append("Qini area <= 0: model ranks no better than random; do not target with it.")
    if sleeping > 0.2:
        warnings.append(f"{sleeping * 100:.0f}% of users have negative predicted uplift: exclude them from the campaign.")
    if abs(t.mean() - 0.5) > 0.3:
        warnings.append(f"Unbalanced assignment (treated={t.mean() * 100:.0f}%): check randomization.")

    panels = [
        {
            "title": "Qini Curve (holdout) vs Random Targeting",
            "badge": "30% holdout",
            "html": svg_line_chart(
                [
                    {"name": "Model", "x": xs, "y": qs, "color": TOKENS["primary"], "fill": True},
                    {"name": "Random", "x": xs, "y": rand, "color": TOKENS["muted"], "dashed": True},
                ],
                x_label="Fraction of population targeted",
            ),
        },
        {
            "title": "Observed Uplift by Predicted Decile",
            "badge": "1 = most persuadable",
            "html": svg_barv([str(d) for d in dec_df["Decile"]], [v if v != "-" else 0.0 for v in dec_df["Obs_Uplift"]], fmt_str="{:.3f}"),
        },
        {
            "title": "Features Correlated with Uplift",
            "badge": "corr(feature, uplift)",
            "html": svg_barh(drv.index.tolist(), drv.values.tolist(), colors=[TOKENS["success"] if v > 0 else TOKENS["danger"] for v in drv.values], fmt_str="{:+.2f}"),
        },
        {"title": "Decile Table", "badge": scored_csv.name, "html": html_table(dec_df, highlight_row=0)},
    ]

    png_path, html_path = render_dashboard(
        title="Uplift Modeling & Targeting (T-Learner)",
        subtitle=f"Dataset: {Path(data_path).name} | Treatment: {treatment}={t_vals[1]} vs {t_vals[0]} | Outcome: {outcome} | {meta['n_features']} features",
        kpis=k_list,
        panels=panels,
        out_dir=out_dir,
        file_stem="uplift_dashboard",
        fmt=fmt,
    )
    emit_result(
        skill="ds-uplift",
        out_dir=out_dir,
        kpis={"ate": round(ate, 5), "top30_uplift": round(top30_obs, 5) if not np.isnan(top30_obs) else None, "qini_area": round(qini_area, 3), "negative_uplift_pct": round(sleeping * 100, 2), "csv": str(scored_csv)},
        visuals=[png_path, html_path],
        warnings=warnings,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Uplift modeling: who should be targeted by a treatment?")
    p.add_argument("--data", required=True)
    p.add_argument("--treatment", required=True, help="Binary treatment / campaign flag column")
    p.add_argument("--outcome", required=True, help="Outcome column (binary or numeric)")
    p.add_argument("--format", default="landscape", choices=["landscape", "portrait"])
    p.add_argument("--theme", default=None, choices=["light", "dark"])
    p.add_argument("--out", default="outputs/ds-uplift")
    args = p.parse_args()
    set_theme(args.theme)
    run_uplift(args.data, args.treatment, args.outcome, args.format, args.out)
