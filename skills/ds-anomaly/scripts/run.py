"""Unsupervised Anomaly Detection (Isolation Forest + LOF + Robust Z-Score)."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ds_core import set_theme, TOKENS, emit_result, html_table, load_dataset, render_dashboard, svg_barh, svg_scatter, split_feature_types


def run_anomaly(data_path: str, target: str | None, contamination: float, fmt: str, out_dir: str) -> None:
    df = load_dataset(data_path)
    types = split_feature_types(df, target=target)
    num_cols = types["numeric"]
    if len(num_cols) < 2:
        raise ValueError("Need at least 2 numeric features for anomaly detection.")

    X = df[num_cols].copy()
    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())
    X_scaled = RobustScaler().fit_transform(X)

    # 1. Isolation Forest
    iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    iso_flag = (iso.fit_predict(X_scaled) == -1).astype(int)
    iso_score = -iso.decision_function(X_scaled)

    # 2. Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=min(20, max(5, len(X) // 20)), contamination=contamination)
    lof_flag = (lof.fit_predict(X_scaled) == -1).astype(int)

    # 3. Robust Z-score (>3.5 MAD on any feature)
    rz_flag = (np.abs(X_scaled) > 3.5).any(axis=1).astype(int)

    votes = iso_flag + lof_flag + rz_flag
    consensus = votes >= 2

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    flagged = df.copy()
    flagged["anomaly_votes"] = votes
    flagged["iso_score"] = iso_score.round(4)
    flagged_csv = out / "anomalies_flagged.csv"
    flagged[consensus].sort_values("iso_score", ascending=False).to_csv(flagged_csv, index=False)

    # PCA 2D projection
    coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    norm_mask = ~consensus
    anom_mask = consensus

    scatter_svg = svg_scatter(
        [
            {"name": f"Normal (n={norm_mask.sum()})", "x": coords[norm_mask, 0].tolist(), "y": coords[norm_mask, 1].tolist(), "color": TOKENS["primary"]},
            {"name": f"Consensus Anomaly (n={anom_mask.sum()})", "x": coords[anom_mask, 0].tolist(), "y": coords[anom_mask, 1].tolist(), "color": TOKENS["danger"]},
        ],
        x_label="PC1 (Robust Scaled)",
    )

    # Feature driver of anomalies (mean |z| difference between anomalies and normals)
    if anom_mask.any():
        z_diff = np.abs(X_scaled[anom_mask].mean(axis=0) - X_scaled[norm_mask].mean(axis=0))
    else:
        z_diff = np.zeros(len(num_cols))
    drv_df = pd.DataFrame({"Feature": num_cols, "Z_Shift": z_diff}).sort_values("Z_Shift", ascending=False).head(8)
    drv_svg = svg_barh(drv_df["Feature"].tolist(), drv_df["Z_Shift"].tolist(), colors=[TOKENS["warning"]] * len(drv_df), fmt_str="{:.2f}")

    top_rows = flagged.sort_values(["anomaly_votes", "iso_score"], ascending=False).head(7)
    show_cols = [c for c in (num_cols[:4] + ["anomaly_votes", "iso_score"]) if c in top_rows.columns]
    top_tbl = top_rows[show_cols].copy()
    for c in num_cols[:4]:
        if c in top_tbl.columns:
            top_tbl[c] = top_tbl[c].round(2)

    kpis = [
        ("Consensus Anomalies", f"{int(consensus.sum())} ({consensus.mean() * 100:.1f}%)", TOKENS["danger"]),
        ("IsoForest Flagged", f"{int(iso_flag.sum())}", TOKENS["warning"]),
        ("LOF Flagged", f"{int(lof_flag.sum())}", TOKENS["purple"]),
        ("Robust Z>3.5", f"{int(rz_flag.sum())}", TOKENS["primary"]),
        ("Top Driver", drv_df.iloc[0]["Feature"][:16], TOKENS["success"]),
    ]

    panels = [
        {"title": "2D PCA Anomaly Map (Consensus >= 2/3 Models)", "badge": f"n={len(df):,}", "html": scatter_svg},
        {"title": "Features Driving Anomalies (Mean |Z| Shift)", "badge": "RobustScaler", "html": drv_svg},
        {"title": "Top Flagged Anomalous Records", "badge": flagged_csv.name, "html": html_table(top_tbl, highlight_row=0), "span": 2},
    ]

    png_path, html_path = render_dashboard(
        title="Consensus Anomaly & Outlier Detection",
        subtitle=f"Dataset: {Path(data_path).name} | Models: IsolationForest + LOF + Robust Z-Score",
        kpis=kpis,
        panels=panels,
        out_dir=out_dir,
        file_stem="anomaly_dashboard",
        fmt=fmt,
    )
    emit_result(
        skill="ds-anomaly",
        out_dir=out_dir,
        kpis={"consensus_anomalies": int(consensus.sum()), "pct": round(float(consensus.mean() * 100), 2), "csv": str(flagged_csv)},
        visuals=[png_path, html_path],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--target", default=None)
    p.add_argument("--contamination", type=float, default=0.05)
    p.add_argument("--format", default="landscape", choices=["landscape", "portrait"])
    p.add_argument("--theme", default=None, choices=["light", "dark"])
    p.add_argument("--out", default="outputs/ds-anomaly")
    args = p.parse_args()
    set_theme(args.theme)
    run_anomaly(args.data, args.target, args.contamination, args.format, args.out)
