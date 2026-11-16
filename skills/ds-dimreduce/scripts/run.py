"""PCA & K-Means clustering Impeccable HTML/PNG dashboard."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ds_core import set_theme
from ds_core import (
    TOKENS,
    emit_result,
    html_heatmap,
    load_dataset,
    render_dashboard,
    split_feature_types,
    svg_barh,
    svg_barv,
    svg_scatter,
)


def run_dimreduce(data_path: str, target: str | None, k_override: int | None, fmt: str, out_dir: str) -> None:
    df = load_dataset(data_path)
    types = split_feature_types(df, target=target)
    num_cols = types["numeric"]
    if len(num_cols) < 2:
        raise ValueError("Need at least 2 numeric columns for PCA & clustering.")

    X = df[num_cols].copy()
    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())
    X_scaled = StandardScaler().fit_transform(X)

    n_comp = min(len(num_cols), 8)
    pca = PCA(n_components=n_comp, random_state=42)
    coords = pca.fit_transform(X_scaled)
    cum_var = np.cumsum(pca.explained_variance_ratio_) * 100
    n_90 = int(np.searchsorted(cum_var, 90.0) + 1)

    sil_idx = np.random.default_rng(42).choice(len(X_scaled), size=min(len(X_scaled), 2500), replace=False)
    sil_scores = {}
    if k_override is None:
        for k in range(2, min(8, len(X_scaled))):
            km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
            sil_scores[k] = float(silhouette_score(X_scaled[sil_idx], km.labels_[sil_idx]))
        best_k = max(sil_scores, key=sil_scores.get)
    else:
        best_k = k_override
        km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X_scaled)
        sil_scores[best_k] = float(silhouette_score(X_scaled[sil_idx], km.labels_[sil_idx]))

    labels = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit_predict(X_scaled)
    best_sil = sil_scores[best_k]

    prof = pd.DataFrame(X_scaled, columns=num_cols)
    prof["Cluster"] = [f"C{i}" for i in labels]
    cluster_means = prof.groupby("Cluster").mean().iloc[:, :6]
    cluster_counts = pd.Series(labels).value_counts().sort_index()

    scree_svg = svg_barv(
        [f"PC{i}" for i in range(1, n_comp + 1)],
        (pca.explained_variance_ratio_ * 100).tolist(),
        colors=[TOKENS["primary"]] * n_comp,
        fmt_str="{:.1f}%",
    )

    groups = []
    for c_id in range(best_k):
        mask = labels == c_id
        groups.append({
            "name": f"C{c_id} (n={cluster_counts[c_id]})",
            "x": coords[mask, 0].tolist(),
            "y": coords[mask, 1].tolist(),
            "color": TOKENS["series"][c_id % 6],
        })
    scatter_svg = svg_scatter(groups, x_label=f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")

    loadings = pd.Series(pca.components_[0], index=num_cols)
    top_load = loadings.reindex(loadings.abs().sort_values(ascending=False).head(8).index)
    load_cols = [TOKENS["success"] if v > 0 else TOKENS["danger"] for v in top_load.values]
    load_svg = svg_barh(top_load.index.tolist(), top_load.values.tolist(), colors=load_cols, fmt_str="{:+.3f}")

    kpis = [
        ("PC1+PC2 Var", f"{cum_var[1]:.1f}%", TOKENS["primary"]),
        ("PCs for 90% Var", f"{min(n_90, n_comp)} / {len(num_cols)}", TOKENS["purple"]),
        ("Optimal K", f"{best_k}", TOKENS["success"]),
        ("Silhouette", f"{best_sil:.3f}", TOKENS["warning"] if best_sil < 0.25 else TOKENS["success"]),
        ("Samples", f"{len(X):,}", TOKENS["muted"]),
    ]

    panels = [
        {"title": "PCA Scree Plot (Variance Explained %)", "badge": f"90% @ {min(n_90, n_comp)} PCs", "html": scree_svg},
        {"title": f"2D PCA Cluster Projection (K={best_k})", "badge": f"Sil={best_sil:.3f}", "html": scatter_svg},
        {"title": "Cluster Profile (Standardized Means)", "badge": "Z-Score Heatmap", "html": html_heatmap(cluster_means, fmt_str=".2f", diverging=True)},
        {"title": "Top PC1 Feature Loadings", "badge": "Principal Axis 1", "html": load_svg},
    ]

    png_path, html_path = render_dashboard(
        title="PCA & Unsupervised Segmentation",
        subtitle=f"Dataset: {Path(data_path).name} | Numeric Features: {len(num_cols)}",
        kpis=kpis,
        panels=panels,
        out_dir=out_dir,
        file_stem="pca_cluster_dashboard",
        fmt=fmt,
    )
    emit_result(
        skill="ds-dimreduce",
        out_dir=out_dir,
        kpis={"pc1_pc2_var": round(float(cum_var[1]), 1), "best_k": best_k, "silhouette": round(best_sil, 3)},
        visuals=[png_path, html_path],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--target", default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--format", default="landscape", choices=["landscape", "portrait"])
    p.add_argument("--theme", default=None, choices=["light", "dark"])
    p.add_argument("--out", default="outputs/ds-dimreduce")
    args = p.parse_args()
    set_theme(args.theme)
    run_dimreduce(args.data, args.target, args.k, args.format, args.out)
