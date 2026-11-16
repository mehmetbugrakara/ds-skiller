"""Data cleaning & preprocessing pipeline with Impeccable HTML/PNG dashboard."""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ds_core import set_theme, TOKENS, emit_result, html_table, load_dataset, render_dashboard, split_feature_types, svg_barv


def run_clean(data_path: str, target: str | None, clip_outliers: bool, fmt: str, out_dir: str) -> None:
    df_raw = load_dataset(data_path)
    df = df_raw.copy()

    raw_rows, raw_cols = df_raw.shape
    raw_nulls = int(df_raw.isna().sum().sum())

    dups_removed = int(df.duplicated().sum())
    if dups_removed > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    target_nulls_dropped = 0
    if target and target in df.columns:
        before_t = len(df)
        df = df.dropna(subset=[target]).reset_index(drop=True)
        target_nulls_dropped = before_t - len(df)

    types = split_feature_types(df, target=target)
    drop_cols = types["constant"] + types["id_like"]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    imputed_cols = 0
    clipped_vals = 0
    for c in types["numeric"]:
        if c not in df.columns:
            continue
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
            imputed_cols += 1
        if clip_outliers:
            q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                mask = (df[c] < lo) | (df[c] > hi)
                clipped_vals += int(mask.sum())
                df[c] = df[c].clip(lo, hi)

    for c in types["categorical"]:
        if c not in df.columns:
            continue
        if df[c].isna().any():
            mode_val = df[c].mode(dropna=True)
            df[c] = df[c].fillna(mode_val.iloc[0] if not mode_val.empty else "MISSING")
            imputed_cols += 1

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cleaned_csv = out_path / f"{Path(data_path).stem}_cleaned.csv"
    df.to_csv(cleaned_csv, index=False)

    kpis = [
        ("Rows", f"{raw_rows:,} -> {len(df):,}", TOKENS["primary"]),
        ("Cols", f"{raw_cols} -> {df.shape[1]}", TOKENS["purple"]),
        ("Nulls Fixed", f"{raw_nulls:,} -> {int(df.isna().sum().sum())}", TOKENS["success"]),
        ("Dups Removed", f"{dups_removed:,}", TOKENS["warning"] if dups_removed else TOKENS["success"]),
        ("Outliers Clipped", f"{clipped_vals:,}" if clip_outliers else "Off", TOKENS["teal"]),
    ]

    bar_svg = svg_barv(
        ["Raw Nulls", "Cleaned Nulls", "Dups Dropped", "Cols Dropped", "Outliers Fixed"],
        [raw_nulls, int(df.isna().sum().sum()), dups_removed, len(drop_cols), clipped_vals],
        colors=[TOKENS["danger"], TOKENS["success"], TOKENS["warning"], TOKENS["purple"], TOKENS["primary"]],
    )

    log_df = pd.DataFrame([
        {"Step": "Duplicates Dropped", "Result": str(dups_removed), "Status": "PASS"},
        {"Step": "Target Null Rows Dropped", "Result": str(target_nulls_dropped), "Status": "PASS"},
        {"Step": "Constant/ID Cols Dropped", "Result": ", ".join(drop_cols) or "None", "Status": "PASS"},
        {"Step": "Columns Imputed", "Result": str(imputed_cols), "Status": "PASS"},
        {"Step": "IQR Outlier Capping", "Result": f"{clipped_vals} vals" if clip_outliers else "Disabled", "Status": "PASS"},
        {"Step": "Exported CSV", "Result": cleaned_csv.name, "Status": "OK"},
    ])

    panels = [
        {"title": "Cleaning Impact Metrics", "badge": "Before vs After", "html": bar_svg},
        {"title": "Preprocessing Audit Log", "badge": cleaned_csv.name, "html": html_table(log_df, status_col="Status")},
    ]

    png_path, html_path = render_dashboard(
        title="Data Cleaning & Preprocessing Report",
        subtitle=f"Source: {Path(data_path).name} | Output: {cleaned_csv.name}",
        kpis=kpis,
        panels=panels,
        out_dir=out_dir,
        file_stem="clean_dashboard",
        fmt=fmt,
    )
    emit_result(
        skill="ds-clean",
        out_dir=out_dir,
        kpis={"rows": f"{raw_rows}->{len(df)}", "cols": f"{raw_cols}->{df.shape[1]}", "nulls": f"{raw_nulls}->{int(df.isna().sum().sum())}", "cleaned_csv": str(cleaned_csv)},
        visuals=[png_path, html_path],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--target", default=None)
    p.add_argument("--clip-outliers", action="store_true")
    p.add_argument("--format", default="landscape", choices=["landscape", "portrait"])
    p.add_argument("--theme", default=None, choices=["light", "dark"])
    p.add_argument("--out", default="outputs/ds-clean")
    args = p.parse_args()
    set_theme(args.theme)
    run_clean(args.data, args.target, args.clip_outliers, args.format, args.out)
