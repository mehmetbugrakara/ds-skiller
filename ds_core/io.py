"""Dataset loading and feature-type inference."""

from pathlib import Path
import numpy as np
import pandas as pd


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load CSV, TSV, Parquet, Feather, Excel, or JSON into a DataFrame."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    ext = p.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(p)
    if ext == ".tsv":
        return pd.read_csv(p, sep="\t")
    if ext == ".parquet":
        return pd.read_parquet(p)
    if ext == ".feather":
        return pd.read_feather(p)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    if ext == ".json":
        return pd.read_json(p)
    raise ValueError(f"Unsupported file format: {ext}")


def split_feature_types(df: pd.DataFrame, target: str | None = None) -> dict[str, list[str]]:
    """Classify columns into numeric, categorical, datetime, constant, and id_like."""
    cols = [c for c in df.columns if c != target]
    numeric, categorical, datetime_cols, constant, id_like = [], [], [], [], []
    n = len(df)

    for c in cols:
        s = df[c]
        if s.nunique(dropna=True) <= 1:
            constant.append(c)
            continue
        if pd.api.types.is_datetime64_any_dtype(s):
            datetime_cols.append(c)
            continue
        if pd.api.types.is_numeric_dtype(s):
            if n > 50 and s.nunique() == n and ("id" in c.lower() or c.lower().endswith("_id")):
                id_like.append(c)
            else:
                numeric.append(c)
            continue
        # Try parsing object as datetime if name suggests date/time
        if any(k in c.lower() for k in ("date", "time", "timestamp", "created_at")):
            parsed = pd.to_datetime(s, errors="coerce")
            if parsed.notna().mean() > 0.8:
                datetime_cols.append(c)
                continue
        if n > 50 and s.nunique() / max(n, 1) > 0.95 and ("id" in c.lower() or s.nunique() == n):
            id_like.append(c)
        else:
            categorical.append(c)

    return {
        "numeric": numeric,
        "categorical": categorical,
        "datetime": datetime_cols,
        "constant": constant,
        "id_like": id_like,
    }


def prepare_features(
    df: pd.DataFrame,
    target: str,
    max_ohe_cardinality: int = 12,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Clean, impute, and encode features for tabular ML without leakage."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in dataset.")

    work = df.dropna(subset=[target]).copy()
    y = work[target]
    types = split_feature_types(work, target=target)
    drop_cols = types["constant"] + types["id_like"] + types["datetime"]
    X_raw = work.drop(columns=[target] + drop_cols, errors="ignore").copy()

    num_cols = [c for c in types["numeric"] if c in X_raw.columns]
    cat_cols = [c for c in types["categorical"] if c in X_raw.columns]

    for c in num_cols:
        if X_raw[c].isna().any():
            X_raw[c] = X_raw[c].fillna(X_raw[c].median())

    low_card = [c for c in cat_cols if X_raw[c].nunique(dropna=True) <= max_ohe_cardinality]
    high_card = [c for c in cat_cols if X_raw[c].nunique(dropna=True) > max_ohe_cardinality]

    for c in cat_cols:
        X_raw[c] = X_raw[c].fillna("__MISSING__").astype(str)

    # Frequency encode high-cardinality categoricals
    for c in high_card:
        freq = X_raw[c].value_counts(normalize=True)
        X_raw[c] = X_raw[c].map(freq).astype(float)

    if low_card:
        X_enc = pd.get_dummies(X_raw, columns=low_card, drop_first=True, dtype=float)
    else:
        X_enc = X_raw

    # Sanitize column names for LightGBM compatibility
    X_enc.columns = [
        str(c).replace("[", "_").replace("]", "_").replace("<", "_").replace(" ", "_")
        for c in X_enc.columns
    ]
    meta = {
        "dropped": drop_cols,
        "numeric": num_cols,
        "ohe": low_card,
        "freq_encoded": high_card,
        "n_rows": len(X_enc),
        "n_features": X_enc.shape[1],
    }
    return X_enc, y, meta


def prepare_inference_features(
    df: pd.DataFrame,
    features: list[str],
    max_ohe_cardinality: int = 12,
) -> tuple[pd.DataFrame, dict]:
    """Encode new data the same way as prepare_features, then align to the trained feature list.

    One-hot is done without drop_first and reindexed, so a batch holding only one category
    still maps correctly. Frequency encoding is recomputed on the batch (approximation).
    """
    def _san(c: str) -> str:
        return str(c).replace("[", "_").replace("]", "_").replace("<", "_").replace(" ", "_")

    work = df.copy()
    types = split_feature_types(work)
    X_raw = work.drop(columns=types["datetime"], errors="ignore").copy()
    for c in X_raw.columns:
        if pd.api.types.is_numeric_dtype(X_raw[c]):
            if X_raw[c].isna().any():
                X_raw[c] = X_raw[c].fillna(X_raw[c].median())
        else:
            X_raw[c] = X_raw[c].fillna("__MISSING__").astype(str)

    feat_set = set(features)
    ohe_cols, freq_cols = [], []
    for c in X_raw.columns:
        if pd.api.types.is_numeric_dtype(X_raw[c]):
            continue
        if _san(c) in feat_set:
            freq_cols.append(c)
        elif any(f.startswith(_san(c) + "_") for f in features):
            ohe_cols.append(c)
    for c in freq_cols:
        X_raw[c] = X_raw[c].map(X_raw[c].value_counts(normalize=True)).astype(float)
    if ohe_cols:
        X_raw = pd.get_dummies(X_raw, columns=ohe_cols, drop_first=False, dtype=float)
    X_raw.columns = [_san(c) for c in X_raw.columns]
    missing = [f for f in features if f not in X_raw.columns]
    num_missing = [f for f in missing if not any(f.startswith(_san(c) + "_") for c in ohe_cols)]
    X = X_raw.reindex(columns=features, fill_value=0.0)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    meta = {"missing_features": num_missing, "n_rows": len(X), "n_features": len(features)}
    return X, meta
