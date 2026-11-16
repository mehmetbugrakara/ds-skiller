"""Text column profiling: length stats, top terms / bigrams, duplicates, and (optional) discriminative terms per class."""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ds_core import TOKENS, emit_result, html_table, load_dataset, render_dashboard, set_theme, svg_barh, svg_barv

STOP = {
    # Turkish
    "ve", "bir", "bu", "da", "de", "çok", "ile", "için", "ama", "gibi", "daha", "ne", "o", "şu", "mi", "mı", "mu", "mü",
    "en", "ki", "ya", "veya", "her", "hiç", "sonra", "kadar", "olarak", "ben", "sen", "biz", "siz", "onlar", "değil",
    # English
    "the", "a", "an", "and", "or", "is", "are", "was", "were", "to", "of", "in", "on", "for", "it", "this", "that",
    "with", "as", "at", "be", "by", "i", "you", "we", "they", "but", "not", "so", "my", "me",
}
TOKEN_RE = r"(?u)\b[^\W\d_]{2,}\b"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).replace("İ", "i").lower()).strip()


def run_text(data_path: str, text_col: str, target: str | None, top_n: int, fmt: str, out_dir: str) -> None:
    df = load_dataset(data_path)
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not in dataset. Available: {list(df.columns)[:12]}")
    raw = df[text_col]
    empty_mask = raw.isna() | (raw.astype(str).str.strip() == "")
    texts = raw[~empty_mask].astype(str).map(_norm)
    if len(texts) < 10:
        raise ValueError("Need at least 10 non-empty texts.")

    n_chars = texts.str.len()
    n_words = texts.str.split().str.len()
    dup_rate = float(texts.duplicated().mean())

    cv = CountVectorizer(token_pattern=TOKEN_RE, stop_words=list(STOP), min_df=2)
    counts = cv.fit_transform(texts)
    vocab = np.array(cv.get_feature_names_out())
    freq = np.asarray(counts.sum(axis=0)).ravel()
    top_idx = freq.argsort()[::-1][:top_n]
    top_terms = pd.Series(freq[top_idx], index=vocab[top_idx])

    bv = CountVectorizer(token_pattern=TOKEN_RE, stop_words=list(STOP), ngram_range=(2, 2), min_df=2)
    try:
        bc = bv.fit_transform(texts)
        bfreq = np.asarray(bc.sum(axis=0)).ravel()
        bvocab = np.array(bv.get_feature_names_out())
        bi = bfreq.argsort()[::-1][:top_n]
        top_bi = pd.Series(bfreq[bi], index=bvocab[bi])
    except ValueError:
        top_bi = pd.Series(dtype=float)

    hist_c, edges = np.histogram(n_words, bins=10)
    len_lbl = [f"{int(edges[i])}" for i in range(len(hist_c))]

    k_list = [
        ("Texts", f"{len(texts):,}", TOKENS["primary"]),
        ("Empty / Missing", f"{empty_mask.mean() * 100:.1f}%", TOKENS["warning"] if empty_mask.mean() > 0.05 else TOKENS["muted"]),
        ("Median Words", f"{n_words.median():.0f}", TOKENS["purple"]),
        ("Vocabulary", f"{len(vocab):,}", TOKENS["teal"]),
        ("Duplicate Texts", f"{dup_rate * 100:.1f}%", TOKENS["danger"] if dup_rate > 0.1 else TOKENS["success"]),
    ]
    kpis = {
        "texts": int(len(texts)),
        "empty_pct": round(float(empty_mask.mean() * 100), 2),
        "median_words": float(n_words.median()),
        "vocab": int(len(vocab)),
        "dup_pct": round(dup_rate * 100, 2),
        "top_terms": top_terms.index[:5].tolist(),
    }
    panels = [
        {"title": f"Top {len(top_terms)} Terms", "badge": "stopwords removed", "html": svg_barh(top_terms.index.tolist(), top_terms.values.astype(float).tolist(), colors=[TOKENS["primary"]] * len(top_terms), fmt_str="{:.0f}")},
        {"title": "Top Bigrams", "badge": "min_df=2", "html": svg_barh(top_bi.index.tolist(), top_bi.values.astype(float).tolist(), colors=[TOKENS["purple"]] * len(top_bi), fmt_str="{:.0f}") if not top_bi.empty else '<div class="empty">No repeated bigrams</div>'},
        {"title": "Text Length Distribution (words)", "badge": f"p95={np.percentile(n_words, 95):.0f}", "html": svg_barv(len_lbl, hist_c.astype(float).tolist())},
    ]
    warnings = []
    if dup_rate > 0.1:
        warnings.append(f"{dup_rate * 100:.0f}% duplicate texts: dedupe before modeling.")

    if target and target in df.columns:
        y = df.loc[texts.index, target].astype(str)
        classes = y.value_counts()
        if 2 <= len(classes) <= 20 and classes.min() >= 5:
            tf = TfidfVectorizer(token_pattern=TOKEN_RE, stop_words=list(STOP), min_df=2, ngram_range=(1, 2), sublinear_tf=True)
            Xt = tf.fit_transform(texts)
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            cv_f1 = float(cross_val_score(clf, Xt, y, cv=min(5, int(classes.min())), scoring="f1_macro").mean())
            clf.fit(Xt, y)
            names = np.array(tf.get_feature_names_out())
            coef = clf.coef_[0] if len(classes) == 2 else clf.coef_[list(clf.classes_).index(classes.index[0])]
            pos_cls = clf.classes_[1] if len(classes) == 2 else classes.index[0]
            neg_cls = clf.classes_[0] if len(classes) == 2 else "others"
            order = coef.argsort()
            sel = np.concatenate([order[:5], order[-5:]])
            disc = pd.Series(coef[sel], index=names[sel])
            k_list.append(("Baseline Macro-F1", f"{cv_f1:.3f}", TOKENS["success"]))
            kpis["tfidf_logreg_f1"] = round(cv_f1, 4)
            panels.append({
                "title": f"Discriminative Terms: {pos_cls} (+) vs {neg_cls} (-)",
                "badge": "TF-IDF LogReg coef",
                "html": svg_barh(disc.index.tolist(), disc.round(3).values.tolist(), colors=[TOKENS["success"] if v > 0 else TOKENS["danger"] for v in disc.values], fmt_str="{:+.2f}"),
            })
            per_cls = pd.DataFrame({"Class": classes.index, "n": classes.values, "Median_Words": [float(n_words[y == c].median()) for c in classes.index]})
            panels.append({"title": "Per-Class Overview", "badge": target, "html": html_table(per_cls)})
        else:
            warnings.append(f"Target '{target}' skipped: needs 2-20 classes with >=5 rows each.")

    samples = pd.DataFrame({"Text": texts.head(6).str.slice(0, 90), "Words": n_words.head(6).values})
    panels.append({"title": "Sample Texts (normalized)", "badge": text_col, "html": html_table(samples), "span": 2})

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"term": vocab, "count": freq}).sort_values("count", ascending=False).to_csv(out / "term_frequencies.csv", index=False)
    kpis["csv"] = str(out / "term_frequencies.csv")

    png_path, html_path = render_dashboard(
        title="Text Profiling & Term Analysis",
        subtitle=f"Dataset: {Path(data_path).name} | Text: {text_col}" + (f" | Target: {target}" if target else ""),
        kpis=k_list,
        panels=panels,
        out_dir=out_dir,
        file_stem="text_dashboard",
        fmt=fmt,
    )
    emit_result(skill="ds-text", out_dir=out_dir, kpis=kpis, visuals=[png_path, html_path], warnings=warnings)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Profile a free-text column (TR/EN).")
    p.add_argument("--data", required=True)
    p.add_argument("--text", required=True, help="Text column")
    p.add_argument("--target", default=None, help="Optional label column for discriminative terms")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--format", default="landscape", choices=["landscape", "portrait"])
    p.add_argument("--theme", default=None, choices=["light", "dark"])
    p.add_argument("--out", default="outputs/ds-text")
    args = p.parse_args()
    set_theme(args.theme)
    run_text(args.data, args.text, args.target, args.top, args.format, args.out)
