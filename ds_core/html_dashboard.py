"""Editorial HTML/SVG dashboard engine (Impeccable + UI-UX Pro Max principles).

Design language
- Warm stone "paper" canvas (default) or refined graphite (DS_SKILLER_THEME=dark).
- One restrained accent; muted, desaturated data palette; semantic colours used sparingly.
- Instrument Serif display title, Geist UI text, Geist Mono tabular figures.
- Hairline borders, generous whitespace, soft elevation, no neon, no gradients-as-decoration.

All colours are CSS custom properties, so the same SVG/HTML renders in either theme.
"""

from __future__ import annotations

import base64
import html as html_lib
import math
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"

# Public tokens: CSS variable references, valid in style attributes for both themes.
TOKENS = {
    "bg": "var(--bg)",
    "card": "var(--card)",
    "border": "var(--line)",
    "text": "var(--ink)",
    "muted": "var(--ink-3)",
    "primary": "var(--c1)",
    "success": "var(--pos)",
    "warning": "var(--warn)",
    "danger": "var(--neg)",
    "purple": "var(--c4)",
    "teal": "var(--c6)",
    "series": ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)", "var(--c6)"],
}

THEMES = {
    "light": {
        "bg": "#F5F4F0", "card": "#FFFFFF", "card-2": "#FAF9F6", "line": "#E8E6E0", "grid": "#EFEDE8",
        "ink": "#18181B", "ink-2": "#52525B", "ink-3": "#8E8B84",
        "c1": "#2E4A7D", "c2": "#5B8A72", "c3": "#C2923F", "c4": "#7B6A9E", "c5": "#B8573F", "c6": "#3F7F86",
        "pos": "#3E7D5A", "warn": "#B5812A", "neg": "#B24A3B",
        "shadow": "0 1px 2px rgba(24,24,27,0.04), 0 8px 24px -12px rgba(24,24,27,0.08)",
    },
    "dark": {
        "bg": "#0D0D0F", "card": "#151518", "card-2": "#1A1A1E", "line": "#26262B", "grid": "#1E1E23",
        "ink": "#EDEDEF", "ink-2": "#A1A1AA", "ink-3": "#6E6E77",
        "c1": "#8FA8D8", "c2": "#8FC1A0", "c3": "#D8B473", "c4": "#B3A1D6", "c5": "#DE8D78", "c6": "#7FB9BE",
        "pos": "#8FC1A0", "warn": "#D8B473", "neg": "#DE8D78",
        "shadow": "0 1px 0 rgba(255,255,255,0.03) inset, 0 12px 32px -16px rgba(0,0,0,0.6)",
    },
}


# ----------------------------------------------------------------------------- helpers

def _esc(val: Any) -> str:
    return html_lib.escape(str(val))


def _fmt_num(v: float) -> str:
    """Human-friendly tick/value formatter (no scientific notation)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    a = abs(v)
    if a >= 1e9:
        return f"{v / 1e9:.1f}B"
    if a >= 1e6:
        return f"{v / 1e6:.1f}M"
    if a >= 1e4:
        return f"{v / 1e3:.0f}k"
    if a >= 1e3:
        return f"{v / 1e3:.1f}k"
    if a >= 100 or float(v).is_integer():
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.2f}" if a >= 0.01 or a == 0 else f"{v:.3f}"


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=10 * mag)
    start = math.floor(lo / step) * step
    end = math.ceil(hi / step) * step
    ticks, t = [], start
    while t <= end + step * 1e-6:
        ticks.append(round(t, 10))
        t += step
    return ticks


def _clean_cell(val: Any) -> str:
    if isinstance(val, (float, np.floating)):
        if math.isnan(val):
            return "–"
        if float(val).is_integer() and abs(val) < 1e15:
            return f"{int(val):,}" if abs(val) >= 1e4 else str(int(val))
        return f"{val:,.4g}" if abs(val) >= 1e4 else (f"{val:.4f}".rstrip("0").rstrip(".") if abs(val) < 1 else f"{val:.2f}")
    if isinstance(val, (int, np.integer)):
        return f"{val:,}" if abs(val) >= 1e4 else str(val)
    return str(val)


def _is_numeric_text(s: str) -> bool:
    t = s.replace(",", "").replace("%", "").replace("+", "").replace("pp", "").strip()
    try:
        float(t)
        return True
    except ValueError:
        return False


def _tone_of(text: str) -> str:
    up = text.upper()
    if any(k in up for k in ("HIGH", "FAIL", "DROP", "CRIT", "INVALID")):
        return "neg"
    if any(k in up for k in ("MOD", "WARN", "CHECK", "REVIEW", "INCONCL")):
        return "warn"
    if any(k in up for k in ("STABLE", "PASS", "KEEP", "SHIP", "OK", "BEST")):
        return "pos"
    return "neutral"


def badge(text: str, tone: str = "neutral") -> str:
    tone = {"success": "pos", "warning": "warn", "danger": "neg", "primary": "accent"}.get(tone, tone)
    return f'<span class="badge badge-{tone}"><i></i>{_esc(text)}</span>'


# ----------------------------------------------------------------------------- tables

def html_table(
    df: pd.DataFrame,
    highlight_row: int | None = None,
    status_col: str | None = None,
    max_rows: int = 10,
) -> str:
    if df.empty:
        return '<div class="empty">No records</div>'
    sub = df.head(max_rows)
    cells = [[_clean_cell(v) for v in row] for row in sub.itertuples(index=False)]
    numeric_cols = [
        all(_is_numeric_text(r[j]) or r[j] in ("-", "–") for r in cells) and col != status_col
        for j, col in enumerate(sub.columns)
    ]
    ths = "".join(
        f'<th class="{"num" if numeric_cols[j] else ""}">{_esc(str(c).replace("_", " "))}</th>'
        for j, c in enumerate(sub.columns)
    )
    trs = []
    for i, row in enumerate(cells):
        cls = ' class="row-best"' if highlight_row is not None and i == highlight_row else ""
        tds = []
        for j, col in enumerate(sub.columns):
            sval = row[j]
            if status_col and col == status_col:
                tds.append(f"<td>{badge(sval, _tone_of(sval))}</td>")
            elif numeric_cols[j]:
                tds.append(f'<td class="num">{_esc(sval)}</td>')
            else:
                tds.append(f"<td>{_esc(sval)}</td>")
        trs.append(f"<tr{cls}>{''.join(tds)}</tr>")
    return f'<div class="tbl-wrap"><table><thead><tr>{ths}</tr></thead><tbody>{"".join(trs)}</tbody></table></div>'


def html_heatmap(df: pd.DataFrame, fmt_str: str = ".2f", diverging: bool = True) -> str:
    if df.empty:
        return '<div class="empty">Empty matrix</div>'
    vals = df.to_numpy(dtype=float)
    vmax = max(abs(np.nanmin(vals)), abs(np.nanmax(vals)), 1e-9) if diverging else max(np.nanmax(vals), 1e-9)

    def cell(v: float) -> str:
        if np.isnan(v):
            return 'style="background:var(--card-2)"'
        inten = min(1.0, abs(v) / vmax)
        base = ("var(--c1)" if v >= 0 else "var(--c5)") if diverging else "var(--c1)"
        pct = 6 + inten * 74
        fg = "#FFFFFF" if inten > 0.55 else "var(--ink)"
        return f'style="background:color-mix(in oklab, {base} {pct:.0f}%, var(--card));color:{fg}"'

    cols = "".join(f"<th>{_esc(str(c).replace('_', ' ')[:12])}</th>" for c in df.columns)
    rows = []
    for lbl, row in df.iterrows():
        tds = [f'<td class="hm-lbl">{_esc(str(lbl).replace("_", " ")[:16])}</td>']
        for v in row.values:
            tds.append(f'<td class="hm-cell" {cell(float(v))}>{format(float(v), fmt_str)}</td>')
        rows.append(f"<tr>{''.join(tds)}</tr>")
    big = len(df) <= 4 and len(df.columns) <= 4
    return f'<div class="tbl-wrap"><table class="hm{" hm-lg" if big else ""}"><thead><tr><th></th>{cols}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


# ----------------------------------------------------------------------------- charts

_SVG_OPEN = '<svg viewBox="0 0 {w} {h}" class="chart" preserveAspectRatio="xMidYMid meet">'


def svg_barh(
    labels: list[str],
    values: list[float],
    colors: list[str] | None = None,
    thresholds: list[tuple[float, str, str]] | None = None,
    fmt_str: str = "{:.3f}",
    width: int = 560,
    height: int = 250,
) -> str:
    if not labels:
        return '<div class="empty">No data</div>'
    n = len(labels)
    pad_l, pad_r, pad_t, pad_b = 150, 68, 34 if thresholds else 18, 26
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    vmin = min(min(values), 0.0)
    vmax = max(max(values), 0.0)
    for tv, _, _ in thresholds or []:
        if tv <= vmax * 1.6:
            vmax = max(vmax, tv)
    ticks = _nice_ticks(vmin, vmax, 5)
    lo, hi = ticks[0], ticks[-1]
    span = max(hi - lo, 1e-9)

    def xp(v: float) -> float:
        return pad_l + (v - lo) / span * pw

    row_h = ph / n
    bar_h = min(12.0, row_h * 0.46)
    out = [_SVG_OPEN.format(w=width, h=height)]
    for t in ticks:
        x = xp(t)
        out.append(f'<line class="grid" x1="{x:.1f}" y1="{pad_t - 6}" x2="{x:.1f}" y2="{height - pad_b}"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{height - 8}" text-anchor="middle">{_fmt_num(t)}</text>')
    if lo < 0 < hi:
        out.append(f'<line class="axis" x1="{xp(0):.1f}" y1="{pad_t - 6}" x2="{xp(0):.1f}" y2="{height - pad_b}"/>')

    for i, (lbl, val) in enumerate(zip(labels, values)):
        cy = pad_t + i * row_h + row_h / 2
        x0, x1 = xp(0), xp(val)
        col = colors[i] if colors and i < len(colors) else "var(--c1)"
        short = str(lbl).replace("_", " ")
        short = short[:22] + ("…" if len(short) > 22 else "")
        out.append(f'<text class="cat" x="{pad_l - 12}" y="{cy + 4:.1f}" text-anchor="end">{_esc(short)}</text>')
        out.append(f'<rect class="track" x="{pad_l}" y="{cy - bar_h / 2:.1f}" width="{pw}" height="{bar_h:.1f}" rx="{bar_h / 2:.1f}"/>')
        out.append(
            f'<rect x="{min(x0, x1):.1f}" y="{cy - bar_h / 2:.1f}" width="{max(abs(x1 - x0), 2):.1f}" '
            f'height="{bar_h:.1f}" rx="{bar_h / 2:.1f}" style="fill:{col}"/>'
        )
        tx = max(x0, x1) + 8
        out.append(f'<text class="val" x="{tx:.1f}" y="{cy + 4:.1f}">{_esc(fmt_str.format(val))}</text>')

    for k, (tv, tcol, tlbl) in enumerate(thresholds or []):
        if lo <= tv <= hi:
            x = xp(tv)
            anchor, dx = ("end", -5) if k % 2 == 0 else ("start", 5)
            out.append(f'<line x1="{x:.1f}" y1="{pad_t - 10}" x2="{x:.1f}" y2="{height - pad_b}" style="stroke:{tcol}" class="thr"/>')
            out.append(f'<text class="thr-lbl" x="{x + dx:.1f}" y="{pad_t - 16}" text-anchor="{anchor}" style="fill:{tcol}">{_esc(tlbl)}</text>')
    out.append("</svg>")
    return "".join(out)


def svg_barv(
    labels: list[str],
    values: list[float],
    colors: list[str] | None = None,
    fmt_str: str = "{:.0f}",
    width: int = 560,
    height: int = 240,
) -> str:
    if not labels:
        return '<div class="empty">No data</div>'
    n = len(labels)
    pad_l, pad_r, pad_t, pad_b = 44, 12, 26, 34
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    ticks = _nice_ticks(min(min(values), 0.0), max(max(values), 0.0) * 1.08, 5)
    lo, hi = ticks[0], ticks[-1]
    span = max(hi - lo, 1e-9)

    def yp(v: float) -> float:
        return pad_t + ph - (v - lo) / span * ph

    col_w = pw / n
    bar_w = min(46.0, col_w * 0.5)
    out = [_SVG_OPEN.format(w=width, h=height)]
    for t in ticks:
        y = yp(t)
        out.append(f'<line class="grid" x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{pad_l - 8}" y="{y + 3.5:.1f}" text-anchor="end">{_fmt_num(t)}</text>')
    out.append(f'<line class="axis" x1="{pad_l}" y1="{yp(0):.1f}" x2="{width - pad_r}" y2="{yp(0):.1f}"/>')
    for i, (lbl, val) in enumerate(zip(labels, values)):
        cx = pad_l + i * col_w + col_w / 2
        y0, y1 = yp(0), yp(val)
        col = colors[i] if colors and i < len(colors) else TOKENS["series"][i % 6]
        top, h = min(y0, y1), max(abs(y1 - y0), 2.0)
        out.append(f'<rect x="{cx - bar_w / 2:.1f}" y="{top:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="4" style="fill:{col}"/>')
        ly = (y1 - 7) if val >= 0 else (y1 + 13)
        out.append(f'<text class="val" x="{cx:.1f}" y="{ly:.1f}" text-anchor="middle">{_esc(fmt_str.format(val))}</text>')
        out.append(f'<text class="cat" x="{cx:.1f}" y="{height - 10}" text-anchor="middle">{_esc(str(lbl).replace("_", " ")[:14])}</text>')
    out.append("</svg>")
    return "".join(out)


def _legend(items: list[tuple[str, str, bool]], x: float, y: float) -> str:
    if not items:
        return ""
    w = 34 + max(len(n[:30]) for n, _, _ in items) * 6.2
    out = [f'<rect class="legend-bg" x="{x - 8:.1f}" y="{y - 10:.1f}" width="{w:.1f}" height="{len(items) * 16 + 6:.1f}" rx="6"/>']
    for i, (name, col, dashed) in enumerate(items):
        yy = y + i * 16
        dash = ' stroke-dasharray="4 3"' if dashed else ""
        out.append(f'<line x1="{x:.1f}" y1="{yy:.1f}" x2="{x + 14:.1f}" y2="{yy:.1f}" style="stroke:{col}" stroke-width="2"{dash} stroke-linecap="round"/>')
        out.append(f'<text class="legend" x="{x + 20:.1f}" y="{yy + 3.5:.1f}">{_esc(name[:30])}</text>')
    return "".join(out)


def svg_line_chart(
    series_list: list[dict],
    x_label: str = "",
    y_label: str = "",
    diag_line: bool = False,
    hline: float | None = None,
    width: int = 560,
    height: int = 250,
) -> str:
    """series_list: [{'name','x','y','color','dashed','fill','band':(lo,hi)}]"""
    xs = [v for s in series_list for v in s["x"]]
    ys = [v for s in series_list for v in s["y"]]
    for s in series_list:
        if s.get("band"):
            ys += list(s["band"][0]) + list(s["band"][1])
    if not xs or not ys:
        return '<div class="empty">No series data</div>'
    xt = _nice_ticks(min(xs), max(xs), 6)
    yt = _nice_ticks(min(ys), max(ys), 5)
    xlo, xhi, ylo, yhi = min(xs), max(xs), yt[0], yt[-1]
    if xhi == xlo:
        xhi = xlo + 1
    pad_l, pad_r, pad_t, pad_b = 46, 14, 16, 40 if x_label else 26
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b

    def sx(v: float) -> float:
        return pad_l + (v - xlo) / (xhi - xlo) * pw

    def sy(v: float) -> float:
        return pad_t + ph - (v - ylo) / max(yhi - ylo, 1e-9) * ph

    uid = abs(hash((width, height, len(xs), round(sum(ys), 4)))) % 10**8
    out = [_SVG_OPEN.format(w=width, h=height), "<defs>"]
    for i, s in enumerate(series_list):
        col = s.get("color", TOKENS["series"][i % 6])
        out.append(
            f'<linearGradient id="g{uid}_{i}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0%" style="stop-color:{col};stop-opacity:0.22"/>'
            f'<stop offset="100%" style="stop-color:{col};stop-opacity:0"/></linearGradient>'
        )
    out.append("</defs>")
    for t in yt:
        out.append(f'<line class="grid" x1="{pad_l}" y1="{sy(t):.1f}" x2="{width - pad_r}" y2="{sy(t):.1f}"/>')
        out.append(f'<text class="tick" x="{pad_l - 8}" y="{sy(t) + 3.5:.1f}" text-anchor="end">{_fmt_num(t)}</text>')
    for t in xt:
        if xlo <= t <= xhi:
            out.append(f'<text class="tick" x="{sx(t):.1f}" y="{pad_t + ph + 16}" text-anchor="middle">{_fmt_num(t)}</text>')
    out.append(f'<line class="axis" x1="{pad_l}" y1="{pad_t + ph}" x2="{width - pad_r}" y2="{pad_t + ph}"/>')
    if diag_line:
        a, b = max(xlo, ylo), min(xhi, yhi)
        out.append(f'<line class="ref" x1="{sx(a):.1f}" y1="{sy(a):.1f}" x2="{sx(b):.1f}" y2="{sy(b):.1f}"/>')
    if hline is not None and ylo <= hline <= yhi:
        out.append(f'<line class="ref" x1="{pad_l}" y1="{sy(hline):.1f}" x2="{width - pad_r}" y2="{sy(hline):.1f}"/>')

    legend = []
    for i, s in enumerate(series_list):
        col = s.get("color", TOKENS["series"][i % 6])
        pts = [(sx(x), sy(y)) for x, y in zip(s["x"], s["y"])]
        if len(pts) < 2:
            continue
        if s.get("band"):
            lo_b, hi_b = s["band"]
            poly = " ".join(f"{sx(x):.1f},{sy(h):.1f}" for x, h in zip(s["x"], hi_b))
            poly += " " + " ".join(f"{sx(x):.1f},{sy(l):.1f}" for x, l in reversed(list(zip(s["x"], lo_b))))
            out.append(f'<polygon points="{poly}" style="fill:{col}" fill-opacity="0.12"/>')
        if s.get("fill"):
            area = f"{pts[0][0]:.1f},{pad_t + ph:.1f} " + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + f" {pts[-1][0]:.1f},{pad_t + ph:.1f}"
            out.append(f'<polygon points="{area}" fill="url(#g{uid}_{i})"/>')
        d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        dash = ' stroke-dasharray="5 4"' if s.get("dashed") else ""
        out.append(f'<path d="{d}" fill="none" style="stroke:{col}" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"{dash}/>')
        lx, ly = pts[-1]
        if not s.get("dashed"):
            out.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="3" style="fill:{col}" class="end-dot"/>')
        legend.append((s["name"], col, bool(s.get("dashed"))))
    out.append(_legend(legend, pad_l + 12, pad_t + 8))
    if x_label:
        out.append(f'<text class="axis-lbl" x="{pad_l + pw / 2:.1f}" y="{height - 4}" text-anchor="middle">{_esc(x_label)}</text>')
    out.append("</svg>")
    return "".join(out)


def svg_scatter(
    groups: list[dict],
    x_label: str = "",
    y_label: str = "",
    diag_line: bool = False,
    hline: float | None = None,
    width: int = 560,
    height: int = 250,
) -> str:
    xs = [v for g in groups for v in g["x"]]
    ys = [v for g in groups for v in g["y"]]
    if not xs or not ys:
        return '<div class="empty">No points</div>'
    xt, yt = _nice_ticks(min(xs), max(xs), 5), _nice_ticks(min(ys), max(ys), 5)
    xlo, xhi, ylo, yhi = xt[0], xt[-1], yt[0], yt[-1]
    pad_l, pad_r, pad_t, pad_b = 46, 14, 16, 40 if x_label else 26
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b

    def sx(v: float) -> float:
        return pad_l + (v - xlo) / max(xhi - xlo, 1e-9) * pw

    def sy(v: float) -> float:
        return pad_t + ph - (v - ylo) / max(yhi - ylo, 1e-9) * ph

    out = [_SVG_OPEN.format(w=width, h=height)]
    for t in yt:
        out.append(f'<line class="grid" x1="{pad_l}" y1="{sy(t):.1f}" x2="{width - pad_r}" y2="{sy(t):.1f}"/>')
        out.append(f'<text class="tick" x="{pad_l - 8}" y="{sy(t) + 3.5:.1f}" text-anchor="end">{_fmt_num(t)}</text>')
    for t in xt:
        out.append(f'<text class="tick" x="{sx(t):.1f}" y="{pad_t + ph + 16}" text-anchor="middle">{_fmt_num(t)}</text>')
    out.append(f'<line class="axis" x1="{pad_l}" y1="{pad_t + ph}" x2="{width - pad_r}" y2="{pad_t + ph}"/>')
    if diag_line:
        a, b = max(xlo, ylo), min(xhi, yhi)
        out.append(f'<line class="ref" x1="{sx(a):.1f}" y1="{sy(a):.1f}" x2="{sx(b):.1f}" y2="{sy(b):.1f}"/>')
    if hline is not None and ylo <= hline <= yhi:
        out.append(f'<line class="ref" x1="{pad_l}" y1="{sy(hline):.1f}" x2="{width - pad_r}" y2="{sy(hline):.1f}"/>')

    for i, g in enumerate(groups):
        col = g.get("color", TOKENS["series"][i % 6])
        pts = list(zip(g["x"], g["y"]))
        if len(pts) > 400:
            pts = pts[:: max(1, len(pts) // 400)]
        out.append(f'<g style="fill:{col}" fill-opacity="0.55">')
        out.extend(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.6"/>' for x, y in pts)
        out.append("</g>")

    if len(groups) > 1:
        x0 = pad_l + 10
        for i, g in enumerate(groups[:6]):
            col = g.get("color", TOKENS["series"][i % 6])
            yy = pad_t + 8 + i * 16
            out.append(f'<circle cx="{x0 + 4}" cy="{yy}" r="3.5" style="fill:{col}"/>')
            out.append(f'<text class="legend" x="{x0 + 14}" y="{yy + 3.5}">{_esc(g["name"][:26])}</text>')
    if x_label:
        out.append(f'<text class="axis-lbl" x="{pad_l + pw / 2:.1f}" y="{height - 4}" text-anchor="middle">{_esc(x_label)}</text>')
    out.append("</svg>")
    return "".join(out)


# ----------------------------------------------------------------------------- page

def _font_css() -> str:
    """Embed local woff2 fonts as data URIs so HTML is self-contained and renders offline."""
    css_file = FONT_DIR / "fonts.css"
    if not css_file.exists():
        return ""
    css = css_file.read_text(encoding="utf-8")
    for f in FONT_DIR.glob("*.woff2"):
        b64 = base64.b64encode(f.read_bytes()).decode("ascii")
        css = css.replace(f"url({f.name})", f"url(data:font/woff2;base64,{b64})")
    return css


def _theme_vars(theme: str) -> str:
    t = THEMES.get(theme, THEMES["light"])
    return "".join(f"--{k}:{v};" for k, v in t.items())


def set_theme(theme: str | None) -> None:
    """Apply a CLI --theme choice process-wide (overrides DS_SKILLER_THEME)."""
    if theme:
        os.environ["DS_SKILLER_THEME"] = theme


def _find_browser() -> str | None:
    override = os.environ.get("DS_SKILLER_BROWSER")
    if override and os.path.exists(override):
        return override
    for c in (
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ):
        if os.path.exists(c):
            return c
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome", "msedge"):
        if shutil.which(name):
            return shutil.which(name)
    return None


_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
html,body{background:var(--bg)}
body{color:var(--ink);font-family:"Geist","Segoe UI Variable","Segoe UI",system-ui,sans-serif;
  font-size:13px;line-height:1.45;-webkit-font-smoothing:antialiased;font-feature-settings:"ss01","cv11";
  padding:var(--pad-y) var(--pad-x)}
.mono,.tick,.val,td.num,.kpi-val{font-family:"Geist Mono",ui-monospace,Consolas,monospace;font-variant-numeric:tabular-nums}
header{display:flex;justify-content:space-between;align-items:flex-end;gap:24px;margin-bottom:28px}
.eyebrow{display:flex;align-items:center;gap:10px;font-size:11px;font-weight:500;letter-spacing:.14em;
  text-transform:uppercase;color:var(--ink-3);margin-bottom:10px}
.eyebrow .dot{width:6px;height:6px;border-radius:50%;background:var(--c1)}
h1{font-family:"Instrument Serif",Georgia,serif;font-weight:400;font-size:var(--h1);line-height:1.02;letter-spacing:-.015em;color:var(--ink)}
h1 em{font-style:italic;color:var(--ink-2)}
.sub{margin-top:10px;color:var(--ink-2);font-size:13px}
.meta{text-align:right;font-size:11px;color:var(--ink-3);line-height:1.7;white-space:nowrap}
.meta b{font-weight:500;color:var(--ink-2)}
.kpis{display:grid;grid-template-columns:repeat(var(--kpi-cols),minmax(0,1fr));gap:1px;background:var(--line);
  border:1px solid var(--line);border-radius:16px;overflow:hidden;margin-bottom:20px;box-shadow:var(--shadow)}
.kpi{background:var(--card);padding:18px 20px 20px}
.kpi-lbl{display:flex;align-items:center;gap:8px;font-size:11px;font-weight:500;color:var(--ink-3);letter-spacing:.02em}
.kpi-lbl i{width:6px;height:6px;border-radius:50%;background:var(--accent);flex:none}
.kpi-val{font-size:var(--kpi);font-weight:500;letter-spacing:-.03em;margin-top:10px;color:var(--ink);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-family:"Geist",system-ui,sans-serif}
.grid{display:grid;grid-template-columns:var(--cols);gap:20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px 22px 18px;box-shadow:var(--shadow);
  display:flex;flex-direction:column;min-width:0}
.span-2{grid-column:span 2}
.card-head{display:flex;justify-content:space-between;align-items:baseline;gap:12px;margin-bottom:14px}
.card-head h2{font-size:14px;font-weight:500;letter-spacing:-.01em;color:var(--ink)}
.tag{font-size:11px;color:var(--ink-3);font-family:"Geist Mono",monospace;white-space:nowrap}
.card-body{flex:1;display:flex;flex-direction:column;justify-content:center}
svg.chart{width:100%;height:auto;max-height:var(--chart-maxh);display:block;overflow:visible;margin:0 auto}
.grid line,.chart .grid{stroke:var(--grid);stroke-width:1}
.chart .axis{stroke:var(--line);stroke-width:1}
.chart .ref{stroke:var(--ink-3);stroke-width:1;stroke-dasharray:3 4}
.chart .thr{stroke-width:1.2;stroke-dasharray:3 3}
.chart .thr-lbl{font-size:10px;font-weight:500;font-family:"Geist Mono",monospace}
.chart .tick{fill:var(--ink-3);font-size:10px}
.chart .cat{fill:var(--ink-2);font-size:11.5px}
.chart .val{fill:var(--ink);font-size:10.5px;font-weight:500;paint-order:stroke;stroke:var(--card);stroke-width:4px;stroke-linejoin:round}
.chart .legend{fill:var(--ink-2);font-size:11px}
.chart .axis-lbl{fill:var(--ink-3);font-size:10.5px;letter-spacing:.02em}
.chart .track{fill:var(--card-2)}
.chart .legend-bg{fill:var(--card);fill-opacity:.88;stroke:var(--line)}
.chart .end-dot{stroke:var(--card);stroke-width:2}
.tbl-wrap{overflow:hidden}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{text-align:left;padding:0 10px 10px;font-size:10.5px;font-weight:500;letter-spacing:.06em;text-transform:uppercase;
  color:var(--ink-3);border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:9px 10px;border-bottom:1px solid var(--grid);color:var(--ink);white-space:nowrap;max-width:240px;overflow:hidden;text-overflow:ellipsis}
tbody tr:last-child td{border-bottom:none}
th.num,td.num{text-align:right}
td.num{font-size:12px;color:var(--ink-2)}
tr.row-best td{background:color-mix(in oklab,var(--c1) 7%,var(--card));color:var(--ink);font-weight:500}
tr.row-best td.num{color:var(--ink)}
tr.row-best td:first-child{box-shadow:inset 2px 0 0 var(--c1)}
.hm th{text-align:center;padding:0 4px 8px}
.hm td{padding:0;border:2px solid var(--card)}
.hm .hm-cell{text-align:center;height:34px;font-family:"Geist Mono",monospace;font-size:11.5px;border-radius:6px}
.hm .hm-lbl{padding-right:12px;color:var(--ink-2);font-size:12px;text-align:right;border:none}
.hm-lg .hm-cell{height:86px;font-size:22px;font-weight:500;letter-spacing:-.02em;border-radius:10px}
.hm-lg td{border-width:3px}
.badge{display:inline-flex;align-items:center;gap:6px;padding:3px 9px 3px 8px;border-radius:999px;font-size:11px;font-weight:500;
  border:1px solid var(--line);background:var(--card-2);color:var(--ink-2)}
.badge i{width:6px;height:6px;border-radius:50%;background:var(--ink-3)}
.badge-pos i{background:var(--pos)}.badge-warn i{background:var(--warn)}.badge-neg i{background:var(--neg)}.badge-accent i{background:var(--c1)}
.badge-pos{color:var(--pos)}.badge-warn{color:var(--warn)}.badge-neg{color:var(--neg)}
.empty{color:var(--ink-3);text-align:center;padding:28px;font-size:12.5px}
footer{margin-top:22px;display:flex;justify-content:space-between;font-size:10.5px;color:var(--ink-3);letter-spacing:.04em}
"""


def render_dashboard(
    title: str,
    subtitle: str,
    kpis: list[tuple[str, str, str | None]],
    panels: list[dict],
    out_dir: str | Path,
    file_stem: str,
    fmt: str = "landscape",
    theme: str | None = None,
) -> tuple[str, str]:
    """Render an editorial dashboard to self-contained HTML + 2x PNG.

    panels: [{'title', 'badge'(opt), 'html', 'span': 1|2 (opt)}]
    theme: 'light' (default) | 'dark'  (or env DS_SKILLER_THEME)
    """
    theme = (theme or os.environ.get("DS_SKILLER_THEME", "light")).lower()
    portrait = fmt == "portrait"
    n_k = len(kpis)
    kpi_cols = (3 if n_k >= 3 else n_k) if portrait else min(n_k, 6)

    kpi_html = []
    for i, item in enumerate(kpis):
        lbl, val = item[0], item[1]
        col = item[2] if len(item) > 2 and item[2] else "var(--ink-3)"
        span = ""
        if portrait and i == n_k - 1 and n_k % kpi_cols:
            span = f' style="--accent:{col};grid-column:span {kpi_cols - (n_k % kpi_cols) + 1}"'
        else:
            span = f' style="--accent:{col}"'
        kpi_html.append(f'<div class="kpi"{span}><div class="kpi-lbl"><i></i>{_esc(lbl)}</div><div class="kpi-val">{_esc(val)}</div></div>')

    panel_html = []
    for p in panels:
        span_cls = " span-2" if (not portrait and p.get("span") == 2) else ""
        tag = f'<span class="tag">{_esc(p["badge"])}</span>' if p.get("badge") else ""
        panel_html.append(
            f'<section class="card{span_cls}"><div class="card-head"><h2>{_esc(p["title"])}</h2>{tag}</div>'
            f'<div class="card-body">{p["html"]}</div></section>'
        )

    # Split title for an italic serif accent on the trailing phrase after '&' or '(' if present.
    title_html = _esc(title)
    for sep in (" & ", " ("):
        if sep in title:
            head, tail = title.split(sep, 1)
            joiner = "& " if sep == " & " else "("
            title_html = f"{_esc(head)} <em>{_esc(joiner)}{_esc(tail)}</em>"
            break

    skill_name = file_stem.replace("_dashboard", "").replace("_", " ")
    now = datetime.now().strftime("%d %b %Y · %H:%M")
    layout_vars = (
        f"--pad-y:{'56px' if portrait else '44px'};--pad-x:{'48px' if portrait else '52px'};"
        f"--h1:{'52px' if portrait else '46px'};--kpi:{'26px' if portrait else '28px'};"
        f"--kpi-cols:{kpi_cols};--cols:{'1fr' if portrait else '1fr 1fr'};"
        f"--chart-maxh:{'360px' if portrait else 'none'};"
    )

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{_esc(title)}</title>
<style>{_font_css()}
:root{{{_theme_vars(theme)}{layout_vars}}}
{_CSS}</style></head>
<body>
<header>
  <div>
    <div class="eyebrow"><span class="dot"></span>ds-skiller<span>/</span>{_esc(skill_name)}</div>
    <h1>{title_html}</h1>
    <div class="sub">{_esc(subtitle)}</div>
  </div>
  <div class="meta"><b>Generated</b><br/>{now}</div>
</header>
<section class="kpis">{"".join(kpi_html)}</section>
<main class="grid">{"".join(panel_html)}</main>
<footer><span>ds-skiller · Senior Data Science Toolkit</span><span>{_esc(theme)} · {_esc(fmt)}</span></footer>
</body></html>"""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    html_path = out / f"{file_stem}.html"
    png_path = out / f"{file_stem}.png"
    html_path.write_text(doc, encoding="utf-8")

    browser = _find_browser()
    if browser:
        win = "1080,5800" if portrait else "1600,3800"
        bg_hex = THEMES.get(theme, THEMES["light"])["bg"].lstrip("#")
        subprocess.run(
            [
                browser, "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
                "--force-device-scale-factor=2", f"--window-size={win}", "--virtual-time-budget=1500",
                f"--default-background-color={bg_hex}FF",
                f"--screenshot={png_path.resolve()}", html_path.resolve().as_uri(),
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
        _trim_png(png_path, bg_hex, portrait)
    else:
        import sys
        print("WARN: Chrome/Edge not found (set DS_SKILLER_BROWSER) -> PNG skipped, HTML only.", file=sys.stderr)
    return (str(png_path) if png_path.exists() else ""), str(html_path)


def _trim_png(png_path: Path, bg_hex: str, portrait: bool) -> None:
    """Crop empty canvas below the footer; pad portrait output to at least 9:16."""
    try:
        from PIL import Image
    except ImportError:
        return
    if not png_path.exists():
        return
    img = Image.open(png_path).convert("RGB")
    arr = np.asarray(img).astype(np.int16)
    bg = np.array([int(bg_hex[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.int16)
    diff_rows = np.where((np.abs(arr - bg).sum(axis=2) > 12).any(axis=1))[0]
    if diff_rows.size == 0:
        return
    bottom = min(arr.shape[0], int(diff_rows[-1]) + (112 if portrait else 88))
    img = img.crop((0, 0, img.width, bottom))
    if portrait:
        min_h = int(img.width * 16 / 9)
        if img.height < min_h:
            canvas = Image.new("RGB", (img.width, min_h), tuple(int(x) for x in bg))
            canvas.paste(img, (0, 0))
            img = canvas
    img.save(png_path, optimize=True)
