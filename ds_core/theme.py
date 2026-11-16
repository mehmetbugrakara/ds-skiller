"""Broadcast-grade visual theme for DS dashboards (16:9 Landscape & 9:16 Reels Portrait)."""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

PALETTE = {
    "bg": "#0F172A",       # Slate 900
    "card": "#1E293B",     # Slate 800
    "border": "#334155",   # Slate 700
    "text": "#F8FAFC",     # Slate 50
    "muted": "#94A3B8",    # Slate 400
    "primary": "#38BDF8",  # Sky 400
    "success": "#22C55E",  # Green 500
    "warning": "#F59E0B",  # Amber 500
    "danger": "#EF4444",   # Red 500
    "purple": "#A78BFA",   # Violet 400
    "series": ["#38BDF8", "#22C55E", "#F59E0B", "#A78BFA", "#F472B6", "#2DD4BF"],
}


def setup_theme() -> None:
    """Apply global matplotlib rcParams for crisp, dark-mode visuals."""
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor": PALETTE["card"],
        "axes.edgecolor": PALETTE["border"],
        "axes.labelcolor": PALETTE["text"],
        "axes.titlecolor": PALETTE["text"],
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "grid.color": PALETTE["border"],
        "grid.alpha": 0.45,
        "grid.linestyle": "--",
        "xtick.color": PALETTE["muted"],
        "ytick.color": PALETTE["muted"],
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "text.color": PALETTE["text"],
        "legend.facecolor": PALETTE["card"],
        "legend.edgecolor": PALETTE["border"],
        "legend.fontsize": 8.5,
        "font.family": "DejaVu Sans",
    })


def create_dashboard(
    title: str,
    subtitle: str,
    fmt: str = "landscape",
    kpis: list[tuple[str, str, str | None]] | None = None,
) -> tuple[plt.Figure, dict]:
    """Create a styled dashboard figure (landscape 16:9 or portrait 9:16) with optional KPI strip."""
    setup_theme()
    is_portrait = fmt == "portrait"
    figsize = (9, 16) if is_portrait else (16, 9.5)
    fig = plt.figure(figsize=figsize, facecolor=PALETTE["bg"])

    title_y = 0.968 if is_portrait else 0.955
    sub_y = 0.946 if is_portrait else 0.922
    fig.text(0.03, title_y, title, fontsize=17 if is_portrait else 18, fontweight="bold", color=PALETTE["text"])
    fig.text(0.03, sub_y, subtitle, fontsize=10, color=PALETTE["muted"])
    fig.text(0.97, title_y, "ds-skiller", fontsize=9, fontweight="bold", color=PALETTE["primary"], ha="right")

    top_margin = 0.89 if is_portrait else 0.87
    if kpis:
        draw_kpi_strip(fig, kpis, y_top=top_margin, is_portrait=is_portrait)
        top_margin -= 0.095 if is_portrait else 0.135

    layout = {
        "is_portrait": is_portrait,
        "top": top_margin,
        "bottom": 0.05,
        "left": 0.16 if is_portrait else 0.09,
        "right": 0.96,
    }
    return fig, layout


def draw_kpi_strip(
    fig: plt.Figure,
    kpis: list[tuple[str, str, str | None]],
    y_top: float = 0.88,
    is_portrait: bool = False,
) -> None:
    """Render a horizontal strip of KPI cards at the top of the figure."""
    n = len(kpis)
    if n == 0:
        return
    height = 0.065 if is_portrait else 0.09
    gap = 0.015
    total_w = 0.94 - gap * (n - 1)
    card_w = total_w / n

    for i, item in enumerate(kpis):
        label, value = item[0], item[1]
        color = item[2] if len(item) > 2 and item[2] else PALETTE["primary"]
        x0 = 0.03 + i * (card_w + gap)
        y0 = y_top - height
        rect = patches.FancyBboxPatch(
            (x0, y0), card_w, height,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            facecolor=PALETTE["card"], edgecolor=PALETTE["border"], linewidth=1.2,
            transform=fig.transFigure, clip_on=False,
        )
        fig.add_artist(rect)
        # Left accent bar
        bar = patches.Rectangle(
            (x0, y0 + 0.006), 0.004, height - 0.012,
            facecolor=color, transform=fig.transFigure, clip_on=False,
        )
        fig.add_artist(bar)
        fig.text(x0 + 0.014, y0 + height * 0.66, label.upper(), fontsize=7.5 if is_portrait else 8.5,
                 color=PALETTE["muted"], fontweight="bold")
        fig.text(x0 + 0.014, y0 + height * 0.20, str(value), fontsize=12 if is_portrait else 14,
                 color=PALETTE["text"], fontweight="bold")


def draw_table(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    highlight_row: int | None = None,
    max_rows: int = 10,
) -> None:
    """Render a clean styled DataFrame table inside a matplotlib Axes."""
    ax.set_axis_off()
    ax.set_title(title, loc="left", pad=8, fontsize=11, fontweight="bold", color=PALETTE["text"])

    sub = df.head(max_rows).copy()
    if sub.empty:
        ax.text(0.5, 0.5, "No rows", ha="center", va="center", color=PALETTE["muted"])
        return

    cell_text = [[str(v) for v in row] for row in sub.values]
    col_labels = [str(c) for c in sub.columns]

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="upper center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.45)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(PALETTE["border"])
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#0B1120")
            cell.set_text_props(color=PALETTE["primary"], fontweight="bold")
        elif highlight_row is not None and row - 1 == highlight_row:
            cell.set_facecolor("#16324F")
            cell.set_text_props(color=PALETTE["success"], fontweight="bold")
        else:
            cell.set_facecolor(PALETTE["card"] if row % 2 == 1 else "#162032")
            cell.set_text_props(color=PALETTE["text"])


def save_figure(fig: plt.Figure, out_path: str | Path, dpi: int = 160) -> str:
    """Save figure to disk and close it to free memory."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, facecolor=PALETTE["bg"])
    plt.close(fig)
    return str(p)
