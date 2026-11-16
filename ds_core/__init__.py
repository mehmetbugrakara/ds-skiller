"""Shared utilities for ds-skiller: IO, broadcast-grade visual theme, and token-friendly output."""

from .io import load_dataset, split_feature_types, prepare_features, prepare_inference_features
from .theme import PALETTE, create_dashboard, draw_kpi_strip, draw_table, save_figure
from .html_dashboard import (
    TOKENS,
    badge,
    html_heatmap,
    html_table,
    render_dashboard,
    set_theme,
    svg_barh,
    svg_barv,
    svg_line_chart,
    svg_scatter,
)
from .output import emit_result

__all__ = [
    "load_dataset",
    "split_feature_types",
    "prepare_features",
    "prepare_inference_features",
    "set_theme",
    "PALETTE",
    "TOKENS",
    "create_dashboard",
    "draw_kpi_strip",
    "draw_table",
    "save_figure",
    "badge",
    "html_heatmap",
    "html_table",
    "render_dashboard",
    "svg_barh",
    "svg_barv",
    "svg_line_chart",
    "svg_scatter",
    "emit_result",
]
