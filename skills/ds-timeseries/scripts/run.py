"""Time Series diagnostics, decomposition & Holt-Winters Impeccable HTML/PNG dashboard."""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*infer_freq.*")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ds_core import set_theme, TOKENS, emit_result, load_dataset, render_dashboard, svg_barv, svg_line_chart


def run_timeseries(data_path: str, date_col: str, target: str, horizon: int, period: int, fmt: str, out_dir: str) -> None:
    df = load_dataset(data_path)
    df[date_col] = pd.to_datetime(df[date_col])
    ts = df.groupby(date_col)[target].sum().sort_index().astype(float)
    ts.index = pd.DatetimeIndex(ts.index, freq=pd.infer_freq(ts.index) or "D")

    adf_res = adfuller(ts.dropna())
    adf_p = float(adf_res[1])
    is_stationary = adf_p < 0.05

    test_len = min(horizon, max(7, len(ts) // 5))
    train, test = ts.iloc[:-test_len], ts.iloc[-test_len:]

    use_seasonal = len(train) >= 2 * period
    bt_model = ExponentialSmoothing(
        train, trend="add", seasonal="add" if use_seasonal else None,
        seasonal_periods=period if use_seasonal else None,
    ).fit(optimized=True)
    bt_pred = bt_model.forecast(test_len)
    bt_rmse = float(root_mean_squared_error(test, bt_pred))
    nonzero = test != 0
    bt_mape = float(mean_absolute_percentage_error(test[nonzero], bt_pred[nonzero]) * 100) if nonzero.any() else 0.0

    full_model = ExponentialSmoothing(
        ts, trend="add", seasonal="add" if len(ts) >= 2 * period else None,
        seasonal_periods=period if len(ts) >= 2 * period else None,
    ).fit(optimized=True)
    fc = full_model.forecast(horizon)
    resid_std = float(np.std(full_model.resid))
    steps = np.arange(1, horizon + 1)
    ci_half = 1.96 * resid_std * np.sqrt(steps / max(period, 1))

    # Build numeric step axis for crisp SVG rendering
    hist_tail = ts.iloc[-min(len(ts), horizon * 4):]
    n_h = len(hist_tail)
    x_hist = list(range(n_h))
    x_bt = list(range(n_h - test_len, n_h))
    x_fc = list(range(n_h, n_h + horizon))

    fc_svg = svg_line_chart(
        [
            {"name": "Actual", "x": x_hist, "y": hist_tail.values.tolist(), "color": TOKENS["primary"]},
            {"name": "Backtest Pred", "x": x_bt, "y": bt_pred.values.tolist(), "color": TOKENS["warning"], "dashed": True},
            {"name": "Forecast (95% CI)", "x": x_fc, "y": fc.values.tolist(), "color": TOKENS["success"],
             "band": ((fc.values - ci_half).tolist(), (fc.values + ci_half).tolist())},
        ],
        x_label=f"Time Steps ({hist_tail.index[0].strftime('%Y-%m-%d')} -> +{horizon}d)",
        width=1160,
        height=300,
    )

    if len(ts) >= 2 * period:
        decomp = seasonal_decompose(ts, model="additive", period=period, extrapolate_trend="period")
        trend_svg = svg_line_chart(
            [{"name": "Trend", "x": list(range(len(decomp.trend))), "y": decomp.trend.values.tolist(), "color": TOKENS["purple"], "fill": True}],
            x_label="Time Index",
        )
        seas_slice = decomp.seasonal.iloc[:period]
        seas_svg = svg_barv([f"S{i + 1}" for i in range(len(seas_slice))], seas_slice.values.tolist(),
                            colors=[TOKENS["success"] if v >= 0 else TOKENS["danger"] for v in seas_slice.values], fmt_str="{:+.1f}")
    else:
        rm = ts.rolling(max(3, len(ts) // 10)).mean().bfill()
        trend_svg = svg_line_chart([{"name": "Rolling Mean", "x": list(range(len(rm))), "y": rm.values.tolist(), "color": TOKENS["purple"]}])
        seas_svg = '<div class="empty">Insufficient length for seasonal cycle</div>'

    kpis = [
        ("Data Points", f"{len(ts):,}", TOKENS["primary"]),
        ("ADF Stationarity", f"p={adf_p:.4f} ({'Stat' if is_stationary else 'Non-Stat'})", TOKENS["success"] if is_stationary else TOKENS["warning"]),
        ("Backtest RMSE", f"{bt_rmse:.2f}", TOKENS["purple"]),
        ("Backtest MAPE", f"{bt_mape:.2f}%", TOKENS["success"] if bt_mape < 15 else TOKENS["warning"]),
        ("Forecast Horizon", f"+{horizon} steps", TOKENS["teal"]),
    ]

    panels = [
        {"title": f"Actual, Backtest & +{horizon} Forecast", "badge": "Holt-Winters 95% CI", "html": fc_svg, "span": 2},
        {"title": "Extracted Trend Component", "badge": "Additive Decomp", "html": trend_svg},
        {"title": f"Seasonal Cycle Pattern (Period={period})", "badge": "Cycle Effect", "html": seas_svg},
    ]

    png_path, html_path = render_dashboard(
        title="Time Series Diagnostics & Forecast",
        subtitle=f"Dataset: {Path(data_path).name} | Target: {target} | Period={period}",
        kpis=kpis,
        panels=panels,
        out_dir=out_dir,
        file_stem="forecast_dashboard",
        fmt=fmt,
    )
    emit_result(
        skill="ds-timeseries",
        out_dir=out_dir,
        kpis={"adf_p": round(adf_p, 4), "bt_rmse": round(bt_rmse, 2), "bt_mape": round(bt_mape, 2)},
        visuals=[png_path, html_path],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--date", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--horizon", type=int, default=14)
    p.add_argument("--period", type=int, default=7)
    p.add_argument("--format", default="landscape", choices=["landscape", "portrait"])
    p.add_argument("--theme", default=None, choices=["light", "dark"])
    p.add_argument("--out", default="outputs/ds-timeseries")
    args = p.parse_args()
    set_theme(args.theme)
    run_timeseries(args.data, args.date, args.target, args.horizon, args.period, args.format, args.out)
