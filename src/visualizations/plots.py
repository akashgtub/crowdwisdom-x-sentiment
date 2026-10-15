"""
Visualization module for CrowdWisdomTrading pipeline.
Generates publication-quality charts using Matplotlib:
1. X Sentiment versus Asset Price over time (distinguishing missing from neutral).
2. Actual versus Out-of-Sample Predicted prices across horizons.
3. Comparative performance metrics (MAE, RMSE, MAPE) across all 6 horizons.
4. Strategy Equity Curve and Drawdown analysis.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI reproducibility
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR

logger = logging.getLogger(__name__)

# Matplotlib styling for high aesthetic fidelity
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "grid.alpha": 0.4
})

def plot_sentiment_vs_price(
    feature_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    asset_name: str = "BTC-USD"
) -> Path:
    """
    Plot dual-panel chart:
    - Top: Price with 7d & 20d SMAs
    - Bottom: Daily X sentiment, color-coded with missing vs neutral distinction.
    """
    out_path = output_path or (FIGURES_DIR / "sentiment_vs_price.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = feature_df.copy()
    df.sort_index(inplace=True)

    fig, (ax_price, ax_sent) = plt.subplots(
        nrows=2, ncols=1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    # Top Panel: Price and Moving Averages
    ax_price.plot(df.index, df["close"], label=f"{asset_name} Close Price", color="#1f77b4", linewidth=1.8)
    if "sma_7" in df.columns:
        ax_price.plot(df.index, df["sma_7"], label="7-Day SMA", color="#ff7f0e", linestyle="--", linewidth=1.2, alpha=0.85)
    if "sma_20" in df.columns:
        ax_price.plot(df.index, df["sma_20"], label="20-Day SMA", color="#2ca02c", linestyle=":", linewidth=1.2, alpha=0.85)

    ax_price.set_title(f"{asset_name} Price Trend & Moving Averages", fontweight="bold", pad=8)
    ax_price.set_ylabel("Price (USD)")
    ax_price.legend(loc="upper left", frameon=True)
    ax_price.yaxis.set_major_formatter("${x:,.0f}")

    # Bottom Panel: Sentiment Bar Chart
    colors = []
    alphas = []
    for _, row in df.iterrows():
        if row.get("has_sentiment_data", 1) == 0:
            colors.append("#7f7f7f")  # Grey for missing / imputed
            alphas.append(0.3)
        elif row["sentiment_mean"] >= 0.05:
            colors.append("#2ca02c")  # Green for positive
            alphas.append(0.85)
        elif row["sentiment_mean"] <= -0.05:
            colors.append("#d62728")  # Red for negative
            alphas.append(0.85)
        else:
            colors.append("#bcbd22")  # Yellow-olive for genuine neutral
            alphas.append(0.7)

    bars = ax_sent.bar(df.index, df["sentiment_mean"], color=colors, width=0.8, alpha=0.8, edgecolor="none")
    ax_sent.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.5)
    ax_sent.axhline(0.05, color="#2ca02c", linewidth=0.6, linestyle=":", alpha=0.5)
    ax_sent.axhline(-0.05, color="#d62728", linewidth=0.6, linestyle=":", alpha=0.5)

    ax_sent.set_title("Aggregated Daily X Sentiment (Compound Score)", fontweight="bold", pad=8)
    ax_sent.set_ylabel("Sentiment Score")
    ax_sent.set_ylim(-1.0, 1.0)

    # Custom legend for sentiment categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", label="Bullish (Compound ≥ 0.05)"),
        Patch(facecolor="#bcbd22", label="Neutral (-0.05 < Compound < 0.05)"),
        Patch(facecolor="#d62728", label="Bearish (Compound ≤ -0.05)"),
        Patch(facecolor="#7f7f7f", alpha=0.4, label="Missing Data / Imputed")
    ]
    ax_sent.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize=8)

    ax_sent.xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %Y"))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Saved sentiment vs price chart to %s", out_path)
    return out_path

def plot_actual_vs_predicted(
    horizon_results: Dict[str, Any],
    output_path: Optional[Path] = None
) -> Path:
    """
    Plot 6 subplots comparing Actual vs Out-of-Sample Predicted prices
    for horizons n+0 through n+5.
    """
    out_path = output_path or (FIGURES_DIR / "actual_vs_predicted_horizons.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    horizons = ["n+0", "n+1", "n+2", "n+3", "n+4", "n+5"]
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 11), sharex=True)
    axes_flat = axes.flatten()

    for idx, h in enumerate(horizons):
        ax = axes_flat[idx]
        if h in horizon_results:
            p_df = horizon_results[h]["predictions_df"]
            rf_m = horizon_results[h]["rf_metrics"]
            ax.plot(p_df.index, p_df["actual"], label="Actual Price", color="#111111", linewidth=1.5)
            ax.plot(p_df.index, p_df["rf_predicted"], label="RF Forecast", color="#1f77b4", linewidth=1.4, linestyle="--")
            ax.plot(p_df.index, p_df["naive_predicted"], label="Naive Baseline", color="#ff7f0e", linewidth=1.1, linestyle=":", alpha=0.7)

            ax.set_title(f"Horizon {h} | MAE: ${rf_m['mae']:,.0f} | MAPE: {rf_m['mape']:.2f}% | DirAcc: {rf_m['directional_accuracy']:.1f}%",
                         fontsize=10, fontweight="bold")
            ax.yaxis.set_major_formatter("${x:,.0f}")
            ax.legend(loc="upper left", fontsize=8)
        else:
            ax.text(0.5, 0.5, f"No Data for {h}", ha="center", va="center")

    fig.suptitle("Out-of-Sample Walk-Forward Predictions across 6 Forecasting Horizons", fontweight="bold", y=0.995)
    axes_flat[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes_flat[-2].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Saved actual vs predicted chart to %s", out_path)
    return out_path

def plot_horizon_comparison(
    summary_metrics_df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> Path:
    """
    Plot comparative bar charts of MAE, RMSE, and MAPE across horizons n+0 to n+5
    contrasting the Random Forest model against the Naive Baseline.
    """
    out_path = output_path or (FIGURES_DIR / "horizon_metrics_comparison.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = summary_metrics_df.copy()
    fig, (ax_mae, ax_mape, ax_dir) = plt.subplots(nrows=1, ncols=3, figsize=(15, 4.5))

    x = np.arange(len(df))
    width = 0.35

    # Panel 1: MAE Comparison
    ax_mae.bar(x - width/2, df["RF_MAE"], width, label="Random Forest", color="#1f77b4")
    ax_mae.bar(x + width/2, df["Naive_MAE"], width, label="Naive Persistence", color="#ff7f0e")
    ax_mae.set_title("Mean Absolute Error (MAE - USD)", fontweight="bold")
    ax_mae.set_xticks(x)
    ax_mae.set_xticklabels(df["Horizon"])
    ax_mae.set_ylabel("USD ($)")
    ax_mae.yaxis.set_major_formatter("${x:,.0f}")
    ax_mae.legend(frameon=True)

    # Panel 2: MAPE Comparison
    ax_mape.bar(x - width/2, df["RF_MAPE_%"], width, label="Random Forest", color="#2ca02c")
    ax_mape.bar(x + width/2, df["Naive_MAPE_%"], width, label="Naive Persistence", color="#d62728")
    ax_mape.set_title("Mean Absolute Percentage Error (MAPE %)", fontweight="bold")
    ax_mape.set_xticks(x)
    ax_mape.set_xticklabels(df["Horizon"])
    ax_mape.set_ylabel("Percentage (%)")
    ax_mape.legend(frameon=True)

    # Panel 3: Directional Accuracy Comparison
    ax_dir.bar(x - width/2, df["RF_DirAcc_%"], width, label="Random Forest", color="#9467bd")
    ax_dir.bar(x + width/2, df["Naive_DirAcc_%"], width, label="Naive Persistence", color="#8c564b")
    ax_dir.axhline(50.0, color="black", linestyle="--", alpha=0.5, label="Random (50%)")
    ax_dir.set_title("Directional Accuracy (%)", fontweight="bold")
    ax_dir.set_xticks(x)
    ax_dir.set_xticklabels(df["Horizon"])
    ax_dir.set_ylabel("Accuracy (%)")
    ax_dir.set_ylim(0, 100)
    ax_dir.legend(frameon=True)

    fig.suptitle("Forecasting Horizon Performance Comparison (n+0 to n+5)", fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Saved horizon metrics comparison to %s", out_path)
    return out_path

def plot_strategy_equity_curve(
    strategy_eval_dict: Dict[str, Any],
    output_path: Optional[Path] = None
) -> Path:
    """
    Plot realized out-of-sample cumulative equity curve and drawdown profile
    for the Long/Flat strategy against the Buy & Hold benchmark.
    """
    out_path = output_path or (FIGURES_DIR / "strategy_equity_curve.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = strategy_eval_dict["evaluation_df"].copy()
    s_m = strategy_eval_dict["strategy_metrics"]
    b_m = strategy_eval_dict["benchmark_metrics"]

    fig, (ax_eq, ax_dd) = plt.subplots(
        nrows=2, ncols=1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    # Top: Equity Curve
    ax_eq.plot(df.index, df["strategy_equity"], label=f"Long/Flat Strategy (Net) [Return: {s_m['cumulative_return_pct']}%, Sharpe: {s_m['sharpe_ratio']}]",
               color="#2ca02c", linewidth=2.0)
    ax_eq.plot(df.index, df["benchmark_equity"], label=f"Buy & Hold Benchmark [Return: {b_m['cumulative_return_pct']}%, Sharpe: {b_m['sharpe_ratio']}]",
               color="#1f77b4", linewidth=1.5, linestyle="--")

    ax_eq.set_title("Out-of-Sample Strategy Cumulative Equity Curve (1.0 = Initial Capital)", fontweight="bold")
    ax_eq.set_ylabel("Equity Multiplier")
    ax_eq.legend(loc="upper left", frameon=True)

    # Bottom: Drawdown
    ax_dd.fill_between(df.index, df["strategy_drawdown"] * 100, 0, color="#d62728", alpha=0.35, label=f"Strategy Max DD ({s_m['max_drawdown_pct']}%)")
    ax_dd.plot(df.index, df["benchmark_drawdown"] * 100, color="#1f77b4", linestyle=":", label=f"Benchmark Max DD ({b_m['max_drawdown_pct']}%)")
    ax_dd.set_title("Portfolio Drawdown Profile (%)", fontweight="bold")
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.set_ylim(bottom=min(-30.0, float(b_m['max_drawdown_pct']) * 1.2), top=2.0)
    ax_dd.legend(loc="lower left", frameon=True)

    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %Y"))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Saved strategy equity curve to %s", out_path)
    return out_path
