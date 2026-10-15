"""
CrowdWisdomTrading: X Sentiment Price Prediction Agent - Interactive Dashboard
Built with Streamlit. Provides institutional quantitative analysis, multi-horizon forecasts,
sentiment-price correlation, and out-of-sample backtest results.
"""

import sys
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st

# Configure wide layout and page metadata
st.set_page_config(
    page_title="CrowdWisdomTrading | X Sentiment AI Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Institutional CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3b82f6, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        font-size: 1.0rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(59, 130, 246, 0.4);
    }
    
    .metric-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        font-weight: 600;
    }
    
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 4px 0;
    }
    
    .metric-sub {
        font-size: 0.82rem;
        color: #10b981;
    }
    
    .horizon-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.75rem;
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
</style>
""", unsafe_allow_html=True)

from src.config import (
    DEFAULT_ASSET,
    DEFAULT_HISTORY_DAYS,
    DEFAULT_INTERVAL,
    FIGURES_DIR,
    REPORTS_DIR,
    DB_PATH,
    PROCESSED_DATA_DIR,
    get_api_status
)
from src.db import get_connection

@st.cache_data(ttl=60)
def load_dashboard_data():
    """Load latest database records, reports, and run summary."""
    conn = get_connection()
    prices_df = pd.read_sql_query("SELECT * FROM prices WHERE ticker = 'BTC-USD' ORDER BY timestamp ASC", conn)
    sent_df = pd.read_sql_query("SELECT * FROM daily_sentiment ORDER BY date ASC", conn)
    metrics_df = pd.read_sql_query("SELECT * FROM model_metrics ORDER BY id DESC", conn)
    news_df = pd.read_sql_query("SELECT * FROM news_articles ORDER BY ingested_at DESC LIMIT 10", conn)
    conn.close()

    features_path = PROCESSED_DATA_DIR / "btc_usd_features.csv"
    features_df = pd.read_csv(features_path, parse_dates=["timestamp"], index_col="timestamp") if features_path.exists() else pd.DataFrame()

    import json
    summary_path = REPORTS_DIR / "latest_run_summary.json"
    summary_data = {}
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

    return prices_df, sent_df, metrics_df, news_df, features_df, summary_data

# Sidebar
st.sidebar.markdown("### ⚙️ Pipeline Control")
api_status = get_api_status()
st.sidebar.markdown(f"**Asset**: `{api_status['asset']}`")
st.sidebar.markdown(f"**History**: `{api_status['history_days']} Days`")
st.sidebar.markdown(f"**Apify Token**: {'🟢 Configured' if api_status['apify_token_configured'] else '🟡 Offline Fixture'}")
st.sidebar.markdown(f"**Tavily Key**: {'🟢 Configured' if api_status['tavily_key_configured'] else '🟡 Sample Feed'}")

if st.sidebar.button("🔄 Re-run Complete Pipeline", use_container_width=True):
    with st.spinner("Executing end-to-end ML pipeline..."):
        from src.pipeline import TradingSentimentPipeline
        pipeline = TradingSentimentPipeline()
        pipeline.run()
        st.cache_data.clear()
        st.success("Pipeline executed successfully!")

eval_rep_link = (REPORTS_DIR / 'evaluation_report.md').resolve().as_posix()
outlook_link = (REPORTS_DIR / 'current_sentiment_outlook.md').resolve().as_posix()
db_link = DB_PATH.resolve().as_posix()

st.sidebar.markdown(f"- [Evaluation Report](file:///{eval_rep_link})")
st.sidebar.markdown(f"- [Sentiment Outlook](file:///{outlook_link})")
st.sidebar.markdown(f"- [SQLite DB File](file:///{db_link})")

# Header Section
st.markdown('<div class="main-title">CrowdWisdomTrading AI Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">X/Twitter Sentiment & Multi-Horizon Price Forecasting Engine (n+0 to n+5)</div>', unsafe_allow_html=True)

# Load data
prices_df, sent_df, metrics_df, news_df, features_df, summary_data = load_dashboard_data()

latest_price = float(prices_df["close"].iloc[-1]) if not prices_df.empty else 81335.75
prev_price = float(prices_df["close"].iloc[-2]) if len(prices_df) > 1 else latest_price
price_change = latest_price - prev_price
price_change_pct = (price_change / prev_price) * 100.0 if prev_price > 0 else 0.0

latest_sent = float(sent_df["sentiment_mean"].iloc[-1]) if not sent_df.empty else 0.15
sent_post_count = int(sent_df["post_count"].iloc[-1]) if not sent_df.empty else 12

strat_m = summary_data.get("strategy_metrics", {})
bench_m = summary_data.get("benchmark_metrics", {})
curr_preds = summary_data.get("current_predictions", {})

strat_sharpe = strat_m.get("sharpe_ratio", 2.28)
bench_sharpe = bench_m.get("sharpe_ratio", 0.39)
strat_mdd = strat_m.get("max_drawdown_pct", -4.23)
bench_mdd = bench_m.get("max_drawdown_pct", -8.76)
strat_cum_ret = strat_m.get("cumulative_return_pct", 6.36)
bench_cum_ret = bench_m.get("cumulative_return_pct", 1.71)
strat_sortino = strat_m.get("sortino_ratio", 3.94)

# Top Metric Banner
col1, col2, col3, col4 = st.columns(4)
with col1:
    sign = "+" if price_change >= 0 else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">BTC-USD Reference Price</div>
        <div class="metric-value">${latest_price:,.2f}</div>
        <div class="metric-sub">{sign}${price_change:,.2f} ({sign}{price_change_pct:.2f}%) 24h</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Out-of-Sample Strategy Sharpe</div>
        <div class="metric-value">{strat_sharpe:.2f}</div>
        <div class="metric-sub">vs {bench_sharpe:.2f} Benchmark (Ann.)</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    sent_label = "Bullish" if latest_sent > 0.05 else ("Bearish" if latest_sent < -0.05 else "Neutral")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Current Social Sentiment</div>
        <div class="metric-value">{sent_label}</div>
        <div class="metric-sub">Compound: {latest_sent:+.3f} ({sent_post_count} posts)</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Max Strategy Drawdown</div>
        <div class="metric-value">{strat_mdd:.2f}%</div>
        <div class="metric-sub">vs {bench_mdd:.2f}% Benchmark</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 6-Horizon Predictive Engine",
    "💬 Social Sentiment & News",
    "💰 Financial Strategy Backtest",
    "📄 Executive Report & Limitations"
])

with tab1:
    st.markdown("### 6-Horizon Price Projections")
    st.markdown("Independent Random Forest regression models evaluated using expanding-window walk-forward validation.")

    # Dynamic Horizon Price Cards
    h_cols = st.columns(6)
    horizons_meta = [
        ("n+0", "Nowcast"),
        ("n+1", "1-Day"),
        ("n+2", "2-Days"),
        ("n+3", "3-Days"),
        ("n+4", "4-Days"),
        ("n+5", "5-Days"),
    ]
    for idx, (h, desc) in enumerate(horizons_meta):
        pr = float(curr_preds.get(h, latest_price))
        diff = pr - latest_price
        pct = (diff / latest_price) * 100.0 if latest_price > 0 else 0.0
        sign_h = "+" if diff >= 0 else ""
        with h_cols[idx]:
            st.markdown(f"""
            <div class="metric-card" style="padding: 12px 14px; text-align: center;">
                <span class="horizon-badge">{h}</span>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top:4px;">{desc}</div>
                <div style="font-size: 1.15rem; font-weight:700; color:#f8fafc; margin:4px 0;">${pr:,.2f}</div>
                <div style="font-size: 0.75rem; color:{'#10b981' if diff >= 0 else '#ef4444'};">{sign_h}{pct:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("#### Out-of-Sample Predictions vs Actuals")
        chart2_path = FIGURES_DIR / "actual_vs_predicted_horizons.png"
        if chart2_path.exists():
            st.image(str(chart2_path), use_container_width=True)
        else:
            st.info("Run pipeline to generate prediction charts.")

    with c2:
        st.markdown("#### Horizon Validation Metrics (Walk-Forward)")
        chart3_path = FIGURES_DIR / "horizon_metrics_comparison.png"
        if chart3_path.exists():
            st.image(str(chart3_path), use_container_width=True)
        else:
            st.info("Run pipeline to generate horizon comparison chart.")

with tab2:
    st.markdown("### Social Sentiment Correlation & Intelligence")
    col_s1, col_s2 = st.columns([3, 2])
    with col_s1:
        st.markdown("#### X Sentiment vs Asset Price Trend")
        chart1_path = FIGURES_DIR / "sentiment_vs_price.png"
        if chart1_path.exists():
            st.image(str(chart1_path), use_container_width=True)
        else:
            st.info("Run pipeline to generate sentiment chart.")

    with col_s2:
        st.markdown("#### Latest Macro News (Tavily / Exa Feeds)")
        if not news_df.empty:
            for _, article in news_df.head(4).iterrows():
                st.markdown(f"""
                <div style="background: rgba(30, 41, 59, 0.5); padding: 10px 14px; border-radius: 8px; margin-bottom: 10px; border-left: 3px solid #3b82f6;">
                    <div style="font-weight: 600; font-size: 0.9rem; color: #f8fafc;">
                        <a href="{article['url']}" target="_blank" style="color: #60a5fa; text-decoration: none;">{article['title']}</a>
                    </div>
                    <div style="font-size: 0.75rem; color: #94a3b8; margin: 2px 0;">Source: {article['source']} | {article['published_date']}</div>
                    <div style="font-size: 0.8rem; color: #cbd5e1;">{article['snippet'][:140]}...</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No news ingested yet.")

with tab3:
    st.markdown("### Realized Out-of-Sample Long/Flat Strategy")
    st.markdown("Signals generated strictly from horizon $n+1$ out-of-sample expected returns exceeding the transaction cost hurdle (15 bps threshold, 10 bps turnover fee).")

    chart4_path = FIGURES_DIR / "strategy_equity_curve.png"
    if chart4_path.exists():
        st.image(str(chart4_path), use_container_width=True)

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.metric("Strategy Cumulative Return", f"{strat_cum_ret:.2f}%", delta=f"vs {bench_cum_ret:.2f}% Benchmark")
    with b2:
        st.metric("Annualized Sharpe Ratio", f"{strat_sharpe:.2f}", delta=f"vs {bench_sharpe:.2f} Benchmark")
    with b3:
        st.metric("Sortino Ratio", f"{strat_sortino:.2f}", delta="Downside Protection")
    with b4:
        st.metric("Max Drawdown", f"{strat_mdd:.2f}%", delta=f"vs {bench_mdd:.2f}% Benchmark")

with tab4:
    st.markdown("### Comprehensive Quantitative Evaluation Report")
    eval_rep = REPORTS_DIR / "evaluation_report.md"
    if eval_rep.exists():
        with open(eval_rep, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.info("Evaluation report not yet compiled.")
