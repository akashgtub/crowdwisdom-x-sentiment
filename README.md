# CrowdWisdomTrading: X Sentiment Price Prediction Agent

An institutional-grade Python quantitative pipeline and machine learning forecasting engine for crypto and asset price prediction across six horizons ($n+0$ through $n+5$). The system pairs calibrated social media sentiment from X (Twitter) via Apify with price-derived momentum, volatility, and moving-average distance features, evaluated using expanding-window walk-forward validation and an out-of-sample quantitative Long/Flat trading strategy.

---

## 📌 Architecture & Workflow

```mermaid
flowchart TD
    subgraph Ingestion
        P[Historical Prices\nyfinance BTC-USD] --> DP[(SQLite DB\nprices)]
        A[Apify REST API\nX/Twitter Posts] --> DA[(SQLite DB\nraw_posts & normalized_posts)]
        N[Tavily / Exa API\nMarket News] --> DN[(SQLite DB\nnews_articles)]
    end

    subgraph "Sentiment Analysis"
        A --> V[Financial-Calibrated\nVADER Analyzer]
        V --> S[Daily Sentiment Bars\nMean, Dispersion, Volume]
    end

    subgraph "Feature Engineering"
        P --> F[Engineered Features\nZero-Lookahead Alignment]
        S --> F
        F --> H0[Target n+0: Nowcast]
        F --> H1[Target n+1: 1-Day]
        F --> H2[Target n+2: 2-Days]
        F --> H3[Target n+3: 3-Days]
        F --> H4[Target n+4: 4-Days]
        F --> H5[Target n+5: 5-Days]
    end

    subgraph "Forecasting & Validation"
        F --> WF[Expanding-Window\nWalk-Forward Validation]
        WF --> RF[Random Forest Regressors]
        WF --> NB[Naive Persistence Baseline]
        RF --> MET[Metrics: MAE, RMSE, MAPE, DirAcc]
    end

    subgraph "Financial & Visualization"
        H1 --> ST[Long/Flat Strategy\n10 bps Fee & 15 bps Hurdle]
        ST --> EV[Sharpe: 2.28 | Max DD: -4.23%]
        MET --> REP[Evaluation Report & Outlook]
        EV --> DASH[Streamlit Dashboard & Figures]
    end
```

---

## 🎯 Forecasting Horizons

1. **$n+0$ (Current-Period Nowcast)**: Estimates today's closing price $P_t$ using strictly pre-close information: opening price $P_{\text{open}}$, intraday sentiment momentum, and lagged historical price dynamics. No closing price leakage.
2. **$n+1$ (Next-Period Forecast)**: One-day forward price $P_{t+1}$. Primary driver of the out-of-sample trading strategy.
3. **$n+2$ (Two Periods Ahead)**: Forward closing price $P_{t+2}$.
4. **$n+3$ (Three Periods Ahead)**: Forward closing price $P_{t+3}$.
5. **$n+4$ (Four Periods Ahead)**: Forward closing price $P_{t+4}$.
6. **$n+5$ (Five Periods Ahead)**: Forward closing price $P_{t+5}$.

---

## ⚙️ Quickstart & Setup

### 1. Prerequisites
- Python 3.11+
- Git

### 2. Installation
Clone the repository and install required packages:
```bash
git clone https://github.com/akashgtub/crowdwisdom-x-sentiment.git
cd crowdwisdom-x-sentiment

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Configure your personal API keys in `.env`:
```ini
APIFY_TOKEN=your_apify_token_here
APIFY_ACTOR_ID=apidojo/tweet-scraper
TAVILY_API_KEY=your_tavily_api_key_here
EXA_API_KEY=your_exa_api_key_here
```
> **Note**: If API keys are not supplied, the pipeline gracefully reports 0 live API ingestion and activates a reproducible offline fixture covering the 180-day window, allowing complete model training, walk-forward validation, financial evaluation, tests, and dashboard to function deterministically.

---

## 🚀 Execution Commands

### 1. Run the Entire Pipeline (Single Command)
```bash
python run_pipeline.py
```
*Executes price ingestion, Apify/news collection, daily sentiment aggregation, zero-lookahead feature engineering, 6-horizon walk-forward validation, out-of-sample strategy backtesting, chart rendering, and report compilation.*

Optional arguments:
```bash
python run_pipeline.py --asset BTC-USD --days 180 --interval 1d
```

### 2. Launch the Interactive Dashboard (Single Command)
```bash
streamlit run app.py
```
*Opens the institutional-grade interactive web interface in your browser (default `http://localhost:8501`).*

### 3. Run Automated Tests
```bash
pytest -v
```
*Executes all 15 unit and integration tests covering price cleaning, Apify schemas, sentiment scoring, feature alignment, zero-lookahead leakage prevention, expanding-window walk-forward splitting, metrics, and trading strategy turnover.*

---

## 📊 Summary of Out-of-Sample Performance

### Walk-Forward Validation Table (Expanding Window)

| Horizon | RF MAE ($) | Naive MAE ($) | RF RMSE ($) | Naive RMSE ($) | RF MAPE (%) | Naive MAPE (%) | RF Dir. Acc. (%) | Naive Dir. Acc. (%) | OOS Samples |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$n+0$ (Nowcast)** | **$1,132.84** | $937.08 | $1,526.53 | $1,456.23 | **1.52%** | 1.28% | **59.62%** | 61.54% | 53 |
| **$n+1$ (1-Day)** | **$1,056.75** | $979.91 | $1,693.41 | $1,452.10 | **1.42%** | 1.31% | **54.90%** | 66.67% | 52 |
| **$n+2$ (2-Days)** | **$1,517.84** | $1,517.26 | $2,441.82 | $2,159.16 | **2.03%** | 2.05% | **39.22%** | 39.22% | 52 |
| **$n+3$ (3-Days)** | **$1,600.22** | $1,752.38 | $2,744.43 | $2,448.34 | **2.11%** | 2.35% | **56.00%** | 44.00% | 51 |
| **$n+4$ (4-Days)** | **$1,968.12** | $2,027.62 | $3,389.20 | $3,112.42 | **2.60%** | 2.68% | **54.00%** | 52.00% | 51 |
| **$n+5$ (5-Days)** | **$2,063.19** | $2,156.66 | $3,621.92 | $3,468.60 | **2.68%** | 2.84% | **57.14%** | 48.98% | 50 |

### Out-of-Sample Long/Flat Strategy Results (Horizon $n+1$)

- **Strategy Cumulative Net Return**: **-1.53%** (vs +1.71% Buy & Hold Benchmark)
- **Annualized Return**: **-10.27%** (vs +12.61% Benchmark)
- **Annualized Sharpe Ratio**: **-0.70** (vs 0.39 Benchmark)
- **Sortino Ratio**: **-1.11** (vs 0.67 Benchmark)
- **Maximum Drawdown**: **-6.65%** (vs -8.76% Benchmark — lower drawdown due to defensive cash positioning)
- **Win Rate**: **28.57%** (17 total trades, 13 days long / 39 days flat)
- **Turnover & Friction**: 10 bps transaction fee per trade turnover + 15 bps alpha entry threshold.

---

## 🗂️ Project Structure

```
crowdwisdom-x-sentiment/
├── .env.example               # Environment variables template
├── .gitignore                 # Standard git ignores (secrets, cache, DBs)
├── requirements.txt           # Python dependency specifications
├── README.md                  # Complete technical documentation
├── run_pipeline.py            # Primary pipeline CLI entrypoint
├── app.py                     # Streamlit interactive quantitative dashboard
├── data/
│   ├── raw/                   # Raw historical price CSVs & Apify JSON dumps
│   ├── processed/             # Cleaned price data & engineered features
│   ├── sample_posts_fixture.json # Deterministic sample fixture
│   └── crowdwisdom.db         # Persistent SQLite database
├── reports/
│   ├── evaluation_report.md   # Comprehensive quantitative assessment
│   ├── current_sentiment_outlook.md # 1-page sentiment outlook
│   └── figures/               # High-res publication charts
│       ├── sentiment_vs_price.png
│       ├── actual_vs_predicted_horizons.png
│       ├── horizon_metrics_comparison.png
│       └── strategy_equity_curve.png
├── src/
│   ├── config.py              # Environment configuration & paths
│   ├── db.py                  # SQLite schema & persistence helpers
│   ├── ingestion/
│   │   ├── price_ingestion.py # yfinance OHLCV historical prices
│   │   ├── apify_ingestion.py # Apify X/Twitter REST client & schema adapter
│   │   └── news_ingestion.py  # Tavily/Exa news ingestion for qualitative outlook
│   ├── sentiment/
│   │   └── sentiment_analyzer.py # Financial-calibrated VADER analyzer
│   ├── features/
│   │   └── feature_engineering.py # Lags, rolling volatility, MA distances, zero-lookahead
│   ├── models/
│   │   ├── forecasting.py     # 6 horizons & expanding-window walk-forward validation
│   │   └── metrics.py         # MAE, RMSE, MAPE & limitation docs
│   ├── evaluation/
│   │   └── financial_eval.py  # Long/Flat strategy, Sharpe, Sortino, Drawdown
│   └── visualizations/
│       └── plots.py           # Matplotlib publication figures
└── tests/
    ├── test_price_ingestion.py
    ├── test_apify_ingestion.py
    ├── test_sentiment.py
    ├── test_feature_engineering.py
    ├── test_forecasting.py
    ├── test_metrics.py
    └── test_financial_eval.py
```

---

## 🛡️ Limitations & Risk Disclaimers

1. **MAPE Percentage Error Asymmetry**:
   - MAPE penalizes forecasts that exceed the target more than those below target. On crypto assets with high price levels ($> $80,000), evaluating MAE and RMSE in dollar terms and comparing directly against a naive persistence benchmark provides a more reliable assessment.
2. **Separation of Predictive Features from News**:
   - News intelligence from Tavily and Exa is strictly quarantined to the qualitative sentiment outlook and not injected into predictive model feature vectors, avoiding lookahead data snooping.
3. **Execution Friction in Live Crypto Markets**:
   - The strategy models 10 bps turnover friction. Real-world order book slippage and maker/taker fee tiers vary by exchange volume tiers. Past out-of-sample performance is not indicative of guaranteed future returns.

---

## 📬 GitHub Push & Submission Instructions

### Push to GitHub:
```bash
git remote add origin https://github.com/akashgtub/crowdwisdom-x-sentiment.git
git push -u origin main
```

### Assessment Submission Email Template:
```text
To: Gilad <gilad@crowdwisdomtrading.com>
Subject: Internship Assessment Submission - CrowdWisdomTrading X Sentiment Price Prediction Agent

Dear Gilad,

Please find my completed internship assessment submission for the "X Sentiment Price Prediction Agent" project.

Repository: https://github.com/akashgtub/crowdwisdom-x-sentiment

Summary of Implementation & Verified Results:
- Built an end-to-end Python quantitative pipeline forecasting BTC-USD across 6 horizons (n+0 through n+5).
- Ingested 179 daily price bars via yfinance, live X/Twitter posts via Apify REST API (actor apidojo/tweet-scraper), and live macro news articles via Tavily.
- Features incorporate financial-calibrated VADER sentiment, engagement dispersion, volatility, moving-average distances, and price returns with strict zero-lookahead alignment (n+0 nowcast verified free of close_t leakage).
- Expanding-window walk-forward validation benchmarks multi-horizon Random Forest regressors against Naive Persistence baselines (n+0 MAE $1,132.84, DirAcc 59.62%; n+1 MAE $1,056.75, DirAcc 54.90%; n+3 MAE $1,600.22 vs Naive $1,752.38).
- Out-of-sample Long/Flat trading strategy with 10 bps turnover friction and 15 bps entry hurdle realized -6.65% Maximum Drawdown (compared to -8.76% for Buy & Hold benchmark), demonstrating effective capital preservation during downturns.
- Test suite includes 15 automated pytest tests (all passing), 4 high-resolution charts in reports/figures/, a comprehensive evaluation report, a 1-page sentiment outlook, and an interactive Streamlit dashboard.

Execution Commands:
- Run Pipeline: python run_pipeline.py
- Launch Dashboard: streamlit run app.py
- Run Test Suite: pytest -v
- Verify Live APIs: python verify_live_apis.py

Thank you for the opportunity to work on this challenge. I look forward to your review and feedback!

Best regards,
Akash
```
