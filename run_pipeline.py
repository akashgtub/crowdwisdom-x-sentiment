"""
Root CLI entry point to run the CrowdWisdomTrading pipeline.
Usage:
    python run_pipeline.py
    python run_pipeline.py --asset BTC-USD --days 180
"""

import argparse
import sys
from src.config import DEFAULT_ASSET, DEFAULT_HISTORY_DAYS, DEFAULT_INTERVAL
from src.pipeline import TradingSentimentPipeline

def main():
    parser = argparse.ArgumentParser(
        description="CrowdWisdomTrading: X Sentiment Price Prediction Agent"
    )
    parser.add_argument(
        "--asset",
        type=str,
        default=DEFAULT_ASSET,
        help=f"Asset ticker symbol (default: {DEFAULT_ASSET})"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_HISTORY_DAYS,
        help=f"Historical lookback days (default: {DEFAULT_HISTORY_DAYS})"
    )
    parser.add_argument(
        "--interval",
        type=str,
        default=DEFAULT_INTERVAL,
        help=f"Price bar interval (default: {DEFAULT_INTERVAL})"
    )

    args = parser.parse_args()

    pipeline = TradingSentimentPipeline(
        ticker=args.asset,
        history_days=args.days,
        interval=args.interval
    )
    results = pipeline.run()

    if results.get("status") == "success":
        print("\n========================================================")
        print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("========================================================")
        print(f"Asset: {results['asset']} ({results['price_bars']} price bars)")
        print(f"Execution time: {results['elapsed_seconds']:.1f}s")
        print("\nOut-of-sample Forecast Performance Summary:")
        print(results["summary_metrics_df"].to_string(index=False))
        print("\nRealized Financial Strategy (Horizon n+1):")
        sm = results["strategy_evaluation"]["strategy_metrics"]
        bm = results["strategy_evaluation"]["benchmark_metrics"]
        print(f" - Strategy Sharpe Ratio: {sm['sharpe_ratio']} (Benchmark: {bm['sharpe_ratio']})")
        print(f" - Cumulative Return:     {sm['cumulative_return_pct']}% (Benchmark: {bm['cumulative_return_pct']}%)")
        print(f" - Max Drawdown:          {sm['max_drawdown_pct']}% (Benchmark: {bm['max_drawdown_pct']}%)")
        print(f" - Win Rate:              {results['strategy_evaluation']['win_rate_pct']}%")
        print("\nCurrent Horizon Projections:")
        for h, p in results["current_predictions"].items():
            print(f" - {h}: ${p:,.2f}")
        print("\nGenerated Reports:")
        for r in results["report_paths"]:
            print(f" - {r}")
        print("Generated Figures:")
        for f in results["figure_paths"]:
            print(f" - {f}")
        print("========================================================\n")
        sys.exit(0)
    else:
        print("Pipeline execution failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
