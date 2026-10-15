"""
Unit tests for financial strategy evaluation.
"""

import pytest
import pandas as pd
import numpy as np
from src.evaluation.financial_eval import StrategyEvaluator

def test_long_flat_strategy_transaction_costs_and_returns():
    evaluator = StrategyEvaluator(transaction_cost_bps=10.0, threshold_bps=15.0)

    # Mock predictions where day 1 is bullish, day 2 is bearish, day 3 is bullish
    dates = pd.date_range("2026-06-01", periods=4, freq="D", tz="UTC")
    pred_df = pd.DataFrame({
        "actual": [100.0, 105.0, 103.0, 108.0],
        # day 0: price_t=100.0, pred=106.0 -> ret=+6% > 15bps (long, pos=1)
        # day 1: price_t=100.0 (shift), pred=95.0 -> ret=-5% < 15bps (flat, pos=0)
        # day 2: price_t=105.0 (shift), pred=115.0 -> ret=+9.5% > 15bps (long, pos=1)
        # day 3: price_t=103.0 (shift), pred=109.0 -> ret=+5.8% > 15bps (long, pos=1)
        "rf_predicted": [106.0, 95.0, 115.0, 109.0]
    }, index=dates)

    res = evaluator.evaluate_long_flat_strategy(pred_df)
    eval_df = res["evaluation_df"]

    # Positions should be: [1, 0, 1, 0 or 1]
    assert eval_df.loc[dates[0], "position"] == 1
    assert eval_df.loc[dates[1], "position"] == 0
    assert eval_df.loc[dates[2], "position"] == 1

    # Check that transaction fee was applied on position turnover
    assert eval_df["transaction_costs"].sum() > 0

    metrics = res["strategy_metrics"]
    assert "sharpe_ratio" in metrics
    assert "sortino_ratio" in metrics
    assert "max_drawdown_pct" in metrics
    assert metrics["max_drawdown_pct"] <= 0.0
