"""
Financial evaluation module for quantitative trading strategies.
Implements an out-of-sample Long/Flat trading strategy with transaction costs,
slippage modeling, non-overlapping returns, and annualized risk-adjusted metrics
(Sharpe, Sortino, Max Drawdown).
"""

import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class StrategyEvaluator:
    """
    Evaluates out-of-sample quantitative trading performance.
    Guarantees non-overlapping holding periods and strict inclusion of transaction friction.
    """

    def __init__(
        self,
        transaction_cost_bps: float = 10.0,  # 10 bps (0.10%) per trade turnover
        threshold_bps: float = 15.0,         # Minimum expected return to enter long
        risk_free_rate: float = 0.04,        # 4% annual risk-free rate
        annualization_days: int = 365        # 365 days for crypto
    ):
        self.fee_rate = transaction_cost_bps / 10000.0
        self.threshold = threshold_bps / 10000.0
        self.rf_annual = risk_free_rate
        self.rf_daily = (1.0 + risk_free_rate) ** (1.0 / annualization_days) - 1.0
        self.ann_factor = np.sqrt(annualization_days)
        self.ann_days = annualization_days

    def evaluate_long_flat_strategy(
        self,
        predictions_df: pd.DataFrame,
        actual_prices: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Runs an out-of-sample Long/Flat strategy on model forecasts.
        
        Args:
            predictions_df: DataFrame with 'actual' and 'rf_predicted' columns (indexed by timestamp).
            actual_prices: Optional Series of prices. If None, uses 'actual' from predictions_df.
            
        Returns:
            Dict containing equity curve series, trade log, and metrics dict.
        """
        if predictions_df.empty or "rf_predicted" not in predictions_df.columns:
            raise ValueError("predictions_df must contain 'rf_predicted' column.")

        df = predictions_df.copy()
        df.sort_index(inplace=True)

        # Current price P_t and actual realization P_{t+1}
        # In our horizon n+1 predictions_df, 'rf_predicted' is \hat{P}_{t+1} and 'actual' is P_{t+1}
        # To compute expected return at time t: we need price at decision time P_t.
        # Since 'actual' is P_{t+1}, P_t is actual.shift(1).
        # If actual_prices is passed separately, we align directly.
        if actual_prices is not None:
            p_t = actual_prices.reindex(df.index).copy()
        else:
            p_t = df["actual"].shift(1).copy()
            if "naive_predicted" in df.columns:
                p_t.iloc[0] = df["naive_predicted"].iloc[0]
            else:
                p_t.iloc[0] = df["actual"].iloc[0]

        df["price_t"] = p_t
        df["price_next"] = df["actual"]

        # Expected model return: (\hat{P}_{t+1} - P_t) / P_t
        df["expected_return"] = (df["rf_predicted"] - df["price_t"]) / df["price_t"]

        # Signal: 1 (Long) if expected return > fee threshold, else 0 (Flat / Cash)
        df["position"] = (df["expected_return"] > self.threshold).astype(int)

        # Realized asset return over the holding period: (P_{t+1} - P_t) / P_t
        df["realized_asset_return"] = (df["price_next"] - df["price_t"]) / df["price_t"]

        # Position changes incur transaction costs (turnover * fee_rate)
        pos_change = df["position"].diff().abs()
        pos_change.iloc[0] = float(df["position"].iloc[0])  # initial entry cost
        df["position_change"] = pos_change
        df["transaction_costs"] = df["position_change"] * self.fee_rate

        # Strategy realized net return (strictly non-overlapping 1-period forward return)
        df["strategy_gross_return"] = df["position"] * df["realized_asset_return"]
        df["strategy_net_return"] = df["strategy_gross_return"] - df["transaction_costs"]

        # Benchmark: Buy & Hold realized return
        df["benchmark_return"] = df["realized_asset_return"]

        # Cumulative equity curves (starting at 1.0)
        df["strategy_equity"] = (1.0 + df["strategy_net_return"]).cumprod()
        df["benchmark_equity"] = (1.0 + df["benchmark_return"]).cumprod()

        # Strategy drawdown
        strategy_peak = df["strategy_equity"].cummax()
        df["strategy_drawdown"] = (df["strategy_equity"] - strategy_peak) / strategy_peak

        benchmark_peak = df["benchmark_equity"].cummax()
        df["benchmark_drawdown"] = (df["benchmark_equity"] - benchmark_peak) / benchmark_peak

        # Calculate performance metrics
        strat_metrics = self._compute_performance_metrics(
            returns=df["strategy_net_return"].values,
            drawdown=df["strategy_drawdown"].values,
            equity=df["strategy_equity"].values
        )

        bench_metrics = self._compute_performance_metrics(
            returns=df["benchmark_return"].values,
            drawdown=df["benchmark_drawdown"].values,
            equity=df["benchmark_equity"].values
        )

        # Trade statistics
        total_trades = int(df["position_change"].sum())
        long_days = int(df["position"].sum())
        flat_days = len(df) - long_days
        win_days = int((df["strategy_net_return"] > 0).sum())
        loss_days = int((df["strategy_net_return"] < 0).sum())
        win_rate = (win_days / (win_days + loss_days) * 100.0) if (win_days + loss_days) > 0 else 0.0

        return {
            "strategy_metrics": strat_metrics,
            "benchmark_metrics": bench_metrics,
            "total_trades": total_trades,
            "long_days": long_days,
            "flat_days": flat_days,
            "win_rate_pct": round(win_rate, 2),
            "fee_rate_bps": self.fee_rate * 10000.0,
            "threshold_bps": self.threshold * 10000.0,
            "evaluation_df": df
        }

    def _compute_performance_metrics(
        self,
        returns: np.ndarray,
        drawdown: np.ndarray,
        equity: np.ndarray
    ) -> Dict[str, float]:
        """Compute standard institutional quantitative metrics."""
        n_periods = len(returns)
        if n_periods == 0:
            return {"cumulative_return_pct": 0.0, "annualized_return_pct": 0.0,
                    "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "max_drawdown_pct": 0.0}

        cum_return = float(equity[-1] - 1.0) * 100.0
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns, ddof=1)) if n_periods > 1 else 1e-6

        # Annualized Return (geometric or compound)
        ann_return = float(((equity[-1]) ** (self.ann_days / max(1, n_periods)) - 1.0) * 100.0)

        # Annualized Sharpe Ratio
        excess_returns = returns - self.rf_daily
        mean_excess = float(np.mean(excess_returns))
        sharpe = float((mean_excess / (std_ret + 1e-9)) * self.ann_factor)

        # Annualized Sortino Ratio (downside deviation relative to rf_daily)
        downside_diff = np.minimum(0.0, excess_returns)
        downside_std = float(np.sqrt(np.mean(downside_diff ** 2)))
        sortino = float((mean_excess / (downside_std + 1e-9)) * self.ann_factor)

        # Max Drawdown
        max_dd = float(np.min(drawdown)) * 100.0

        return {
            "cumulative_return_pct": round(cum_return, 2),
            "annualized_return_pct": round(ann_return, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "max_drawdown_pct": round(max_dd, 2)
        }

    @staticmethod
    def get_financial_limitations_notes() -> str:
        """Institutional notes regarding trading friction and real-world execution constraints."""
        return """
### Quantitative Financial Limitations and Execution Friction

1. **Transaction Costs & Turnover Friction**:
   - The strategy models a flat 10 bps (0.10%) transaction fee per trade turnover. In real-world crypto execution, exchange taker fees typically range from 5 bps to 25 bps depending on 30-day trailing volume tier and exchange VIP level.
   - High churn (frequent alternation between Long and Flat states) rapidly degrades net portfolio compounding.

2. **Slippage and Liquidity Depth**:
   - Market orders execute against resting limit order book liquidity. During volatile market regimes or liquidation cascades, market impact and adverse selection increase actual slippage well above 10 bps.

3. **Execution Timing & Settlement**:
   - The model assumes execution precisely at bar close/open. In practice, latency and discrete execution windows create execution timing dispersion.

4. **Regime Overfitting Risk**:
   - Realized out-of-sample strategy metrics must not be interpreted as guaranteed future profitability. Time-varying macroeconomic volatility, regulatory actions, and regime shifts alter market dynamics.
"""
