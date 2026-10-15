"""
Evaluation metrics module for time-series forecasting.
Computes MAE, RMSE, MAPE, Directional Accuracy, and provides
comprehensive explanations of metric properties and limitations.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute MAE, RMSE, MAPE, and Directional Accuracy.
    
    y_true: Array of ground-truth prices
    y_pred: Array of model predicted prices
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Filter out NaNs if any
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t = y_true[valid_mask]
    y_p = y_pred[valid_mask]

    if len(y_t) == 0:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "directional_accuracy": 0.0, "sample_count": 0}

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))

    # MAPE: Mean Absolute Percentage Error
    # Avoid division by zero with small epsilon
    denom = np.where(np.abs(y_t) < 1e-6, 1e-6, np.abs(y_t))
    mape = float(np.mean(np.abs((y_t - y_p) / denom)) * 100.0)

    # Directional Accuracy (comparing predicted step changes against actual step changes)
    if len(y_t) > 1:
        actual_diff = np.diff(y_t)
        pred_diff = np.diff(y_p)
        correct_directions = (np.sign(actual_diff) == np.sign(pred_diff))
        dir_acc = float(np.mean(correct_directions) * 100.0)
    else:
        dir_acc = 50.0

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "directional_accuracy": round(dir_acc, 2),
        "sample_count": len(y_t)
    }

def get_metric_limitations_explanation() -> str:
    """
    Returns an academic and practical explanation of error metric limitations
    in financial time series forecasting, specifically addressing MAPE.
    """
    return """
### Metric Interpretation and Critical Limitations

1. **MAPE (Mean Absolute Percentage Error)**:
   - **Asymmetry Bias**: MAPE treats positive and negative prediction errors unequally. Over-predictions are penalized more heavily than under-predictions because percentage error is bounded on the downside by 100% but unbounded on the upside.
   - **Zero/Small Denominator Sensitivity**: When the true price or return approaches zero, MAPE approaches infinity. In crypto forecasting, while BTC prices are large (> $80,000), evaluating MAPE on raw prices can mask large dollar drawdowns (e.g., a 2% error on Bitcoin is ~$1,600/coin, which is substantial for trading strategies).
   - **Scale Invariance vs Portfolio Impact**: A 1.5% MAPE might appear quantitatively acceptable, yet if prediction errors consistently occur at local tops and bottoms, the downstream trading strategy will suffer severe whipsaws.

2. **MAE (Mean Absolute Error)**:
   - Expressed directly in asset currency (USD). Provides a linear, easily interpretable penalty for dollar deviations, robust to rare market outliers.

3. **RMSE (Root Mean Squared Error)**:
   - Heavily penalizes large forecasting misses due to quadratic squaring. In financial risk management, large sudden errors (e.g., missing a 10% flash crash) carry catastrophic liquidation risk, making RMSE a vital metric for tail risk assessment.

4. **Directional Accuracy**:
   - Assesses whether the model correctly anticipates the sign of the price movement. For trading strategies, directional sign accuracy often correlates more strongly with trading profitability than pure point-forecast dollar minimization.
"""
