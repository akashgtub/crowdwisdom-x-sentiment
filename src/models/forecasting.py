"""
Forecasting module for CrowdWisdomTrading pipeline.
Trains separate regression models for horizons n+0 through n+5 using
expanding-window walk-forward validation without data leakage or shuffling.
Compares machine learning models against a naive persistence baseline.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src.features.feature_engineering import get_feature_columns_for_horizon
from src.models.metrics import calculate_metrics
from src.db import save_metrics

logger = logging.getLogger(__name__)

HORIZONS = ["n+0", "n+1", "n+2", "n+3", "n+4", "n+5"]

class MultiHorizonForecaster:
    """
    Manages training, expanding-window walk-forward validation,
    and out-of-sample predictions across horizons n+0 to n+5.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        min_train_size: int = 50,
        step_size: int = 1,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_train_size = min_train_size
        self.step_size = step_size
        self.random_state = random_state

    def run_walk_forward_validation(
        self,
        feature_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Executes expanding-window walk-forward validation across all 6 horizons.
        
        Returns:
            Dict containing:
            - 'horizon_results': Dict per horizon with 'rf_metrics', 'naive_metrics', 'predictions_df', 'feature_importances'
            - 'summary_metrics_df': Consolidated metrics DataFrame
        """
        if len(feature_df) < self.min_train_size + 15:
            raise ValueError(f"Insufficient historical data: {len(feature_df)} rows. Need at least {self.min_train_size + 15}.")

        horizon_results: Dict[str, Any] = {}
        all_metrics_to_db = []

        for h_idx, horizon in enumerate(HORIZONS):
            target_col = f"target_n{h_idx}"
            if target_col not in feature_df.columns:
                logger.warning("Target %s not found in feature dataset. Skipping.", target_col)
                continue

            feature_cols = get_feature_columns_for_horizon(horizon)
            # Filter valid rows where target is present (forward shifts have NaNs at the end)
            valid_df = feature_df.dropna(subset=[target_col]).copy()
            valid_df = valid_df[valid_df[target_col] > 0]

            n_samples = len(valid_df)
            if n_samples < self.min_train_size + 10:
                logger.warning("Insufficient samples (%d) for horizon %s.", n_samples, horizon)
                continue

            preds_rf = []
            preds_naive = []
            actuals = []
            pred_dates = []

            # Expanding window walk-forward iteration
            # Train on [0 : t], test on [t]
            for t in range(self.min_train_size, n_samples, self.step_size):
                train_data = valid_df.iloc[:t]
                test_row = valid_df.iloc[t:t+1]

                X_train = train_data[feature_cols].values
                y_train = train_data[target_col].values

                X_test = test_row[feature_cols].values
                y_test = float(test_row[target_col].values[0])

                # Train ML model
                model = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    min_samples_split=4,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                y_pred_rf = float(model.predict(X_test)[0])

                # Naive baseline:
                # For n+0: naive forecast is today's open price (or prior close)
                # For n+1..n+5: naive persistence forecast is current close price
                if horizon == "n+0":
                    y_pred_naive = float(test_row["open"].values[0])
                else:
                    y_pred_naive = float(test_row["close"].values[0])

                preds_rf.append(y_pred_rf)
                preds_naive.append(y_pred_naive)
                actuals.append(y_test)
                pred_dates.append(test_row.index[0])

            # Convert out-of-sample walk-forward predictions to arrays
            y_actual_arr = np.array(actuals)
            y_rf_arr = np.array(preds_rf)
            y_naive_arr = np.array(preds_naive)

            rf_metrics = calculate_metrics(y_actual_arr, y_rf_arr)
            naive_metrics = calculate_metrics(y_actual_arr, y_naive_arr)

            # Fit final model on all data for feature importances and future forecasting
            final_model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=4,
                random_state=self.random_state,
                n_jobs=-1
            )
            final_model.fit(valid_df[feature_cols].values, valid_df[target_col].values)
            importances = dict(zip(feature_cols, final_model.feature_importances_))
            sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True)[:10])

            # Store predictions DataFrame
            pred_df = pd.DataFrame({
                "actual": y_actual_arr,
                "rf_predicted": y_rf_arr,
                "naive_predicted": y_naive_arr
            }, index=pd.DatetimeIndex(pred_dates, name="timestamp"))

            horizon_results[horizon] = {
                "horizon": horizon,
                "rf_metrics": rf_metrics,
                "naive_metrics": naive_metrics,
                "predictions_df": pred_df,
                "feature_importances": sorted_importances,
                "sample_count": len(actuals),
                "final_model": final_model
            }

            all_metrics_to_db.append({
                "model_name": "RandomForest",
                "horizon": horizon,
                "mae": rf_metrics["mae"],
                "rmse": rf_metrics["rmse"],
                "mape": rf_metrics["mape"],
                "directional_accuracy": rf_metrics["directional_accuracy"]
            })
            all_metrics_to_db.append({
                "model_name": "NaivePersistence",
                "horizon": horizon,
                "mae": naive_metrics["mae"],
                "rmse": naive_metrics["rmse"],
                "mape": naive_metrics["mape"],
                "directional_accuracy": naive_metrics["directional_accuracy"]
            })

            logger.info("Horizon %s - RF MAE: $%.2f (MAPE: %.2f%%) | Naive MAE: $%.2f (MAPE: %.2f%%)",
                        horizon, rf_metrics["mae"], rf_metrics["mape"],
                        naive_metrics["mae"], naive_metrics["mape"])

        # Persist metrics to SQLite
        save_metrics(all_metrics_to_db)

        # Build summary comparison DataFrame
        summary_rows = []
        for h in HORIZONS:
            if h in horizon_results:
                rf_m = horizon_results[h]["rf_metrics"]
                na_m = horizon_results[h]["naive_metrics"]
                summary_rows.append({
                    "Horizon": h,
                    "RF_MAE": rf_m["mae"],
                    "Naive_MAE": na_m["mae"],
                    "RF_RMSE": rf_m["rmse"],
                    "Naive_RMSE": na_m["rmse"],
                    "RF_MAPE_%": rf_m["mape"],
                    "Naive_MAPE_%": na_m["mape"],
                    "RF_DirAcc_%": rf_m["directional_accuracy"],
                    "Naive_DirAcc_%": na_m["directional_accuracy"],
                    "OOS_Samples": rf_m["sample_count"]
                })

        summary_metrics_df = pd.DataFrame(summary_rows)

        return {
            "horizon_results": horizon_results,
            "summary_metrics_df": summary_metrics_df
        }

    def predict_current_horizons(
        self,
        feature_df: pd.DataFrame,
        horizon_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Generate forward predictions from the latest available bar for horizons n+0 to n+5.
        """
        latest_row = feature_df.iloc[-1:]
        current_predictions = {}

        for horizon in HORIZONS:
            if horizon not in horizon_results:
                continue
            model = horizon_results[horizon]["final_model"]
            feature_cols = get_feature_columns_for_horizon(horizon)
            X_latest = latest_row[feature_cols].values
            pred_price = float(model.predict(X_latest)[0])
            current_predictions[horizon] = round(pred_price, 2)

        return current_predictions
