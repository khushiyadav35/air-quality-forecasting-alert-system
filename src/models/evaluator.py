"""
Time-Series Evaluation Metrics and Atmospheric Performance Benchmarking
Author: Khushi Yadav (@khushiyadav35)
"""

from typing import Dict, Any, Union
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ForecastingEvaluator:
    """Computes comprehensive regression and peak-detection accuracy metrics."""

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculates:
        - MAE (Mean Absolute Error)
        - RMSE (Root Mean Squared Error)
        - MAPE (Mean Absolute Percentage Error)
        - R2 (Coefficient of Determination)
        - High AQI Recall (Ability to flag AQI > 200 episodes)
        """
        y_true_flat = np.asarray(y_true, dtype=float).ravel()
        y_pred_flat = np.asarray(y_pred, dtype=float).ravel()

        mask = ~np.isnan(y_true_flat) & ~np.isnan(y_pred_flat)
        y_t = y_true_flat[mask]
        y_p = y_pred_flat[mask]

        if len(y_t) == 0:
            return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "r2": 0.0, "spike_recall": 0.0}

        mae = float(mean_absolute_error(y_t, y_p))
        rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))

        # MAPE with epsilon to avoid division by zero
        epsilon = 1e-5
        mape = float(np.mean(np.abs((y_t - y_p) / np.maximum(y_t, epsilon))) * 100.0)

        r2 = float(r2_score(y_t, y_p))

        # Peak detection: evaluate binary classification at Poor/Severe threshold (> 200 AQI)
        true_spikes = (y_t >= 200.0)
        pred_spikes = (y_p >= 200.0)
        if true_spikes.sum() > 0:
            spike_recall = float((true_spikes & pred_spikes).sum() / true_spikes.sum())
        else:
            spike_recall = 1.0

        return {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 2),
            "r2": round(r2, 4),
            "spike_recall": round(spike_recall * 100.0, 1),
        }

    @staticmethod
    def format_summary_table(metrics_dict: Dict[str, Dict[str, Any]]) -> str:
        """Formats model benchmarking results into a markdown table."""
        header = "| Model | Horizon | MAE (AQI) | RMSE (AQI) | MAPE (%) | R² Score | Spike Recall (>200) |\n"
        separator = "|---|---|---|---|---|---|---|\n"
        rows = []
        for model_name, horizons in metrics_dict.items():
            for horizon, m in horizons.items():
                rows.append(
                    f"| {model_name} | {horizon} | {m['mae']} | {m['rmse']} | {m['mape']}% | {m['r2']} | {m['spike_recall']}% |"
                )
        return header + separator + "\n".join(rows)
