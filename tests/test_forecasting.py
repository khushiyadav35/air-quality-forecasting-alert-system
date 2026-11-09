"""
Unit Tests for Forecasting Models (PyTorch LSTM and Prophet)
Author: Khushi Yadav (@khushiyadav35)
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from config.settings import MODELS_DIR
from src.preprocessing.feature_engineering import TimeSeriesFeatureEngineer
from src.models.lstm_forecaster import LSTMTrainer
from src.models.prophet_forecaster import AdditiveProphetForecaster
from src.models.evaluator import ForecastingEvaluator


class TestForecastingModels:
    """Verifies recurrent sequence construction and inference output shapes."""

    def test_lstm_checkpoint_loading_and_inference(self):
        """Verifies PyTorch LSTM model checkpoint can be loaded and outputs 24h predictions."""
        lstm_path = MODELS_DIR / "lstm_delhi_24h.pt"
        fe_path = MODELS_DIR / "feature_engineer_delhi.pkl"

        if lstm_path.exists() and fe_path.exists():
            lstm_trainer = LSTMTrainer.load_checkpoint(lstm_path)
            fe = TimeSeriesFeatureEngineer.load(fe_path)

            # Create dummy input sequence of shape (1, 72, input_dim)
            dummy_seq = np.random.uniform(0, 1, size=(1, 72, lstm_trainer.model.input_dim)).astype(np.float32)
            preds = lstm_trainer.predict(dummy_seq)

            assert preds.shape == (1, 24)
            y_hat, y_low, y_up = lstm_trainer.predict_with_intervals(dummy_seq, fe)
            assert len(y_hat[0]) == 24
            assert (y_up >= y_low).all()

    def test_prophet_model_future_projection(self):
        """Verifies Prophet/Additive model produces future trajectory and intervals."""
        prophet_path = MODELS_DIR / "prophet_lucknow.pkl"
        if prophet_path.exists():
            prophet = AdditiveProphetForecaster.load(prophet_path)
            now = pd.Timestamp.now()
            future_df, y_hat, y_low, y_up = prophet.predict_future(now, forecast_hours=48)

            assert len(future_df) == 48
            assert len(y_hat) == 48
            assert (y_hat >= 0).all()
            assert (y_up >= y_low).all()

    def test_evaluation_metrics_calculation(self):
        """Verifies MAE, RMSE, MAPE, and Spike Recall computations."""
        y_true = np.array([210.0, 240.0, 180.0, 310.0])
        y_pred = np.array([215.0, 230.0, 185.0, 300.0])

        metrics = ForecastingEvaluator.compute_metrics(y_true, y_pred)
        assert metrics["mae"] > 0
        assert metrics["rmse"] >= metrics["mae"]
        assert metrics["r2"] > 0.8
        assert metrics["spike_recall"] == 100.0
