"""
Model Training and Cross-Horizon Benchmarking Pipeline
Trains PyTorch LSTM neural networks and Prophet/Additive models for Delhi and Lucknow.
Author: Khushi Yadav (@khushiyadav35)
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import json
import logging
import numpy as np
import pandas as pd

from config.settings import MODELS_DIR, RAW_DATA_DIR, LOOKBACK_WINDOW_HOURS, FORECAST_HORIZON_24H
from src.data.ingestion import load_dataset_for_city
from src.preprocessing.cpcb_calculator import CPCBCalculator
from src.preprocessing.feature_engineering import TimeSeriesFeatureEngineer
from src.models.lstm_forecaster import LSTMTrainer
from src.models.prophet_forecaster import AdditiveProphetForecaster
from src.models.evaluator import ForecastingEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_and_benchmark_city(city_name: str = "Delhi") -> dict:
    """Trains and benchmarks LSTM & Prophet models on a chronological 80/20 train-test split."""
    logger.info(f"=== Starting Model Training Pipeline for {city_name} ===")

    # 1. Load data and compute ground truth CPCB NAQI
    raw_df = load_dataset_for_city(city_name)
    logger.info(f"Loaded {len(raw_df)} raw records for {city_name}. Computing CPCB NAQI...")
    df_with_naqi = CPCBCalculator.compute_dataframe_naqi(raw_df)

    # 2. Chronological Train-Test Split (80% Train, 20% Test)
    split_idx = int(len(df_with_naqi) * 0.80)
    train_df = df_with_naqi.iloc[:split_idx].copy()
    test_df = df_with_naqi.iloc[split_idx:].copy()
    logger.info(f"Train split: {len(train_df)} hrs | Test split: {len(test_df)} hrs")

    # 3. Train Additive / Prophet Forecaster
    logger.info(f"Fitting Additive/Prophet Forecaster for {city_name}...")
    prophet_model = AdditiveProphetForecaster(target_col="cpcb_aqi")
    prophet_model.fit(train_df)
    prophet_path = MODELS_DIR / f"prophet_{city_name.lower()}.pkl"
    prophet_model.save(prophet_path)
    logger.info(f"Saved Prophet checkpoint to {prophet_path}")

    # Evaluate Prophet on test set
    test_ref_time = train_df["timestamp"].iloc[-1]
    _, p_preds, _, _ = prophet_model.predict_future(
        last_timestamp=test_ref_time,
        forecast_hours=min(len(test_df), 168), # test on 7-day horizon
        future_weather=test_df[["temperature", "relative_humidity", "wind_speed"]].head(168),
    )
    actual_test_slice = test_df["cpcb_aqi"].values[: len(p_preds)]
    prophet_metrics = ForecastingEvaluator.compute_metrics(actual_test_slice, p_preds)

    # 4. Feature Engineering for PyTorch LSTM
    logger.info(f"Engineering temporal lag and rolling features for LSTM...")
    fe = TimeSeriesFeatureEngineer(target_col="cpcb_aqi")
    train_feats = fe.engineer_features(train_df)
    test_feats = fe.engineer_features(test_df)

    X_train_raw, y_train_raw = fe.get_feature_matrix(train_feats)
    X_test_raw, y_test_raw = fe.get_feature_matrix(test_feats)

    fe.fit_scalers(X_train_raw, y_train_raw)
    X_train_scaled, y_train_scaled = fe.transform(X_train_raw, y_train_raw)
    X_test_scaled, y_test_scaled = fe.transform(X_test_raw, y_test_raw)

    fe_path = MODELS_DIR / f"feature_engineer_{city_name.lower()}.pkl"
    fe.save(fe_path)
    logger.info(f"Saved Feature Engineer to {fe_path}")

    # 5. Create LSTM sequences (Lookback = 72h, Forecast = 24h)
    X_train_seq, y_train_seq = fe.create_lstm_sequences(
        X_train_scaled, y_train_scaled, lookback_window=LOOKBACK_WINDOW_HOURS, forecast_horizon=FORECAST_HORIZON_24H
    )
    X_test_seq, y_test_seq = fe.create_lstm_sequences(
        X_test_scaled, y_test_scaled, lookback_window=LOOKBACK_WINDOW_HOURS, forecast_horizon=FORECAST_HORIZON_24H
    )

    # Validation split from training sequences (10%)
    val_split = int(len(X_train_seq) * 0.90)
    X_tr, y_tr = X_train_seq[:val_split], y_train_seq[:val_split]
    X_va, y_va = X_train_seq[val_split:], y_train_seq[val_split:]

    # 6. Train PyTorch Deep LSTM
    input_dim = X_train_seq.shape[2]
    logger.info(f"Training PyTorch LSTM (Input Dim={input_dim}, Hidden=64, Seq=72h, Horizon=24h)...")
    lstm_trainer = LSTMTrainer(
        input_dim=input_dim,
        forecast_horizon=FORECAST_HORIZON_24H,
        hidden_dim=64,
        num_layers=2,
        learning_rate=0.002,
    )
    history = lstm_trainer.train_epochs(X_tr, y_tr, X_va, y_va, epochs=15, batch_size=128)
    logger.info(f"LSTM Training complete. Final Val Loss: {history['val_loss'][-1]:.4f}")

    lstm_path = MODELS_DIR / f"lstm_{city_name.lower()}_24h.pt"
    lstm_trainer.save_checkpoint(lstm_path)
    logger.info(f"Saved PyTorch LSTM checkpoint to {lstm_path}")

    # 7. Evaluate LSTM on unseen Test Set
    scaled_test_preds = lstm_trainer.predict(X_test_seq[:1000])
    actual_test_targets = fe.inverse_transform_target(y_test_seq[:1000])
    pred_test_targets = fe.inverse_transform_target(scaled_test_preds)

    lstm_24h_metrics = ForecastingEvaluator.compute_metrics(actual_test_targets, pred_test_targets)

    return {
        "city": city_name,
        "metrics": {
            "PyTorch LSTM": {"24h Horizon": lstm_24h_metrics},
            "Prophet / Additive": {"Multi-Day": prophet_metrics},
        },
    }


def main():
    benchmarks = {}
    for city in ["Delhi", "Lucknow"]:
        result = train_and_benchmark_city(city)
        benchmarks[city] = result["metrics"]

    # Save benchmark metrics to JSON
    meta_path = MODELS_DIR / "benchmark_metrics.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2)
    logger.info(f"Saved benchmarking results to {meta_path}")

    print("\n================== BENCHMARK REPORT ==================")
    for city, models in benchmarks.items():
        print(f"\n[*] City: {city.upper()}")
        print(ForecastingEvaluator.format_summary_table(models))
    print("======================================================")


if __name__ == "__main__":
    main()
