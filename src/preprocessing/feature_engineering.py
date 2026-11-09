"""
Feature Engineering and Sequence Creation for Time-Series Forecasting
Author: Khushi Yadav (@khushiyadav35)
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
from pathlib import Path

from config.settings import MODELS_DIR


class TimeSeriesFeatureEngineer:
    """
    Transforms raw atmospheric observations into structured supervised learning matrices:
    - Lag features capturing temporal memory
    - Rolling window statistics capturing recent trends
    - Trigonometric periodic embeddings for diurnal and seasonal cycles
    - Multi-step sequence creation for Recurrent Neural Networks (LSTM)
    """

    def __init__(self, target_col: str = "cpcb_aqi"):
        self.target_col = target_col
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns: List[str] = []
        self.is_fitted = False

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Constructs temporal, lag, rolling, and meteorological features."""
        data = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
            data["timestamp"] = pd.to_datetime(data["timestamp"])

        data = data.sort_values("timestamp").reset_index(drop=True)

        # 1. Cyclical Time Encodings (Harmonics)
        hour = data["timestamp"].dt.hour
        day_of_week = data["timestamp"].dt.dayofweek
        day_of_year = data["timestamp"].dt.dayofyear

        data["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
        data["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
        data["sin_dow"] = np.sin(2 * np.pi * day_of_week / 7.0)
        data["cos_dow"] = np.cos(2 * np.pi * day_of_week / 7.0)
        data["sin_doy"] = np.sin(2 * np.pi * day_of_year / 365.25)
        data["cos_doy"] = np.cos(2 * np.pi * day_of_year / 365.25)

        # 2. Meteorological Inversion & Dispersion Proxies
        if "wind_speed" in data.columns:
            # Low wind speed creates air stagnation
            data["stagnation_index"] = 1.0 / (data["wind_speed"].clip(lower=0.1))
        if "temperature" in data.columns and "relative_humidity" in data.columns:
            # Dew point / Fog index proxy
            data["fog_propensity"] = (data["relative_humidity"] / 100.0) * (30.0 - data["temperature"]).clip(lower=0)

        # 3. Lag Features
        lag_steps = [1, 2, 3, 6, 12, 24, 48, 72]
        for lag in lag_steps:
            data[f"{self.target_col}_lag_{lag}"] = data[self.target_col].shift(lag)
            if "pm25" in data.columns:
                data[f"pm25_lag_{lag}"] = data["pm25"].shift(lag)

        # 4. Rolling Window Statistics
        rolling_windows = [6, 12, 24]
        for window in rolling_windows:
            data[f"{self.target_col}_roll_mean_{window}"] = data[self.target_col].shift(1).rolling(window=window).mean()
            data[f"{self.target_col}_roll_std_{window}"] = data[self.target_col].shift(1).rolling(window=window).std()
            data[f"{self.target_col}_roll_max_{window}"] = data[self.target_col].shift(1).rolling(window=window).max()

        # Drop initial rows with NaNs resulting from 72h lags
        data = data.dropna().reset_index(drop=True)
        return data

    def get_feature_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Separates features and target variable."""
        ignore_cols = {"timestamp", "city", "aqi_category", "prominent_pollutant"}
        candidate_cols = [c for c in df.columns if c not in ignore_cols and c != self.target_col]

        self.feature_columns = candidate_cols
        X = df[self.feature_columns].values
        y = df[[self.target_col]].values
        return X, y

    def fit_scalers(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits normalization scalers on training fold only."""
        self.feature_scaler.fit(X)
        self.target_scaler.fit(y)
        self.is_fitted = True

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Scales inputs into [0, 1] interval."""
        if not self.is_fitted:
            raise RuntimeError("Scalers must be fitted before calling transform.")
        X_scaled = self.feature_scaler.transform(X)
        y_scaled = self.target_scaler.transform(y) if y is not None else None
        return X_scaled, y_scaled

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """Reverts scaled predictions back to actual NAQI scale."""
        if not self.is_fitted:
            raise RuntimeError("Target scaler is not fitted.")
        original_shape = y_scaled.shape
        reshaped = y_scaled.reshape(-1, 1)
        inversed = self.target_scaler.inverse_transform(reshaped)
        return inversed.reshape(original_shape)

    def create_lstm_sequences(
        self,
        X_scaled: np.ndarray,
        y_scaled: np.ndarray,
        lookback_window: int = 72,
        forecast_horizon: int = 24,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates 3D input tensor sequences for LSTM:
        X_seq: (num_samples, lookback_window, num_features)
        y_seq: (num_samples, forecast_horizon)
        """
        num_samples = len(X_scaled) - lookback_window - forecast_horizon + 1
        if num_samples <= 0:
            raise ValueError("Sequence length + forecast horizon exceeds dataset size.")

        X_seq = np.empty((num_samples, lookback_window, X_scaled.shape[1]), dtype=np.float32)
        y_seq = np.empty((num_samples, forecast_horizon), dtype=np.float32)

        for i in range(num_samples):
            X_seq[i] = X_scaled[i : i + lookback_window]
            y_seq[i] = y_scaled[i + lookback_window : i + lookback_window + forecast_horizon, 0]

        return X_seq, y_seq

    def save(self, filepath: Path) -> None:
        """Saves fitted scalers and metadata."""
        bundle = {
            "feature_scaler": self.feature_scaler,
            "target_scaler": self.target_scaler,
            "feature_columns": self.feature_columns,
            "target_col": self.target_col,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(bundle, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "TimeSeriesFeatureEngineer":
        """Loads serialized feature engineer."""
        bundle = joblib.load(filepath)
        instance = cls(target_col=bundle["target_col"])
        instance.feature_scaler = bundle["feature_scaler"]
        instance.target_scaler = bundle["target_scaler"]
        instance.feature_columns = bundle["feature_columns"]
        instance.is_fitted = bundle["is_fitted"]
        return instance
