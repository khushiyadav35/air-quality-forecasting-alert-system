"""
Prophet & Generalized Additive Atmospheric Forecaster
Decomposes air quality into trend, multi-period harmonics (diurnal, weekly, annual),
and exogenous meteorological drivers (temperature, wind speed, relative humidity).
Author: Khushi Yadav (@khushiyadav35)
"""

from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import joblib


class AdditiveProphetForecaster:
    """
    Generalized Additive Time-Series Model (inspired by Prophet architecture).
    Model Formulation:
        y(t) = g(t) [trend] + s_daily(t) + s_weekly(t) + s_annual(t) + beta^T * X_weather(t) + eps(t)
    Uses Fourier series basis functions to capture diurnal and seasonal cycles smoothly.
    """

    def __init__(self, target_col: str = "cpcb_aqi", alpha: float = 1.0):
        self.target_col = target_col
        self.alpha = alpha
        self.regressor = Ridge(alpha=alpha)
        self.ref_time: Optional[pd.Timestamp] = None
        self.residual_sigma: float = 15.0
        self.is_fitted = False

    def _build_design_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Constructs trend and Fourier basis columns."""
        timestamps = pd.to_datetime(df["timestamp"])
        if self.ref_time is None:
            self.ref_time = timestamps.min()

        # Continuous trend (hours elapsed from reference point)
        hours_elapsed = (timestamps - self.ref_time).dt.total_seconds() / 3600.0
        trend = hours_elapsed.values.reshape(-1, 1)

        # Diurnal Fourier Harmonics (Period = 24h, Order = 3)
        diurnal_features = []
        for k in range(1, 4):
            diurnal_features.append(np.sin(2 * np.pi * k * hours_elapsed / 24.0))
            diurnal_features.append(np.cos(2 * np.pi * k * hours_elapsed / 24.0))

        # Weekly Fourier Harmonics (Period = 168h, Order = 2)
        weekly_features = []
        for k in range(1, 3):
            weekly_features.append(np.sin(2 * np.pi * k * hours_elapsed / 168.0))
            weekly_features.append(np.cos(2 * np.pi * k * hours_elapsed / 168.0))

        # Annual Fourier Harmonics (Period = 8766h, Order = 4)
        annual_features = []
        for k in range(1, 5):
            annual_features.append(np.sin(2 * np.pi * k * hours_elapsed / 8766.0))
            annual_features.append(np.cos(2 * np.pi * k * hours_elapsed / 8766.0))

        # Exogenous Weather Features if present
        exog_list = []
        for col in ["temperature", "relative_humidity", "wind_speed"]:
            if col in df.columns:
                exog_list.append(df[col].fillna(df[col].median()).values)
            else:
                exog_list.append(np.zeros(len(df)))

        stacked = np.column_stack(
            [trend]
            + [f.values if hasattr(f, "values") else f for f in diurnal_features]
            + [f.values if hasattr(f, "values") else f for f in weekly_features]
            + [f.values if hasattr(f, "values") else f for f in annual_features]
            + exog_list
        )
        return stacked

    def fit(self, df: pd.DataFrame) -> "AdditiveProphetForecaster":
        """Fits trend, seasonality, and regressor coefficients."""
        X = self._build_design_matrix(df)
        y = df[self.target_col].values

        self.regressor.fit(X, y)
        preds = self.regressor.predict(X)
        self.residual_sigma = float(np.std(y - preds))
        self.is_fitted = True
        return self

    def predict_future(
        self,
        last_timestamp: pd.Timestamp,
        forecast_hours: int = 72,
        future_weather: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """
        Projects future trajectory for next H hours.
        Returns: (future_df, y_hat, y_lower, y_upper)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_future.")

        future_times = pd.date_range(
            start=last_timestamp + pd.Timedelta(hours=1),
            periods=forecast_hours,
            freq="h",
        )

        future_df = pd.DataFrame({"timestamp": future_times})
        if future_weather is not None:
            for col in ["temperature", "relative_humidity", "wind_speed"]:
                if col in future_weather.columns:
                    future_df[col] = future_weather[col].values[:forecast_hours]

        X_future = self._build_design_matrix(future_df)
        y_hat = np.clip(self.regressor.predict(X_future), a_min=15.0, a_max=None)

        # Uncertainty bounds expand over time
        growth = np.linspace(1.0, 1.7, forecast_hours)
        y_lower = np.clip(y_hat - 1.96 * self.residual_sigma * growth, a_min=10.0, a_max=None)
        y_upper = y_hat + 1.96 * self.residual_sigma * growth

        return future_df, y_hat, y_lower, y_upper

    def save(self, filepath: Path) -> None:
        """Serializes trained model parameters."""
        bundle = {
            "regressor": self.regressor,
            "ref_time": self.ref_time,
            "residual_sigma": self.residual_sigma,
            "target_col": self.target_col,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(bundle, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "AdditiveProphetForecaster":
        """Loads serialized model parameters."""
        bundle = joblib.load(filepath)
        instance = cls(target_col=bundle["target_col"])
        instance.regressor = bundle["regressor"]
        instance.ref_time = bundle["ref_time"]
        instance.residual_sigma = bundle["residual_sigma"]
        instance.is_fitted = bundle["is_fitted"]
        return instance
