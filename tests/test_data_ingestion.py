"""
Unit Tests for Data Ingestion and Atmospheric Baseline Loading
Author: Khushi Yadav (@khushiyadav35)
"""

import pytest
import pandas as pd
from src.data.ingestion import DataIngestionPipeline, load_dataset_for_city


class TestDataIngestion:
    """Tests data pipeline reliability and schema integrity."""

    def test_delhi_historical_dataset_schema(self):
        """Verifies Delhi historical repository integrity."""
        df = load_dataset_for_city("Delhi")
        assert len(df) > 1000
        required_cols = {"timestamp", "city", "pm25", "pm10", "no2", "so2", "co", "o3", "temperature", "wind_speed"}
        assert required_cols.issubset(set(df.columns))
        assert df["pm25"].isna().sum() == 0
        assert (df["pm25"] >= 0).all()

    def test_lucknow_historical_dataset_schema(self):
        """Verifies Lucknow historical repository integrity."""
        df = load_dataset_for_city("Lucknow")
        assert len(df) > 1000
        assert (df["city"] == "Lucknow").all()
        assert df["temperature"].mean() > 10.0

    def test_invalid_city_raises_value_error(self):
        """Ensures unknown city requests fail early."""
        with pytest.raises(ValueError):
            DataIngestionPipeline("Kolkata")
