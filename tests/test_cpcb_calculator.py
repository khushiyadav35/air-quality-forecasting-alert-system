"""
Unit Tests for Official Indian CPCB NAQI Calculation Engine
Author: Khushi Yadav (@khushiyadav35)
"""

import pytest
import numpy as np
from src.preprocessing.cpcb_calculator import CPCBCalculator


class TestCPCBCalculator:
    """Validates piecewise linear interpolation and criteria compliance."""

    def test_pm25_exact_breakpoints(self):
        """Tests exact CPCB PM2.5 boundary transitions."""
        assert CPCBCalculator.calculate_sub_index("pm25", 0.0) == 0.0
        assert CPCBCalculator.calculate_sub_index("pm25", 30.0) == 50.0
        assert CPCBCalculator.calculate_sub_index("pm25", 60.0) == 100.0
        assert CPCBCalculator.calculate_sub_index("pm25", 90.0) == 200.0
        assert CPCBCalculator.calculate_sub_index("pm25", 120.0) == 300.0
        assert CPCBCalculator.calculate_sub_index("pm25", 250.0) == 400.0
        assert CPCBCalculator.calculate_sub_index("pm25", 500.0) == 500.0

    def test_pm10_linear_interpolation(self):
        """Tests midpoint interpolation for PM10."""
        # 0 to 50 maps to 0 to 50
        assert CPCBCalculator.calculate_sub_index("pm10", 25.0) == 25.0
        # 50.1 to 100 maps to 51 to 100
        mid = CPCBCalculator.calculate_sub_index("pm10", 75.0)
        assert 74.0 <= mid <= 76.0

    def test_overall_aqi_prominent_pollutant(self):
        """Tests maximum sub-index extraction and prominent pollutant assignment."""
        pollutants = {
            "pm25": 110.0,  # in 90-120 range -> Poor ~266
            "pm10": 80.0,   # Satisfactory
            "no2": 30.0,    # Good
            "so2": 15.0,    # Good
        }
        res = CPCBCalculator.calculate_overall_aqi(pollutants)
        assert res["is_strictly_valid"] is True
        assert res["category"] == "Poor"
        assert res["prominent_pollutant"] == "PM25"
        assert 201 <= res["aqi"] <= 300

    def test_cpcb_three_pollutant_rule(self):
        """Verifies criteria: minimum 3 pollutants with at least one particulate."""
        # Only 2 pollutants
        res_few = CPCBCalculator.calculate_overall_aqi({"pm25": 45.0, "no2": 20.0})
        assert res_few["is_strictly_valid"] is False

        # 3 pollutants but NO particulate
        res_no_particulate = CPCBCalculator.calculate_overall_aqi({"no2": 50.0, "so2": 40.0, "co": 1.5})
        assert res_no_particulate["is_strictly_valid"] is False

        # 3 pollutants WITH particulate
        res_valid = CPCBCalculator.calculate_overall_aqi({"pm10": 120.0, "no2": 50.0, "so2": 40.0})
        assert res_valid["is_strictly_valid"] is True

    def test_category_metadata_ranges(self):
        """Verifies category metadata mappings."""
        assert CPCBCalculator.get_category_metadata(35)["category"] == "Good"
        assert CPCBCalculator.get_category_metadata(85)["category"] == "Satisfactory"
        assert CPCBCalculator.get_category_metadata(150)["category"] == "Moderate"
        assert CPCBCalculator.get_category_metadata(250)["category"] == "Poor"
        assert CPCBCalculator.get_category_metadata(350)["category"] == "Very Poor"
        assert CPCBCalculator.get_category_metadata(450)["category"] == "Severe"
        assert CPCBCalculator.get_category_metadata(550)["category"] == "Severe+"
