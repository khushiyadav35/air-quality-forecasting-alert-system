"""
Official Central Pollution Control Board (CPCB) NAQI Calculation Engine
Implements the exact Indian National Air Quality Index piecewise linear interpolation formula
Reference: Ministry of Environment, Forest and Climate Change (MoEFCC) / CPCB India (2014)
Author: Khushi Yadav (@khushiyadav35)
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

from config.naqi_breakpoints import CPCB_BREAKPOINTS, NAQI_CATEGORIES


class CPCBCalculator:
    """
    Computes sub-indices and aggregate National Air Quality Index (NAQI)
    according to Indian CPCB standards.
    """

    @staticmethod
    def calculate_sub_index(pollutant: str, concentration: float) -> Optional[float]:
        """
        Computes sub-index Ip for a given pollutant and concentration Cp using linear interpolation:
        Ip = I_lo + ((I_hi - I_lo) / (B_hi - B_lo)) * (Cp - B_lo)
        """
        pollutant = pollutant.lower().replace(".", "").replace("_", "")
        if pollutant not in CPCB_BREAKPOINTS:
            return None

        if concentration is None or np.isnan(concentration) or concentration < 0:
            return None

        breakpoints = CPCB_BREAKPOINTS[pollutant]

        # Check lowest boundary
        if concentration <= breakpoints[0][0]:
            return float(breakpoints[0][2])

        # Standard piecewise search
        for b_lo, b_hi, i_lo, i_hi in breakpoints:
            if b_lo <= concentration <= b_hi:
                sub_index = i_lo + ((i_hi - i_lo) / (b_hi - b_lo)) * (concentration - b_lo)
                return round(float(sub_index), 2)

        # Upper extreme (exceeding severe breakpoint)
        last_b_lo, last_b_hi, last_i_lo, last_i_hi = breakpoints[-1]
        if concentration > last_b_hi:
            # Linear extrapolation cap
            excess = concentration - last_b_hi
            sub_index = last_i_hi + (excess / (last_b_hi - last_b_lo)) * 50.0
            return round(min(sub_index, 999.0), 2)

        return None

    @classmethod
    def calculate_overall_aqi(cls, pollutant_concentrations: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculates aggregate Indian NAQI from available pollutant concentrations.
        CPCB Criteria:
        1. Minimum 3 pollutants must be monitored.
        2. At least one particulate (PM2.5 or PM10) must be present.
        3. Overall AQI = max(sub-indices).
        4. Prominent pollutant = pollutant yielding the maximum sub-index.
        """
        sub_indices = {}
        for pollutant, conc in pollutant_concentrations.items():
            if conc is not None and not np.isnan(conc):
                val = cls.calculate_sub_index(pollutant, conc)
                if val is not None:
                    clean_name = pollutant.lower().replace(".", "").replace("_", "")
                    sub_indices[clean_name] = val

        has_pm25 = "pm25" in sub_indices
        has_pm10 = "pm10" in sub_indices
        valid_particulate = has_pm25 or has_pm10
        total_valid_pollutants = len(sub_indices)

        # Strict CPCB validation check
        is_strictly_valid = (total_valid_pollutants >= 3) and valid_particulate

        if not sub_indices:
            return {
                "aqi": None,
                "category": "Unknown",
                "color": "#9CA3AF",
                "hex_badge": "#6B7280",
                "emoji": "⚪",
                "prominent_pollutant": None,
                "sub_indices": {},
                "health_impact": "Insufficient observational data to compute NAQI.",
                "advisory": "Ensure continuous air quality monitoring instruments are operational.",
                "is_strictly_valid": False,
            }

        # Prominent pollutant and overall AQI
        prominent_pollutant, max_sub_index = max(sub_indices.items(), key=lambda item: item[1])
        overall_aqi = round(max_sub_index)

        category_meta = cls.get_category_metadata(overall_aqi)

        return {
            "aqi": overall_aqi,
            "category": category_meta["category"],
            "color": category_meta["color"],
            "hex_badge": category_meta["hex_badge"],
            "emoji": category_meta["emoji"],
            "prominent_pollutant": prominent_pollutant.upper(),
            "sub_indices": sub_indices,
            "health_impact": category_meta["health_impact"],
            "advisory": category_meta["advisory"],
            "is_strictly_valid": is_strictly_valid,
        }

    @staticmethod
    def get_category_metadata(aqi_value: Union[int, float]) -> Dict[str, Any]:
        """Retrieves CPCB category classification and advisories."""
        if aqi_value is None or np.isnan(aqi_value):
            return {
                "category": "Unknown",
                "color": "#9CA3AF",
                "hex_badge": "#6B7280",
                "emoji": "⚪",
                "health_impact": "No data",
                "advisory": "No data",
            }

        val = round(aqi_value)
        for cat in NAQI_CATEGORIES:
            if cat["aqi_min"] <= val <= cat["aqi_max"]:
                return cat

        # Severe+ extreme
        if val > 500:
            return {
                "category": "Severe+",
                "color": "#4A0000",
                "hex_badge": "#450A0A",
                "emoji": "☠️",
                "health_impact": "Emergency public health hazard. Dangerous for all age demographics.",
                "advisory": "Impose total outdoor work shutdown; shelter in place with air purification.",
            }

        return NAQI_CATEGORIES[0]

    @classmethod
    def compute_dataframe_naqi(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized/batched computation for entire time-series DataFrame."""
        out_df = df.copy()

        aqi_list = []
        cat_list = []
        prominent_list = []

        pollutant_cols = [c for c in ["pm25", "pm10", "no2", "so2", "co", "o3", "nh3"] if c in out_df.columns]

        for _, row in out_df.iterrows():
            conc_dict = {col: row[col] for col in pollutant_cols if pd.notna(row[col])}
            res = cls.calculate_overall_aqi(conc_dict)
            aqi_list.append(res["aqi"])
            cat_list.append(res["category"])
            prominent_list.append(res["prominent_pollutant"])

        out_df["cpcb_aqi"] = aqi_list
        out_df["aqi_category"] = cat_list
        out_df["prominent_pollutant"] = prominent_list
        return out_df
