"""
CPCB & OpenAQ Time-Series Dataset Generator
Generates realistic historical multi-year hourly observations for Delhi and Lucknow
matching published Central Pollution Control Board (CPCB) seasonal & meteorological patterns.
Author: Khushi Yadav (@khushiyadav35)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from config.settings import MONITORED_CITIES, RAW_DATA_DIR


def generate_city_timeseries(
    city_name: str,
    start_date: str = "2024-01-01 00:00:00",
    end_date: str = "2026-03-31 23:00:00",
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Simulates high-fidelity hourly ambient air pollution observations based on
    Indo-Gangetic Plain atmospheric dynamics:
    - Diurnal vehicular peaks (08:00 - 10:00 & 19:00 - 23:00)
    - Stubble burning & post-monsoon inversion spikes (Oct - Nov)
    - Winter dense fog & low mixing height trap (Dec - Jan)
    - Monsoon precipitation washout (Jul - Sep)
    - Summer dust storms & photochemical ozone peaks (Apr - Jun)
    """
    rng = np.random.RandomState(random_seed if city_name == "Delhi" else random_seed + 10)
    timestamps = pd.date_range(start=start_date, end=end_date, freq="h")
    n = len(timestamps)

    # City-specific baseline scaling (Delhi is slightly higher in traffic/density; Lucknow in valley dust)
    city_factor = 1.15 if city_name == "Delhi" else 0.95

    # 1. Seasonal Wave (Annual periodic cycle, peaking in mid-November/December)
    # Day of year: peak around day 315-345 (Nov 11 - Dec 15)
    day_of_year = timestamps.dayofyear.values
    seasonal_pm25 = np.exp(-((day_of_year - 325) ** 2) / (2 * 45 ** 2)) * 180.0
    # Secondary winter peak for January
    winter_jan = np.exp(-((day_of_year - 15) ** 2) / (2 * 25 ** 2)) * 140.0
    # Monsoon dip (Jul - Sep, days 180 - 260)
    monsoon_factor = 1.0 - 0.65 * np.exp(-((day_of_year - 225) ** 2) / (2 * 35 ** 2))

    # 2. Diurnal Cycle (Rush hour traffic and nocturnal boundary layer lowering)
    hours = timestamps.hour.values
    morning_peak = np.exp(-((hours - 8.5) ** 2) / (2 * 1.5 ** 2)) * 35.0
    evening_peak = np.exp(-((hours - 20.5) ** 2) / (2 * 2.5 ** 2)) * 55.0
    diurnal_pm25 = morning_peak + evening_peak

    # 3. Base concentrations + noise
    base_pm25 = (45.0 + seasonal_pm25 + winter_jan + diurnal_pm25) * monsoon_factor * city_factor
    pm25_noise = rng.normal(0, 15.0, size=n)
    pm25 = np.clip(base_pm25 + pm25_noise, a_min=8.0, a_max=650.0)

    # PM10 correlates strongly with PM2.5, elevated during summer dust storms (May-Jun)
    summer_dust = np.exp(-((day_of_year - 140) ** 2) / (2 * 30 ** 2)) * 120.0
    pm10 = np.clip(pm25 * 1.75 + summer_dust + rng.normal(0, 20.0, size=n), a_min=15.0, a_max=850.0)

    # NO2 (Vehicular & thermal power plant tracer, heavy evening rush)
    no2_base = (30.0 + evening_peak * 0.7 + morning_peak * 0.8) * city_factor
    no2 = np.clip(no2_base + rng.normal(0, 8.0, size=n), a_min=5.0, a_max=220.0)

    # SO2 (Industrial fuel combustion)
    so2_base = 18.0 * city_factor + rng.normal(0, 4.0, size=n)
    so2 = np.clip(so2_base, a_min=2.0, a_max=110.0)

    # CO (Carbon monoxide in mg/m3)
    co_base = (0.7 + (evening_peak + morning_peak) * 0.025) * city_factor
    co = np.clip(co_base + rng.normal(0, 0.25, size=n), a_min=0.1, a_max=12.5)

    # O3 (Photochemical smog, peaks during bright hot summer afternoons 13:00-16:00)
    afternoon_sun = np.exp(-((hours - 14.5) ** 2) / (2 * 2.0 ** 2)) * 45.0
    summer_sun = np.exp(-((day_of_year - 145) ** 2) / (2 * 45 ** 2)) * 30.0
    o3 = np.clip(25.0 + afternoon_sun + summer_sun + rng.normal(0, 7.0, size=n), a_min=5.0, a_max=240.0)

    # Meteorological Covariates
    # Temperature (Celsius)
    temp_seasonal = 26.0 - 12.0 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
    temp_diurnal = 5.0 * np.sin(2 * np.pi * (hours - 9) / 24)
    temperature = temp_seasonal + temp_diurnal + rng.normal(0, 1.5, size=n)

    # Relative Humidity (%)
    # Inverted from temperature, high in monsoon and foggy winter mornings
    rh_seasonal = 55.0 + 25.0 * np.exp(-((day_of_year - 230) ** 2) / (2 * 40 ** 2)) + 15.0 * np.exp(-((day_of_year - 10) ** 2) / (2 * 20 ** 2))
    rh_diurnal = -12.0 * np.sin(2 * np.pi * (hours - 9) / 24)
    relative_humidity = np.clip(rh_seasonal + rh_diurnal + rng.normal(0, 3.0, size=n), a_min=15.0, a_max=99.0)

    # Wind Speed (m/s) - low winds in winter (< 1.5 m/s) prevent dispersion
    wind_base = 2.8 - 1.2 * np.exp(-((day_of_year - 330) ** 2) / (2 * 35 ** 2))
    wind_speed = np.clip(wind_base + rng.normal(0, 0.6, size=n), a_min=0.3, a_max=8.5)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "city": city_name,
        "pm25": np.round(pm25, 2),
        "pm10": np.round(pm10, 2),
        "no2": np.round(no2, 2),
        "so2": np.round(so2, 2),
        "co": np.round(co, 2),
        "o3": np.round(o3, 2),
        "temperature": np.round(temperature, 1),
        "relative_humidity": np.round(relative_humidity, 1),
        "wind_speed": np.round(wind_speed, 2),
    })

    return df


def ensure_baseline_datasets() -> dict[str, Path]:
    """Generates and writes historical CSVs if not present."""
    paths = {}
    for city in ["Delhi", "Lucknow"]:
        out_file = RAW_DATA_DIR / f"{city.lower()}_hourly_aqi.csv"
        if not out_file.exists():
            print(f"Generating realistic CPCB time-series for {city}...")
            df = generate_city_timeseries(city)
            df.to_csv(out_file, index=False)
            print(f"Saved {len(df)} records to {out_file}")
        paths[city] = out_file
    return paths


if __name__ == "__main__":
    ensure_baseline_datasets()
