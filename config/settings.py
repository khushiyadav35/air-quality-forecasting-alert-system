"""
System Configuration and Environmental Constants
Author: Khushi Yadav (@khushiyadav35)
Project: Air Quality Forecasting and Alert System
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure runtime directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Monitored Indian Megacities & Coordinates
MONITORED_CITIES = {
    "Delhi": {
        "state": "National Capital Region (NCR)",
        "latitude": 28.6139,
        "longitude": 77.2090,
        "elevation_m": 216,
        "stations": [
            {"id": "DL001", "name": "Anand Vihar", "type": "Industrial / Transport Hub", "lat": 28.6502, "lon": 77.3152},
            {"id": "DL002", "name": "Punjabi Bagh", "type": "Residential / Commercial", "lat": 28.6683, "lon": 77.1167},
            {"id": "DL003", "name": "RK Puram", "type": "Urban / Institutional", "lat": 28.5660, "lon": 77.1767},
            {"id": "DL004", "name": "IGI Airport (T3)", "type": "Aviation / Open Perimeter", "lat": 28.5562, "lon": 77.1000},
        ],
        "primary_concerns": ["Winter Smog (Oct-Jan)", "Agricultural Residue Stubble Burning", "Vehicular Congestion", "Inversion Trap"],
    },
    "Lucknow": {
        "state": "Uttar Pradesh",
        "latitude": 26.8467,
        "longitude": 80.9462,
        "elevation_m": 123,
        "stations": [
            {"id": "UP001", "name": "Talkatora Industrial Area", "type": "Industrial Hotspot", "lat": 26.8322, "lon": 80.8936},
            {"id": "UP002", "name": "Lalbagh", "type": "Commercial Core", "lat": 26.8489, "lon": 80.9394},
            {"id": "UP003", "name": "Gomti Nagar", "type": "Planned Residential", "lat": 26.8532, "lon": 80.9984},
            {"id": "UP004", "name": "Kendriya Vidyalaya (Aliganj)", "type": "School / Institutional Zone", "lat": 26.8920, "lon": 80.9410},
        ],
        "primary_concerns": ["Indo-Gangetic Basin Valley Trapping", "Biomass Fuel Emissions", "Construction Dust", "Low Boundary Layer"],
    },
}

# Forecasting Horizon Hyperparameters
LOOKBACK_WINDOW_HOURS = 72    # 3 days of historical hourly observations
FORECAST_HORIZON_24H = 24     # Short-term warning horizon
FORECAST_HORIZON_72H = 72     # Extended advisory horizon

# Alert System Thresholds (Indian CPCB NAQI Standard)
ALERT_THRESHOLDS = {
    "MODERATE": 101,      # Caution: sensitive individuals should limit outdoor exertion
    "POOR": 201,          # Warning: mask advised, healthy individuals experience discomfort
    "VERY_POOR": 301,     # High Alert: avoid outdoor cardio, N95 respirator mandatory
    "SEVERE": 401,        # Emergency: schools close advisory, air purifier recommended indoors
}

# Alert Dispatcher Defaults
DEFAULT_ALERT_COOLDOWN_HOURS = 4  # Prevent spamming duplicate alerts within 4h window
ALERT_SENDER_EMAIL = os.getenv("ALERT_SENDER_EMAIL", "aqi.alerts.delhi.lucknow@gmail.com")
ALERT_EMAIL_PASSWORD = os.getenv("ALERT_EMAIL_PASSWORD", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
