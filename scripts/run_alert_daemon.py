"""
Air Quality Periodic Monitoring and Alert Daemon
Pulls latest data for Delhi and Lucknow, computes CPCB NAQI,
generates 24h-72h predictions, and fires alerts if thresholds are breached.
Author: Khushi Yadav (@khushiyadav35)
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import argparse
import json
import logging
import time
from datetime import datetime

from config.settings import MONITORED_CITIES, ALERT_THRESHOLDS
from src.data.ingestion import DataIngestionPipeline
from src.preprocessing.cpcb_calculator import CPCBCalculator
from src.models.prophet_forecaster import AdditiveProphetForecaster
from src.alerts.alert_manager import AlertManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_cycle(threshold: int = ALERT_THRESHOLDS["POOR"], dry_run: bool = False):
    """Executes a single monitoring cycle across Delhi and Lucknow."""
    logger.info(f"=== Starting Air Quality Monitoring Sweep (Threshold: {threshold} AQI) ===")
    alert_mgr = AlertManager()

    for city in MONITORED_CITIES.keys():
        logger.info(f"Processing real-time feed for {city}...")
        pipeline = DataIngestionPipeline(city)
        df = pipeline.fetch_live_hourly(past_days=3, forecast_days=3)

        df_naqi = CPCBCalculator.compute_dataframe_naqi(df)
        latest_row = df_naqi.iloc[-1]

        current_aqi = int(latest_row["cpcb_aqi"])
        category = latest_row["aqi_category"]
        prominent = latest_row["prominent_pollutant"]

        # Calculate forecast peak over upcoming 24 hours
        prophet_model_path = BASE_DIR / "models" / f"prophet_{city.lower()}.pkl"
        if prophet_model_path.exists():
            forecaster = AdditiveProphetForecaster.load(prophet_model_path)
            _, f_preds, _, _ = forecaster.predict_future(
                last_timestamp=pd.Timestamp(latest_row["timestamp"]),
                forecast_hours=24,
            )
            forecast_peak_aqi = int(np.max(f_preds))
            peak_hour_idx = int(np.argmax(f_preds))
            peak_time = f"+{peak_hour_idx + 1}h ahead (Projected {forecast_peak_aqi} AQI)"
        else:
            forecast_peak_aqi = current_aqi
            peak_time = "Next 24h"

        category_meta = CPCBCalculator.get_category_metadata(current_aqi)

        alert_payload = {
            "city": city,
            "current_aqi": current_aqi,
            "category": category,
            "color": category_meta["color"],
            "emoji": category_meta["emoji"],
            "prominent_pollutant": prominent,
            "forecast_peak_aqi": forecast_peak_aqi,
            "peak_time": peak_time,
            "advisory": category_meta["advisory"],
            "sub_indices": {
                p: int(latest_row[p])
                for p in ["pm25", "pm10", "no2", "so2"]
                if p in latest_row and pd.notna(latest_row[p])
            },
        }

        result = alert_mgr.dispatch_alert(
            alert_payload=alert_payload,
            threshold=threshold,
            force=dry_run,
        )
        logger.info(f"Result for {city}: Triggered={result['triggered']} | Reason={result['reason']}")


if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    parser = argparse.ArgumentParser(description="Air Quality Alert Daemon")
    parser.add_argument("--threshold", type=int, default=201, help="AQI alert threshold (default 201 for Poor)")
    parser.add_argument("--dry-run", action="store_true", help="Force alert dispatch without threshold block")
    args = parser.parse_args()

    run_cycle(threshold=args.threshold, dry_run=args.dry_run)
