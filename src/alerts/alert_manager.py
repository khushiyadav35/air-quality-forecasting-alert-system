"""
Alert Manager with Cooldown Tracker and Multi-Channel Dispatch Engine
Author: Khushi Yadav (@khushiyadav35)
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from config.settings import ALERT_THRESHOLDS, DEFAULT_ALERT_COOLDOWN_HOURS, DATA_DIR
from src.alerts.telegram_bot import TelegramAlertService
from src.alerts.email_alert import EmailAlertService

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Coordinates multi-channel alert delivery (Telegram + Email).
    Features:
    - Debounce and Cooldown engine to prevent spamming notification channels
    - Delta Trigger: Fires immediately if air quality spikes drastically (> 50 points)
    - Predictive Advisory: Warns if upcoming 24h-72h forecast crosses emergency thresholds
    - Historical dispatch audit trail logging
    """

    def __init__(
        self,
        cooldown_hours: int = DEFAULT_ALERT_COOLDOWN_HOURS,
        history_file: Optional[Path] = None,
    ):
        self.cooldown_hours = cooldown_hours
        self.history_file = history_file or (DATA_DIR / "processed" / "alert_history.json")
        self.telegram_service = TelegramAlertService()
        self.email_service = EmailAlertService()
        self.history = self._load_history()

    def _load_history(self) -> List[Dict[str, Any]]:
        """Loads historical alert log."""
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_history(self) -> None:
        """Persists alert logs to disk."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.history[-100:], f, indent=2)  # Keep last 100 entries
        except Exception as exc:
            logger.warning(f"Could not persist alert history: {exc}")

    def should_trigger_alert(
        self,
        city: str,
        current_aqi: int,
        category: str,
        forecast_peak_aqi: Optional[int] = None,
        threshold: int = ALERT_THRESHOLDS["POOR"],
    ) -> Tuple[bool, str]:
        """
        Evaluates whether an alert should be dispatched based on thresholds and cooldowns.
        Returns: (should_alert, reason)
        """
        # Condition 1: Check absolute threshold
        is_above_current = current_aqi >= threshold
        is_above_forecast = (forecast_peak_aqi is not None) and (forecast_peak_aqi >= threshold)

        if not is_above_current and not is_above_forecast:
            return False, f"AQI ({current_aqi}) and forecast ({forecast_peak_aqi}) are below alert threshold ({threshold})."

        # Check previous alert for this city
        recent_alerts = [a for a in self.history if a.get("city", "").lower() == city.lower()]
        if not recent_alerts:
            return True, f"First alert triggered for {city}: AQI is {current_aqi} ({category})."

        last_alert = recent_alerts[-1]
        last_time = datetime.fromisoformat(last_alert["timestamp"])
        last_aqi = last_alert["current_aqi"]
        last_category = last_alert.get("category", "")
        time_elapsed = datetime.now() - last_time

        # Condition 2: Delta Trigger (Drastic deterioration)
        if (current_aqi - last_aqi) >= 50:
            return True, f"Delta Trigger: AQI deteriorated sharply from {last_aqi} to {current_aqi} (+{current_aqi - last_aqi} points)."

        # Condition 3: Escalation to worse category
        if category in ["Very Poor", "Severe", "Severe+"] and last_category in ["Moderate", "Poor"]:
            return True, f"Category Escalation: Status escalated from {last_category} to {category}."

        # Condition 4: Cooldown period elapsed
        if time_elapsed >= timedelta(hours=self.cooldown_hours):
            return True, f"Cooldown elapsed: Last alert sent {time_elapsed.total_seconds() / 3600.0:.1f} hours ago."

        hours_left = self.cooldown_hours - (time_elapsed.total_seconds() / 3600.0)
        return False, f"Alert suppressed by cooldown: {hours_left:.1f} hours remaining in cooldown window."

    def dispatch_alert(
        self,
        alert_payload: Dict[str, Any],
        send_telegram: bool = True,
        send_email: bool = True,
        recipient_emails: Optional[str] = None,
        force: bool = False,
        threshold: int = ALERT_THRESHOLDS["POOR"],
    ) -> Dict[str, Any]:
        """Evaluates triggers and coordinates delivery across channels."""
        city = alert_payload.get("city", "Delhi")
        current_aqi = alert_payload.get("current_aqi", 0)
        category = alert_payload.get("category", "Moderate")
        forecast_peak = alert_payload.get("forecast_peak_aqi")

        should_fire, reason = self.should_trigger_alert(
            city=city,
            current_aqi=current_aqi,
            category=category,
            forecast_peak_aqi=forecast_peak,
            threshold=threshold,
        )

        if not should_fire and not force:
            logger.info(f"Alert not dispatched for {city}: {reason}")
            return {
                "triggered": False,
                "reason": reason,
                "city": city,
                "current_aqi": current_aqi,
                "channels": {},
            }

        logger.info(f"Dispatching alerts for {city} (Reason: {reason})...")
        dispatch_results = {}

        if send_telegram:
            dispatch_results["telegram"] = self.telegram_service.send_alert(alert_payload)

        if send_email:
            recipients = recipient_emails or "public-alert@delhi-lucknow-aqi.gov.in"
            dispatch_results["email"] = self.email_service.send_email(alert_payload, recipients)

        # Log into history
        record = {
            "timestamp": datetime.now().isoformat(),
            "city": city,
            "current_aqi": current_aqi,
            "category": category,
            "prominent_pollutant": alert_payload.get("prominent_pollutant", "PM2.5"),
            "forecast_peak_aqi": forecast_peak,
            "reason": reason,
            "dispatch_results": dispatch_results,
        }
        self.history.append(record)
        self._save_history()

        return {
            "triggered": True,
            "reason": reason,
            "city": city,
            "current_aqi": current_aqi,
            "channels": dispatch_results,
            "record": record,
        }


if __name__ == "__main__":
    mgr = AlertManager()
    sample_payload = {
        "city": "Delhi",
        "current_aqi": 342,
        "category": "Very Poor",
        "color": "#EF4444",
        "emoji": "🟣",
        "prominent_pollutant": "PM2.5",
        "forecast_peak_aqi": 388,
        "peak_time": "Tomorrow 07:00 AM (Winter Inversion Peak)",
        "advisory": "Respiratory illness expected on prolonged exposure. Avoid all morning outdoor exertion.",
        "sub_indices": {"pm25": 342, "pm10": 260, "no2": 115, "so2": 45},
    }
    res = mgr.dispatch_alert(sample_payload, force=True)
    print("Alert Dispatch Result:\n", json.dumps(res, indent=2))
