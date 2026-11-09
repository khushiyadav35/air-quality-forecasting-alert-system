"""
Unit Tests for Alert Manager, Cooldown Debounce, and Formatters
Author: Khushi Yadav (@khushiyadav35)
"""

import pytest
from datetime import datetime, timedelta
from src.alerts.alert_manager import AlertManager
from src.alerts.telegram_bot import TelegramAlertService
from src.alerts.email_alert import EmailAlertService


class TestAlertSystem:
    """Verifies notification dispatch rules, cooldown enforcement, and templates."""

    def test_alert_threshold_trigger(self, tmp_path):
        """Verifies an alert triggers when AQI crosses the threshold."""
        hist_file = tmp_path / "test_history.json"
        mgr = AlertManager(history_file=hist_file, cooldown_hours=4)

        should_fire, _ = mgr.should_trigger_alert(
            city="Delhi",
            current_aqi=245,
            category="Poor",
            threshold=201,
        )
        assert should_fire is True

    def test_alert_suppressed_below_threshold(self, tmp_path):
        """Verifies no alert when AQI is below threshold."""
        hist_file = tmp_path / "test_history.json"
        mgr = AlertManager(history_file=hist_file, cooldown_hours=4)

        should_fire, _ = mgr.should_trigger_alert(
            city="Lucknow",
            current_aqi=140,
            category="Moderate",
            forecast_peak_aqi=160,
            threshold=201,
        )
        assert should_fire is False

    def test_cooldown_suppression(self, tmp_path):
        """Verifies repeated alerts inside cooldown window are suppressed."""
        hist_file = tmp_path / "test_history.json"
        mgr = AlertManager(history_file=hist_file, cooldown_hours=4)

        # Simulate alert sent 1 hour ago
        mgr.history = [
            {
                "timestamp": (datetime.now() - timedelta(hours=1)).isoformat(),
                "city": "Delhi",
                "current_aqi": 220,
                "category": "Poor",
            }
        ]

        # Minor increase (220 -> 230) inside 4h window should be suppressed
        should_fire, reason = mgr.should_trigger_alert(
            city="Delhi",
            current_aqi=230,
            category="Poor",
            threshold=201,
        )
        assert should_fire is False
        assert "cooldown" in reason.lower()

    def test_delta_spike_bypasses_cooldown(self, tmp_path):
        """Verifies drastic sudden deterioration (>= +50 points) overrides cooldown."""
        hist_file = tmp_path / "test_history.json"
        mgr = AlertManager(history_file=hist_file, cooldown_hours=4)

        mgr.history = [
            {
                "timestamp": (datetime.now() - timedelta(hours=1)).isoformat(),
                "city": "Delhi",
                "current_aqi": 210,
                "category": "Poor",
            }
        ]

        # Spikes to 280 (+70 points)
        should_fire, reason = mgr.should_trigger_alert(
            city="Delhi",
            current_aqi=280,
            category="Poor",
            threshold=201,
        )
        assert should_fire is True
        assert "delta" in reason.lower()

    def test_telegram_message_contains_essential_elements(self):
        """Checks formatted Telegram message content."""
        svc = TelegramAlertService()
        msg = svc.format_alert_message({
            "city": "Delhi",
            "current_aqi": 340,
            "category": "Very Poor",
            "emoji": "🟣",
            "prominent_pollutant": "PM2.5",
            "forecast_peak_aqi": 390,
            "peak_time": "Tomorrow Morning",
            "advisory": "Stay indoors.",
            "sub_indices": {"pm25": 340, "pm10": 260},
        })
        assert "DELHI" in msg
        assert "340" in msg
        assert "VERY POOR" in msg
        assert "PM2.5" in msg
        assert "N95" in msg

    def test_email_html_structure(self):
        """Checks formatted HTML email generation."""
        svc = EmailAlertService()
        html = svc.build_html_template({
            "city": "Lucknow",
            "current_aqi": 280,
            "category": "Poor",
            "color": "#F97316",
            "prominent_pollutant": "PM2.5",
            "forecast_peak_aqi": 310,
            "sub_indices": {"pm25": 280},
        })
        assert "<!DOCTYPE html>" in html
        assert "Lucknow" in html
        assert "280 AQI" in html
