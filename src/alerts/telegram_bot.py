"""
Telegram Atmospheric Alert Dispatcher
Author: Khushi Yadav (@khushiyadav35)
"""

import logging
from typing import Dict, Any, Optional
import requests

from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)


class TelegramAlertService:
    """
    Dispatches real-time air pollution advisories and 24h-72h forecast warnings
    via the official Telegram Bot API (HTML mode).
    Includes safe test/dry-run mode if live credentials are not set.
    """

    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        self.bot_token = bot_token or TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or TELEGRAM_CHAT_ID
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage" if self.bot_token else None

    def format_alert_message(self, alert_payload: Dict[str, Any]) -> str:
        """Constructs an alert message with HTML formatting and emojis."""
        city = alert_payload.get("city", "Unknown")
        current_aqi = alert_payload.get("current_aqi", 0)
        category = alert_payload.get("category", "Unknown")
        emoji = alert_payload.get("emoji", "⚠️")
        prominent = alert_payload.get("prominent_pollutant", "PM2.5")
        forecast_peak = alert_payload.get("forecast_peak_aqi", current_aqi)
        peak_time = alert_payload.get("peak_time", "Next 24 Hours")
        advisory = alert_payload.get("advisory", "Limit outdoor activities.")
        sub_indices = alert_payload.get("sub_indices", {})

        sub_str = " | ".join([f"{k.upper()}: {v}" for k, v in sub_indices.items()])

        message = (
            f"<b>{emoji} AIR QUALITY ALERT | {city.upper()} {emoji}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>CPCB NAQI Status:</b> {current_aqi} (<b>{category.upper()}</b>)\n"
            f"<b>Primary Hazard:</b> {prominent}\n"
            f"<b>Sub-Indices:</b> <code>{sub_str}</code>\n\n"
            f"<b>📈 24h–72h Forecast Outlook:</b>\n"
            f"• Projected Peak: <b>{forecast_peak} AQI</b>\n"
            f"• Peak Period: <i>{peak_time}</i>\n\n"
            f"<b>🩺 Public Health Advisory:</b>\n"
            f"<blockquote>{advisory}</blockquote>\n\n"
            f"<b>🛡️ Recommended Precautionary Actions:</b>\n"
            f"• Wear certified N95/FFP2 respirators when stepping outdoors.\n"
            f"• Seal windows during early morning and late evening inversion hours.\n"
            f"• Sensitive individuals (asthma, cardiac, elderly) must avoid outdoor workouts.\n"
            f"• Operate indoor HEPA air filtration if available.\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"<i>Broadcasted by Air Quality Forecasting System (@khushiyadav35)</i>"
        )
        return message

    def send_alert(self, alert_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatches alert to Telegram channel/group or records dry-run output."""
        formatted_text = self.format_alert_message(alert_payload)

        if not self.bot_token or not self.chat_id or self.bot_token.startswith("your_"):
            logger.info(f"[TELEGRAM DRY-RUN MODE] Would send alert to {self.chat_id or 'Demo-Channel'}:\n{formatted_text}")
            return {
                "status": "dry_run_success",
                "channel": "Telegram",
                "delivered": True,
                "message": formatted_text,
                "note": "Configured in local simulation mode. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env for live dispatch.",
            }

        try:
            payload = {
                "chat_id": self.chat_id,
                "text": formatted_text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }
            resp = requests.post(self.api_url, json=payload, timeout=8)
            if resp.status_code == 200:
                logger.info(f"Telegram alert successfully transmitted to {self.chat_id}")
                return {"status": "success", "channel": "Telegram", "delivered": True, "response": resp.json()}
            else:
                logger.error(f"Telegram API error {resp.status_code}: {resp.text}")
                return {"status": "failed", "channel": "Telegram", "delivered": False, "error": resp.text}
        except Exception as exc:
            logger.error(f"Telegram transmission failure: {exc}")
            return {"status": "error", "channel": "Telegram", "delivered": False, "error": str(exc)}
