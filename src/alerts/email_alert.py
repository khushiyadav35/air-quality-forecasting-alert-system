"""
SMTP Email Alert Dispatcher with Responsive HTML Template
Author: Khushi Yadav (@khushiyadav35)
"""

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, Any, List, Optional, Union

from config.settings import ALERT_SENDER_EMAIL, ALERT_EMAIL_PASSWORD

logger = logging.getLogger(__name__)


class EmailAlertService:
    """Dispatches styled HTML email alerts via SMTP (Gmail, Outlook, AWS SES)."""

    def __init__(
        self,
        sender_email: Optional[str] = None,
        email_password: Optional[str] = None,
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
    ):
        self.sender_email = sender_email or ALERT_SENDER_EMAIL
        self.email_password = email_password or ALERT_EMAIL_PASSWORD
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port

    def build_html_template(self, alert_payload: Dict[str, Any]) -> str:
        """Constructs an HTML email layout."""
        city = alert_payload.get("city", "Delhi")
        current_aqi = alert_payload.get("current_aqi", 0)
        category = alert_payload.get("category", "Severe")
        color = alert_payload.get("color", "#EF4444")
        prominent = alert_payload.get("prominent_pollutant", "PM2.5")
        forecast_peak = alert_payload.get("forecast_peak_aqi", current_aqi)
        peak_time = alert_payload.get("peak_time", "Upcoming 24 Hours")
        advisory = alert_payload.get("advisory", "High pollution advisory in effect.")
        sub_indices = alert_payload.get("sub_indices", {})

        sub_rows = "".join(
            [
                f"<tr style='border-bottom: 1px solid #e5e7eb;'><td style='padding: 8px 12px; font-weight: 600;'>{p.upper()}</td><td style='padding: 8px 12px; text-align: right;'>{v}</td></tr>"
                for p, v in sub_indices.items()
            ]
        )

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f3f4f6; margin: 0; padding: 20px; }}
                .container {{ max-width: 620px; margin: 0 auto; background: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }}
                .header {{ background-color: #0f172a; color: #ffffff; padding: 24px; text-align: center; }}
                .header h1 {{ margin: 0; font-size: 20px; text-transform: uppercase; letter-spacing: 1px; }}
                .status-badge {{ display: inline-block; padding: 12px 24px; margin: 20px 0; border-radius: 9999px; background-color: {color}; color: #ffffff; font-size: 22px; font-weight: 700; text-shadow: 0 1px 2px rgba(0,0,0,0.2); }}
                .content {{ padding: 24px 30px; color: #374151; }}
                .stat-grid {{ display: flex; justify-content: space-between; margin-bottom: 24px; background: #f8fafc; border-radius: 8px; padding: 16px; border: 1px solid #e2e8f0; }}
                .stat-item {{ text-align: center; width: 33%; }}
                .stat-item .val {{ font-size: 20px; font-weight: 700; color: #1e293b; }}
                .stat-item .label {{ font-size: 11px; text-transform: uppercase; color: #64748b; margin-top: 4px; }}
                .advisory-box {{ background-color: #fef2f2; border-left: 4px solid #ef4444; padding: 14px 18px; border-radius: 4px; margin: 20px 0; color: #991b1b; }}
                .actions {{ background: #eff6ff; border-radius: 8px; padding: 16px 20px; margin-top: 20px; border: 1px solid #bfdbfe; }}
                .actions ul {{ margin: 0; padding-left: 20px; color: #1e40af; line-height: 1.6; }}
                .footer {{ background: #f8fafc; text-align: center; padding: 16px; font-size: 12px; color: #94a3b8; border-top: 1px solid #e2e8f0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Air Quality Alert System</h1>
                    <p style="margin: 6px 0 0; color: #94a3b8; font-size: 13px;">Official CPCB NAQI Monitoring Protocol | {city.upper()}</p>
                </div>
                <div class="content">
                    <div style="text-align: center;">
                        <span class="status-badge">{current_aqi} AQI &bull; {category.upper()}</span>
                    </div>
                    <div class="stat-grid">
                        <div class="stat-item">
                            <div class="val">{city}</div>
                            <div class="label">Location</div>
                        </div>
                        <div class="stat-item">
                            <div class="val">{prominent}</div>
                            <div class="label">Main Hazard</div>
                        </div>
                        <div class="stat-item">
                            <div class="val">{forecast_peak} AQI</div>
                            <div class="label">24h–72h Peak</div>
                        </div>
                    </div>

                    <div class="advisory-box">
                        <strong>Public Health Advisory:</strong><br>
                        {advisory}
                    </div>

                    <h3 style="font-size: 15px; color: #1e293b; margin-top: 20px;">Sub-Index Pollutant Distribution</h3>
                    <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                        <tr style="background: #f1f5f9;"><th style="padding: 8px 12px; text-align: left;">Pollutant</th><th style="padding: 8px 12px; text-align: right;">Sub-Index (CPCB)</th></tr>
                        {sub_rows}
                    </table>

                    <div class="actions">
                        <strong style="color: #1e3a8a;">Mandatory Protective Guidelines:</strong>
                        <ul style="margin-top: 8px;">
                            <li>Wear N95/FFP2 particle respirators during all outdoor movement.</li>
                            <li>Avoid strenuous early morning exercise (peak thermal inversion).</li>
                            <li>Close doors and windows, and activate HEPA filters indoors.</li>
                            <li>Children, pregnant women, and pulmonary patients should stay indoors.</li>
                        </ul>
                    </div>
                </div>
                <div class="footer">
                    Automated atmospheric forecast dispatch &bull; B.Tech Capstone Project &bull; Khushi Yadav (@khushiyadav35)
                </div>
            </div>
        </body>
        </html>
        """
        return html

    def send_email(
        self,
        alert_payload: Dict[str, Any],
        recipient_emails: Union[str, List[str]],
    ) -> Dict[str, Any]:
        """Sends formatted HTML email or logs dry-run output."""
        recipients = [r.strip() for r in (recipient_emails if isinstance(recipient_emails, list) else recipient_emails.split(",")) if r.strip()]
        if not recipients:
            recipients = ["demo-user@institution.ac.in"]

        html_content = self.build_html_template(alert_payload)
        city = alert_payload.get("city", "Delhi")
        current_aqi = alert_payload.get("current_aqi", 0)
        category = alert_payload.get("category", "Warning")
        subject = f"⚠️ [AQI ALERT] {city.upper()} Air Quality Reached {current_aqi} ({category.upper()})"

        if not self.sender_email or not self.email_password or self.email_password.startswith("your_"):
            logger.info(f"[EMAIL DRY-RUN MODE] Would send email alert to {recipients}:\nSubject: {subject}")
            return {
                "status": "dry_run_success",
                "channel": "Email",
                "delivered": True,
                "recipients": recipients,
                "subject": subject,
                "note": "Configured in local simulation mode. Set ALERT_SENDER_EMAIL and ALERT_EMAIL_PASSWORD for SMTP dispatch.",
            }

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.sender_email
            msg["To"] = ", ".join(recipients)
            msg.attach(MIMEText(html_content, "html"))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.email_password)
                server.sendmail(self.sender_email, recipients, msg.as_string())

            logger.info(f"Email successfully delivered to {recipients}")
            return {"status": "success", "channel": "Email", "delivered": True, "recipients": recipients}
        except Exception as exc:
            logger.error(f"SMTP dispatch failure: {exc}")
            return {"status": "error", "channel": "Email", "delivered": False, "error": str(exc)}
