from __future__ import annotations
import asyncio
from datetime import datetime, timezone
import smtplib
from email.message import EmailMessage
from typing import List

from app.config import settings

class Watchdog:
    def __init__(self):
        now = datetime.now(timezone.utc)
        self.last_heartbeat_ts = now
        self.last_decision_ts = now
        self.error_times: List[datetime] = []
        self.total_errors = 0

    async def run(self):
        """Periodically check for silence or error bursts and issue alerts."""
        while True:
            await asyncio.sleep(30)
            now = datetime.now(timezone.utc)
            alerts = []
            hb_gap = (now - self.last_heartbeat_ts).total_seconds()
            dec_gap = (now - self.last_decision_ts).total_seconds()
            silence_threshold = settings.MAX_SILENCE_SECONDS
            if hb_gap > silence_threshold:
                alerts.append(f"heartbeat silence for {int(hb_gap)} (threshold {silence_threshold})")
            if dec_gap > silence_threshold:
                alerts.append(f"decision silence for {int(dec_gap)} (threshold {silence_threshold})")
            # error burst detection over a 60s window
            window = 60
            self.error_times = [t for t in self.error_times if (now - t).total_seconds() <= window]
            if len(self.error_times) >= settings.ERROR_BURST_THRESHOLD:
                alerts.append(f"error burst detected count={len(self.error_times)}")
            for reason in alerts:
                self.log_alert(reason)
                await self.send_alert(reason)

    def log_alert(self, reason: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        print(f"[ALERT] {ts} UTC: {reason}")

    async def send_alert(self, reason: str) -> None:
        """Send an email alert if SMTP settings are configured; otherwise, do nothing."""
        if (
            settings.ALERT_EMAIL
            and settings.SMTP_HOST
            and settings.SMTP_USER
            and settings.SMTP_PASS
            and settings.SMTP_PORT
        ):
            msg = EmailMessage()
            msg["Subject"] = "Mossy-4X ALERT"
            msg["From"] = settings.SMTP_USER
            msg["To"] = settings.ALERT_EMAIL
            body_ts = datetime.now(timezone.utc).isoformat()
            msg.set_content(f"{body_ts} UTC: {reason}")
            try:
                with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as smtp:
                    smtp.starttls()
                    smtp.login(settings.SMTP_USER, settings.SMTP_PASS)
                    smtp.send_message(msg)
            except Exception as e:
                print(f"[ALERT] email send failed: {e}")

    def record_error(self) -> None:
        """Record the timestamp of an error occurrence for burst detection."""
        self.error_times.append(datetime.now(timezone.utc))
        self.total_errors += 1

# Global instance of Watchdog
watchdog = Watchdog()
