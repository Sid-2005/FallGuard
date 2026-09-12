"""
FallGuard AI - Alert Service
Handles sound, browser push, email, SMS alerts.
"""

import logging
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class AlertService:
    def __init__(self, app_config: dict):
        self.config = app_config

    def dispatch_fall_alert(self, event_data: dict, contacts: list, settings: dict):
        """Fire all enabled alert channels in background threads."""
        if settings.get('enable_email') and self.config.get('MAIL_USERNAME'):
            threading.Thread(
                target=self._send_email_alerts,
                args=(event_data, contacts),
                daemon=True
            ).start()

        if settings.get('enable_sms') and self.config.get('TWILIO_ACCOUNT_SID'):
            threading.Thread(
                target=self._send_sms_alerts,
                args=(event_data, contacts),
                daemon=True
            ).start()

    def _send_email_alerts(self, event_data: dict, contacts: list):
        try:
            server = smtplib.SMTP(self.config['MAIL_SERVER'], self.config['MAIL_PORT'])
            server.starttls()
            server.login(self.config['MAIL_USERNAME'], self.config['MAIL_PASSWORD'])

            for contact in contacts:
                if not contact.get('notify_email') or not contact.get('email'):
                    continue
                msg = MIMEMultipart('alternative')
                msg['Subject'] = '🚨 FallGuard Alert: Fall Detected'
                msg['From'] = self.config['MAIL_DEFAULT_SENDER']
                msg['To'] = contact['email']
                html = self._build_email_html(event_data, contact)
                msg.attach(MIMEText(html, 'html'))
                server.sendmail(self.config['MAIL_DEFAULT_SENDER'], contact['email'], msg.as_string())
                logger.info(f"Email alert sent to {contact['email']}")
            server.quit()
        except Exception as e:
            logger.error(f"Email alert failed: {e}")

    def _send_sms_alerts(self, event_data: dict, contacts: list):
        try:
            from twilio.rest import Client
            client = Client(self.config['TWILIO_ACCOUNT_SID'], self.config['TWILIO_AUTH_TOKEN'])
            ts = event_data.get('timestamp', datetime.utcnow().isoformat())
            msg_body = (
                f"🚨 FALL DETECTED by FallGuard AI\n"
                f"Time: {ts}\n"
                f"Confidence: {event_data.get('confidence', 0):.0%}\n"
                f"Event ID: {event_data.get('event_id', 'N/A')}\n"
                f"Please check on the patient immediately."
            )
            for contact in contacts:
                if not contact.get('notify_sms') or not contact.get('phone'):
                    continue
                client.messages.create(
                    body=msg_body,
                    from_=self.config['TWILIO_FROM_NUMBER'],
                    to=contact['phone']
                )
                logger.info(f"SMS alert sent to {contact['phone']}")
        except Exception as e:
            logger.error(f"SMS alert failed: {e}")

    def _build_email_html(self, event_data: dict, contact: dict) -> str:
        conf = event_data.get('confidence', 0)
        ts   = event_data.get('timestamp', datetime.utcnow().isoformat())
        eid  = event_data.get('event_id', 'N/A')
        return f"""
        <html><body style="font-family:Arial,sans-serif;background:#0d1117;color:#e6edf3;padding:20px">
        <div style="max-width:600px;margin:auto;background:#161b22;border-radius:12px;padding:30px;border:1px solid #30363d">
          <h2 style="color:#ff4444;margin-top:0">🚨 Fall Detected — FallGuard AI</h2>
          <p>Dear {contact.get('name','Caregiver')},</p>
          <p>A fall has been detected. Please check on the monitored individual immediately.</p>
          <table style="width:100%;border-collapse:collapse;margin:20px 0">
            <tr><td style="padding:8px;color:#8b949e">Event ID</td><td style="padding:8px;font-weight:bold">{eid}</td></tr>
            <tr><td style="padding:8px;color:#8b949e">Timestamp</td><td style="padding:8px">{ts}</td></tr>
            <tr><td style="padding:8px;color:#8b949e">Confidence</td><td style="padding:8px;color:#ff4444;font-weight:bold">{conf:.0%}</td></tr>
          </table>
          <p style="color:#8b949e;font-size:12px">FallGuard AI — Healthcare Monitoring Platform</p>
        </div></body></html>
        """
