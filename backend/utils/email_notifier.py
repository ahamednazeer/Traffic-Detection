"""
Email notification utilities for accident alerts.
"""
from __future__ import annotations

import re
import smtplib
import ssl
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, Optional

from config.settings import (
    ALERT_EMAIL_ENABLED,
    SMTP_FROM_EMAIL,
    SMTP_FROM_NAME,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_USERNAME,
    SMTP_USE_TLS,
)

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def is_valid_email(recipient: str) -> bool:
    """Minimal email format validation for UI-provided addresses."""
    return bool(recipient and EMAIL_REGEX.match(recipient.strip()))


def is_email_configured() -> bool:
    """Check required SMTP configuration values."""
    return (
        ALERT_EMAIL_ENABLED
        and bool(SMTP_HOST)
        and bool(SMTP_PORT)
        and bool(SMTP_FROM_EMAIL)
        and bool(SMTP_USERNAME)
        and bool(SMTP_PASSWORD)
    )


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "N/A"
    seconds = max(0.0, float(seconds))
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{mins:02d}:{secs:02d}.{millis:03d}"


def _build_html(payload: Dict) -> str:
    detection_time = _format_duration(payload.get("best_timestamp"))
    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    score = float(payload.get("score", 0.0)) * 100.0
    threshold = float(payload.get("threshold", 0.0)) * 100.0
    class_name = payload.get("class_name") or "accident"
    model_name = payload.get("model_name") or "accident_yolo11x"
    file_name = payload.get("file_name") or "uploaded_video.mp4"
    job_id = payload.get("job_id") or "N/A"

    return f"""
<!DOCTYPE html>
<html>
  <head>
    <meta charset="UTF-8" />
    <title>Accident Alert</title>
  </head>
  <body style="margin:0;padding:24px;background:#030A1D;font-family:Inter,Segoe UI,Arial,sans-serif;color:#D6E3FF;">
    <table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="max-width:720px;margin:0 auto;border:1px solid #1F3B7A;border-radius:10px;overflow:hidden;background:linear-gradient(180deg,#081532 0%,#061126 100%);">
      <tr>
        <td style="padding:20px 24px;border-bottom:1px solid #173267;">
          <div style="font-size:12px;letter-spacing:2px;text-transform:uppercase;color:#7EA8FF;">Traffic Detection Alert</div>
          <div style="font-size:24px;font-weight:700;color:#EAF2FF;margin-top:8px;">Accident Detected</div>
          <div style="font-size:13px;color:#9CB6E5;margin-top:6px;">This is an automated event notification from your video detection pipeline.</div>
        </td>
      </tr>
      <tr>
        <td style="padding:20px 24px;">
          <table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="background:#07142D;border:1px solid #173267;border-radius:8px;">
            <tr>
              <td style="padding:10px 12px;color:#8FA8D9;font-size:12px;">Video</td>
              <td style="padding:10px 12px;color:#EAF2FF;font-size:13px;text-align:right;">{file_name}</td>
            </tr>
            <tr>
              <td style="padding:10px 12px;color:#8FA8D9;font-size:12px;">Model</td>
              <td style="padding:10px 12px;color:#EAF2FF;font-size:13px;text-align:right;">{model_name}</td>
            </tr>
            <tr>
              <td style="padding:10px 12px;color:#8FA8D9;font-size:12px;">Detection Time</td>
              <td style="padding:10px 12px;color:#EAF2FF;font-size:13px;text-align:right;">{detection_time}</td>
            </tr>
            <tr>
              <td style="padding:10px 12px;color:#8FA8D9;font-size:12px;">Score / Threshold</td>
              <td style="padding:10px 12px;color:#EAF2FF;font-size:13px;text-align:right;">{score:.1f}% / {threshold:.1f}%</td>
            </tr>
            <tr>
              <td style="padding:10px 12px;color:#8FA8D9;font-size:12px;">Class</td>
              <td style="padding:10px 12px;color:#EAF2FF;font-size:13px;text-align:right;">{class_name}</td>
            </tr>
            <tr>
              <td style="padding:10px 12px;color:#8FA8D9;font-size:12px;">Job ID</td>
              <td style="padding:10px 12px;color:#EAF2FF;font-size:13px;text-align:right;">{job_id}</td>
            </tr>
          </table>
          <div style="margin-top:14px;padding:10px 12px;background:#081B3F;border:1px solid #234C93;border-radius:8px;color:#AFC6EF;font-size:12px;">
            Snapshot from the detected event is attached below with an embedded timestamp overlay.
          </div>
          <div style="margin-top:16px;border:1px solid #1A3C7C;border-radius:8px;overflow:hidden;background:#040B1E;text-align:center;">
            <img src="cid:accident_snapshot" alt="Accident Snapshot" style="display:block;width:100%;height:auto;" />
          </div>
        </td>
      </tr>
      <tr>
        <td style="padding:14px 24px;background:#020915;border-top:1px solid #173267;color:#6784B8;font-size:11px;">
          Generated at {generated_at}
        </td>
      </tr>
    </table>
  </body>
</html>
"""


def send_accident_email(
    recipient: str,
    payload: Dict,
    snapshot_png: bytes,
) -> Dict[str, str]:
    """
    Send accident alert email with an inline snapshot.
    Returns status metadata for API response/debugging.
    """
    if not ALERT_EMAIL_ENABLED:
        return {"sent": "false", "reason": "email_alerts_disabled"}
    if not is_valid_email(recipient):
        return {"sent": "false", "reason": "invalid_recipient"}
    if not is_email_configured():
        return {"sent": "false", "reason": "smtp_not_configured"}

    subject = f"[Accident Alert] {payload.get('file_name', 'Video')} - {_format_duration(payload.get('best_timestamp'))}"

    message = MIMEMultipart("related")
    message["Subject"] = subject
    message["From"] = f"{SMTP_FROM_NAME} <{SMTP_FROM_EMAIL}>"
    message["To"] = recipient

    alt_part = MIMEMultipart("alternative")
    message.attach(alt_part)
    alt_part.attach(MIMEText("Traffic Detection alert: accident detected. Open this email in HTML mode for full details.", "plain"))
    alt_part.attach(MIMEText(_build_html(payload), "html"))

    image_part = MIMEImage(snapshot_png, _subtype="png")
    image_part.add_header("Content-ID", "<accident_snapshot>")
    image_part.add_header("Content-Disposition", "inline", filename="accident_snapshot.png")
    message.attach(image_part)

    if SMTP_USE_TLS:
        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20, source_address=("0.0.0.0", 0)) as server:
            server.starttls(context=context)
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM_EMAIL, [recipient], message.as_string())
    else:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=20, source_address=("0.0.0.0", 0)) as server:
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM_EMAIL, [recipient], message.as_string())

    return {"sent": "true", "reason": "ok"}
