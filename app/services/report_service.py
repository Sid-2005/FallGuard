"""
FallGuard AI - PDF Report & CSV Export
"""

import csv
import io
import logging
from datetime import datetime
from typing import List

logger = logging.getLogger(__name__)


def export_events_csv(events: list) -> bytes:
    """Return CSV bytes for a list of Event dicts."""
    output = io.StringIO()
    if not events:
        return b''
    fieldnames = ['event_id','timestamp','event_type','activity_label','is_fall',
                  'confidence_total','body_angle','velocity','context_surface','detected_objects']
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    for e in events:
        writer.writerow(e)
    return output.getvalue().encode('utf-8')


def generate_pdf_report(events: list, user: dict, period: str = 'Monthly') -> bytes:
    """Generate a minimal HTML-based PDF report using weasyprint (if available)."""
    html = _build_report_html(events, user, period)
    try:
        from weasyprint import HTML
        return HTML(string=html).write_pdf()
    except ImportError:
        logger.warning("weasyprint not installed — returning HTML bytes")
        return html.encode('utf-8')


def _build_report_html(events: list, user: dict, period: str) -> str:
    fall_events  = [e for e in events if e.get('is_fall')]
    total_events = len(events)
    fall_count   = len(fall_events)
    avg_conf     = (sum(e.get('confidence_total', 0) for e in fall_events) / fall_count) if fall_count else 0
    rows = ''
    for e in events[:50]:
        ts    = e.get('timestamp','')[:19] if e.get('timestamp') else ''
        atype = e.get('activity_label','—')
        conf  = f"{e.get('confidence_total',0):.0%}"
        is_f  = '🚨 FALL' if e.get('is_fall') else '✅ Normal'
        rows += f"<tr><td>{ts}</td><td>{atype}</td><td>{conf}</td><td>{is_f}</td></tr>"

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body{{font-family:Arial;margin:40px;color:#1a1a2e}}
  h1{{color:#e94560}} h2{{color:#0f3460}}
  table{{width:100%;border-collapse:collapse;margin-top:20px}}
  th{{background:#0f3460;color:white;padding:10px;text-align:left}}
  td{{padding:8px;border-bottom:1px solid #ddd}}
  .stat{{display:inline-block;background:#f0f4ff;border-radius:8px;padding:15px 25px;margin:10px;text-align:center}}
  .stat .val{{font-size:2em;font-weight:bold;color:#e94560}}
</style></head><body>
<h1>FallGuard AI — {period} Report</h1>
<p>Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC &nbsp;|&nbsp; User: {user.get('full_name','—')}</p>
<div>
  <div class="stat"><div class="val">{total_events}</div>Total Events</div>
  <div class="stat"><div class="val">{fall_count}</div>Falls Detected</div>
  <div class="stat"><div class="val">{avg_conf:.0%}</div>Avg Confidence</div>
</div>
<h2>Event Log</h2>
<table><thead><tr><th>Timestamp</th><th>Activity</th><th>Confidence</th><th>Type</th></tr></thead>
<tbody>{rows}</tbody></table>
<p style="margin-top:40px;color:#999;font-size:11px">
FallGuard AI Healthcare Monitoring Platform — Confidential
</p></body></html>"""
