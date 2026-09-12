"""
daily_report.py – Daily and weekly patient activity report generator.

Generates HTML reports (printable / saveable as PDF via browser).
Reports include:
  - Fall event timeline
  - Per-hour activity breakdown
  - Camera uptime summary
  - Risk assessment per ward
  - Recommendations
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

log = logging.getLogger(__name__)


def _get_events(days_back: int = 1) -> List[dict]:
    """Fetch events from DB for the last N days."""
    try:
        from app.models.event import Event
        cutoff = datetime.utcnow() - timedelta(days=days_back)
        events = Event.query.filter(
            Event.timestamp >= cutoff
        ).order_by(Event.timestamp.desc()).all()
        return [e.to_dict() for e in events]
    except Exception as e:
        log.error("Could not fetch events: %s", e)
        return []


def _hour_buckets(events: List[dict]) -> Dict[int, int]:
    """Count falls per hour of day (0–23)."""
    buckets = {i: 0 for i in range(24)}
    for e in events:
        if e.get('is_fall'):
            try:
                ts = datetime.fromisoformat(e['timestamp'])
                buckets[ts.hour] += 1
            except Exception:
                pass
    return buckets


def _risk_level(fall_count: int) -> tuple:
    """Returns (label, colour) based on fall count."""
    if fall_count == 0:
        return "Low", "#3fb950"
    elif fall_count <= 2:
        return "Medium", "#e5c000"
    elif fall_count <= 5:
        return "High", "#f0883e"
    else:
        return "Critical", "#f85149"


def generate_daily_report(date: Optional[datetime] = None) -> str:
    """Generate HTML daily report. Returns HTML string."""
    if date is None:
        date = datetime.utcnow()

    start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end   = start + timedelta(days=1)

    try:
        from app.models.event import Event
        events = Event.query.filter(
            Event.timestamp >= start,
            Event.timestamp < end,
        ).order_by(Event.timestamp).all()
        events_dict = [e.to_dict() for e in events]
    except Exception:
        events_dict = []

    falls     = [e for e in events_dict if e.get('is_fall')]
    resolved  = [e for e in falls if e.get('is_resolved')]
    hour_data = _hour_buckets(events_dict)
    risk_lbl, risk_col = _risk_level(len(falls))

    peak_hour = max(hour_data, key=hour_data.get)
    peak_count = hour_data[peak_hour]

    date_str = date.strftime("%B %d, %Y")

    # Build hourly chart bars
    max_count = max(hour_data.values()) or 1
    bar_html  = ""
    for hour, count in hour_data.items():
        height = int((count / max_count) * 80) if count > 0 else 2
        color  = "#f85149" if count > 0 else "#21262d"
        label  = f"{hour:02d}:00"
        bar_html += f"""
        <div style="display:flex;flex-direction:column;align-items:center;gap:4px">
          <span style="font-size:9px;color:#8b949e">{count if count else ''}</span>
          <div style="width:24px;height:{height}px;background:{color};
                      border-radius:2px;min-height:2px"></div>
          <span style="font-size:8px;color:#555;transform:rotate(-45deg);
                       transform-origin:top right;white-space:nowrap">{label}</span>
        </div>"""

    # Build events table
    events_table = ""
    if falls:
        for e in falls:
            ts       = e.get('timestamp', '')[:19].replace('T', ' ')
            conf     = f"{int(e.get('confidence_total', 0) * 100)}%"
            resolved = "✓ Resolved" if e.get('is_resolved') else "⚠ Unresolved"
            res_col  = "#3fb950" if e.get('is_resolved') else "#f85149"
            events_table += f"""
            <tr>
              <td>{ts}</td>
              <td>{e.get('activity_label', '—')}</td>
              <td>{conf}</td>
              <td style="color:{res_col}">{resolved}</td>
            </tr>"""
    else:
        events_table = '<tr><td colspan="4" style="text-align:center;color:#8b949e;padding:20px">No fall events recorded today</td></tr>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FallGuard AI — Daily Report — {date_str}</title>
<style>
  * {{ margin:0;padding:0;box-sizing:border-box; }}
  body {{ font-family:'Segoe UI',Arial,sans-serif;background:#0d1117;color:#e6edf3;padding:32px; }}
  .header {{ display:flex;justify-content:space-between;align-items:center;
             margin-bottom:32px;padding-bottom:20px;border-bottom:1px solid #21262d }}
  .header h1 {{ font-size:24px;font-weight:700 }}
  .header .meta {{ font-size:13px;color:#8b949e;text-align:right }}
  .grid {{ display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px }}
  .stat {{ background:#161b22;border:1px solid #21262d;border-radius:8px;padding:20px;text-align:center }}
  .stat .val {{ font-size:36px;font-weight:800;margin:8px 0 }}
  .stat .lbl {{ font-size:12px;color:#8b949e;text-transform:uppercase;letter-spacing:.8px }}
  .card {{ background:#161b22;border:1px solid #21262d;border-radius:8px;
           padding:20px;margin-bottom:20px }}
  .card h3 {{ font-size:14px;font-weight:600;color:#8b949e;text-transform:uppercase;
              letter-spacing:.8px;margin-bottom:16px }}
  .chart {{ display:flex;align-items:flex-end;gap:4px;height:120px;
            padding-bottom:20px }}
  table {{ width:100%;border-collapse:collapse;font-size:13px }}
  th {{ padding:10px 12px;text-align:left;color:#8b949e;font-weight:500;
        border-bottom:1px solid #21262d;font-size:11px;text-transform:uppercase }}
  td {{ padding:10px 12px;border-bottom:1px solid #161b22;color:#e6edf3 }}
  tr:hover td {{ background:#1c2128 }}
  .badge {{ display:inline-block;padding:2px 10px;border-radius:20px;
            font-size:11px;font-weight:600 }}
  .footer {{ margin-top:32px;text-align:center;font-size:12px;color:#555 }}
  @media print {{
    body {{ background:#fff;color:#000;padding:20px }}
    .card,.stat {{ border:1px solid #ddd;background:#fff }}
    .card h3 {{ color:#555 }}
    th {{ color:#555 }}
    td {{ color:#000 }}
  }}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>⚡ FallGuard AI — Daily Report</h1>
    <div style="font-size:14px;color:#8b949e;margin-top:4px">{date_str}</div>
  </div>
  <div class="meta">
    Generated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}<br>
    <span class="badge" style="background:{risk_col}20;color:{risk_col};margin-top:6px;display:inline-block">
      Risk Level: {risk_lbl}
    </span>
  </div>
</div>

<!-- Stats -->
<div class="grid">
  <div class="stat">
    <div class="lbl">Total Falls</div>
    <div class="val" style="color:#f85149">{len(falls)}</div>
  </div>
  <div class="stat">
    <div class="lbl">Resolved</div>
    <div class="val" style="color:#3fb950">{len(resolved)}</div>
  </div>
  <div class="stat">
    <div class="lbl">Unresolved</div>
    <div class="val" style="color:#e5c000">{len(falls) - len(resolved)}</div>
  </div>
  <div class="stat">
    <div class="lbl">Peak Hour</div>
    <div class="val" style="color:#388bfd">{peak_hour:02d}:00</div>
    <div class="lbl">{peak_count} fall{'s' if peak_count != 1 else ''}</div>
  </div>
</div>

<!-- Hourly chart -->
<div class="card">
  <h3>Fall Events by Hour</h3>
  <div class="chart">
    {bar_html}
  </div>
</div>

<!-- Events table -->
<div class="card">
  <h3>Fall Event Log</h3>
  <table>
    <thead>
      <tr>
        <th>Time</th>
        <th>Activity</th>
        <th>Confidence</th>
        <th>Status</th>
      </tr>
    </thead>
    <tbody>
      {events_table}
    </tbody>
  </table>
</div>

<!-- Recommendations -->
<div class="card">
  <h3>Recommendations</h3>
  <ul style="padding-left:20px;line-height:2;font-size:14px;color:#c9d1d9">
    {'<li style="color:#f85149">High fall count detected — consider increasing monitoring frequency</li>' if len(falls) > 3 else ''}
    {'<li>Falls concentrated between ' + f'{peak_hour:02d}:00 and {(peak_hour+1):02d}:00' + ' — ensure staff availability during this period</li>' if peak_count > 0 else ''}
    {'<li>' + str(len(falls) - len(resolved)) + ' unresolved events require attention</li>' if (len(falls) - len(resolved)) > 0 else ''}
    {'<li style="color:#3fb950">No falls recorded today — monitoring effective</li>' if len(falls) == 0 else ''}
    <li>Review camera angles if detection confidence is consistently below 70%</li>
    <li>Ensure floor is clear of obstacles to reduce fall risk</li>
  </ul>
</div>

<div class="footer">
  FallGuard AI — Advanced Hospital Fall Detection System<br>
  This report is auto-generated. For medical decisions, consult qualified staff.
</div>

</body>
</html>"""

    return html


def generate_weekly_report() -> str:
    """Generate a 7-day summary report."""
    end   = datetime.utcnow()
    start = end - timedelta(days=7)

    try:
        from app.models.event import Event
        events = Event.query.filter(
            Event.timestamp >= start,
            Event.timestamp < end,
        ).order_by(Event.timestamp).all()
        events_dict = [e.to_dict() for e in events]
    except Exception:
        events_dict = []

    falls = [e for e in events_dict if e.get('is_fall')]

    # Per-day breakdown
    day_data = {}
    for i in range(7):
        day = (start + timedelta(days=i)).strftime("%a %d")
        day_data[day] = 0
    for e in falls:
        try:
            ts  = datetime.fromisoformat(e['timestamp'])
            day = ts.strftime("%a %d")
            if day in day_data:
                day_data[day] += 1
        except Exception:
            pass

    risk_lbl, risk_col = _risk_level(len(falls))

    bars = ""
    max_v = max(day_data.values()) or 1
    for day, count in day_data.items():
        h     = int((count / max_v) * 80) if count > 0 else 2
        color = "#f85149" if count > 0 else "#21262d"
        bars += f"""
        <div style="display:flex;flex-direction:column;align-items:center;gap:4px">
          <span style="font-size:10px;color:#8b949e">{count if count else ''}</span>
          <div style="width:36px;height:{h}px;background:{color};
                      border-radius:2px;min-height:2px"></div>
          <span style="font-size:10px;color:#8b949e">{day}</span>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FallGuard AI — Weekly Report</title>
<style>
  body {{ font-family:'Segoe UI',Arial,sans-serif;background:#0d1117;color:#e6edf3;padding:32px }}
  .card {{ background:#161b22;border:1px solid #21262d;border-radius:8px;padding:20px;margin-bottom:20px }}
  .card h3 {{ font-size:14px;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:.8px;margin-bottom:16px }}
  .chart {{ display:flex;align-items:flex-end;gap:12px;height:120px;padding-bottom:8px }}
  .stat-row {{ display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:20px }}
  .stat {{ background:#161b22;border:1px solid #21262d;border-radius:8px;padding:16px;text-align:center }}
  .stat .val {{ font-size:32px;font-weight:800;margin:6px 0 }}
  .stat .lbl {{ font-size:11px;color:#8b949e;text-transform:uppercase }}
  @media print {{ body {{ background:#fff;color:#000 }} .card,.stat {{ background:#fff;border:1px solid #ddd }} }}
</style>
</head>
<body>
<h1 style="margin-bottom:8px">⚡ FallGuard AI — Weekly Report</h1>
<p style="color:#8b949e;margin-bottom:24px">
  {start.strftime("%B %d")} – {end.strftime("%B %d, %Y")} &nbsp;·&nbsp;
  <span style="color:{risk_col}">Risk: {risk_lbl}</span>
</p>
<div class="stat-row">
  <div class="stat">
    <div class="lbl">Total Falls</div>
    <div class="val" style="color:#f85149">{len(falls)}</div>
  </div>
  <div class="stat">
    <div class="lbl">Daily Average</div>
    <div class="val" style="color:#388bfd">{len(falls)/7:.1f}</div>
  </div>
  <div class="stat">
    <div class="lbl">Busiest Day</div>
    <div class="val" style="color:#e5c000;font-size:20px">
      {max(day_data, key=day_data.get)}
    </div>
  </div>
</div>
<div class="card">
  <h3>Falls Per Day</h3>
  <div class="chart">{bars}</div>
</div>
<p style="font-size:12px;color:#555;text-align:center;margin-top:24px">
  FallGuard AI — Generated {end.strftime("%Y-%m-%d %H:%M UTC")}
</p>
</body>
</html>"""


def save_report(html: str, filename: str) -> str:
    """Save HTML report to static/reports/ and return path."""
    out_dir = os.path.join("static", "reports")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    log.info("Report saved: %s", path)
    return path
