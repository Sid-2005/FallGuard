from flask import Blueprint, render_template, jsonify
from flask_login import login_required
from app.models.event import Event
from app.models.database import db
from sqlalchemy import func
from datetime import datetime, timedelta

analytics_bp = Blueprint('analytics', __name__)

@analytics_bp.route('/')
@login_required
def index():
    return render_template('pages/analytics.html')

@analytics_bp.route('/data')
@login_required
def data():
    now = datetime.utcnow()
    daily = []
    for i in range(6, -1, -1):
        day = now - timedelta(days=i)
        s = day.replace(hour=0, minute=0, second=0, microsecond=0)
        e = day.replace(hour=23, minute=59, second=59, microsecond=999999)
        count = Event.query.filter(Event.is_fall==True, Event.timestamp.between(s, e)).count()
        daily.append({'date': day.strftime('%b %d'), 'count': count})

    rows = db.session.query(Event.activity_label, func.count(Event.id)).group_by(Event.activity_label).all()
    activities = [{'label': r[0] or 'unknown', 'count': r[1]} for r in rows]

    total_falls = Event.query.filter_by(is_fall=True).count()
    total_events = Event.query.count()
    avg_conf = db.session.query(func.avg(Event.confidence_total)).filter_by(is_fall=True).scalar() or 0

    return jsonify({
        'daily': daily,
        'activity_distribution': activities,
        'stats': {'total_falls': total_falls, 'total_events': total_events, 'avg_confidence': round(float(avg_conf), 3)}
    })
