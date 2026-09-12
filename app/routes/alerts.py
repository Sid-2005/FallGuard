from flask import Blueprint, render_template, jsonify, request
from flask_login import login_required
from app.models.event import Event
from app.models.database import db
from datetime import datetime

alerts_bp = Blueprint('alerts', __name__)

@alerts_bp.route('/')
@login_required
def index():
    falls = Event.query.filter_by(is_fall=True).order_by(Event.timestamp.desc()).limit(50).all()
    return render_template('pages/alerts.html', falls=[e.to_dict() for e in falls])

@alerts_bp.route('/resolve/<string:event_id>', methods=['POST'])
@login_required
def resolve(event_id):
    event = None
    try:
        val = int(event_id)
        event = Event.query.get(val)
    except ValueError:
        pass
    if not event:
        event = Event.query.filter_by(event_id=event_id).first_or_404()
    event.is_resolved = True
    event.resolved_at = datetime.utcnow()
    db.session.commit()
    return jsonify({'success': True})
