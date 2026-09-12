from flask import Blueprint, render_template, redirect, url_for, request
from flask_login import login_required, current_user
from app.models.event import Event
from app.models.contact import EmergencyContact

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/')
def index():
    if not current_user.is_authenticated:
        return redirect(url_for('auth.login'))
    return redirect(url_for('dashboard.monitor'))

@dashboard_bp.route('/monitor')
@login_required
def monitor():
    contacts = EmergencyContact.query.filter_by(user_id=current_user.id).all()
    recent   = Event.query.filter_by(is_fall=True).order_by(Event.timestamp.desc()).limit(5).all()
    return render_template('pages/monitor.html',
                           contacts=[c.to_dict() for c in contacts],
                           recent_events=[e.to_dict() for e in recent])

@dashboard_bp.route('/history')
@login_required
def history():
    page   = int(request.args.get('page', 1))
    events = Event.query.order_by(Event.timestamp.desc()).paginate(page=page, per_page=20, error_out=False)
    return render_template('pages/history.html', events=events)

@dashboard_bp.route('/about')
def about():
    return render_template('pages/about.html')
