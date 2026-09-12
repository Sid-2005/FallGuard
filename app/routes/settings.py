"""
FallGuard AI - Settings routes
Saves user preferences and hot-reloads detector thresholds.
"""
from flask import Blueprint, render_template, request, jsonify, current_app
from flask_login import login_required, current_user
from app.models.database import db
from app.models.settings_model import UserSettings
from app.models.contact import EmergencyContact

settings_bp = Blueprint('settings', __name__)


@settings_bp.route('/')
@login_required
def index():
    s = UserSettings.query.filter_by(user_id=current_user.id).first()
    if not s:
        s = UserSettings(user_id=current_user.id)
        db.session.add(s)
        db.session.commit()
    contacts = EmergencyContact.query.filter_by(user_id=current_user.id).all()
    return render_template('pages/settings.html',
                           settings=s.to_dict(),
                           contacts=[c.to_dict() for c in contacts])


@settings_bp.route('/save', methods=['POST'])
@login_required
def save():
    data = request.get_json() or {}
    s = UserSettings.query.filter_by(user_id=current_user.id).first()
    if not s:
        s = UserSettings(user_id=current_user.id)
        db.session.add(s)

    for field in ['confidence_threshold', 'alert_cooldown',
                  'inactivity_threshold', 'camera_index']:
        if field in data:
            val = float(data[field]) if 'threshold' in field else int(data[field])
            setattr(s, field, val)

    for bfield in ['enable_sound', 'enable_browser_notif',
                   'enable_sms', 'enable_email', 'dark_mode']:
        if bfield in data:
            setattr(s, bfield, bool(data[bfield]))

    db.session.commit()

    # Hot-reload detector thresholds (no restart required)
    detector = current_app.config.get('_detector')
    if detector:
        detector.update_config({
            'FALL_CONFIDENCE_THRESHOLD': s.confidence_threshold,
            'ALERT_COOLDOWN_SECONDS':    s.alert_cooldown,
            'INACTIVITY_THRESHOLD_SECONDS': s.inactivity_threshold,
        })

    return jsonify({'success': True})


@settings_bp.route('/contacts/add', methods=['POST'])
@login_required
def add_contact():
    data = request.get_json(silent=True) or {}
    name = (data.get('name') or '').strip()
    if not name:
        return jsonify({'success': False, 'error': 'Name is required'}), 400
    try:
        allowed = ['name', 'relationship', 'phone', 'email', 'notify_sms', 'notify_email']
        c = EmergencyContact(
            user_id=current_user.id,
            **{k: data[k] for k in allowed if k in data}
        )
        db.session.add(c)
        db.session.commit()
        return jsonify({'success': True, 'contact': c.to_dict()})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@settings_bp.route('/contacts/delete/<int:cid>', methods=['DELETE'])
@login_required
def delete_contact(cid):
    c = EmergencyContact.query.filter_by(id=cid, user_id=current_user.id).first_or_404()
    db.session.delete(c)
    db.session.commit()
    return jsonify({'success': True})
