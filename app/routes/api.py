"""
FallGuard AI — REST API endpoints (Final)
"""
import io
from flask import Blueprint, jsonify, request, current_app, send_file
from flask_login import login_required, current_user
from app.models.event import Event

api_bp = Blueprint('api', __name__)


def _cam():
    return current_app.config.get('_camera_service')


# ── Camera control ────────────────────────────────────────────

@api_bp.route('/camera/start', methods=['POST'])
@login_required
def camera_start():
    cam = _cam()
    if cam is None:
        return jsonify({'success': False, 'message': 'Camera service unavailable'}), 503
    data   = request.get_json() or {}
    source = data.get('source', data.get('camera_index', 0))
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass
    return jsonify(cam.start(source))


@api_bp.route('/camera/stop', methods=['POST'])
@login_required
def camera_stop():
    cam = _cam()
    if cam is None:
        return jsonify({'success': False}), 503
    return jsonify(cam.stop())


@api_bp.route('/camera/status')
@login_required
def camera_status():
    cam = _cam()
    if cam is None:
        return jsonify({'is_running': False})
    return jsonify(cam.get_status())


@api_bp.route('/camera/switch', methods=['POST'])
@login_required
def camera_switch():
    cam = _cam()
    idx = (request.get_json() or {}).get('index', 0)
    return jsonify(cam.switch_camera(int(idx)))


@api_bp.route('/camera/upload', methods=['POST'])
@login_required
def camera_upload():
    import os
    from werkzeug.utils import secure_filename
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file provided'}), 400
    f    = request.files['file']
    name = secure_filename(f.filename)
    if not name:
        return jsonify({'success': False, 'message': 'Invalid filename'}), 400
    allowed = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.webm'}
    ext = os.path.splitext(name)[1].lower()
    if ext not in allowed:
        return jsonify({'success': False, 'message': f'File type {ext} not allowed'}), 400
    upload_dir = os.path.join('static', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    save_path = os.path.join(upload_dir, name)
    f.save(save_path)
    cam = _cam()
    if cam is None:
        return jsonify({'success': False, 'message': 'Camera service unavailable'}), 503
    if cam.is_running:
        cam.stop()
        import time; time.sleep(0.3)
    result = cam.start(source=save_path)
    result['filename'] = name
    return jsonify(result)


# ── Events ────────────────────────────────────────────────────

@api_bp.route('/events')
@login_required
def events():
    page       = int(request.args.get('page', 1))
    limit      = int(request.args.get('limit', 20))
    falls_only = request.args.get('falls_only', 'false').lower() == 'true'
    q = Event.query
    if falls_only:
        q = q.filter_by(is_fall=True)
    pag = q.order_by(Event.timestamp.desc()).paginate(
        page=page, per_page=limit, error_out=False
    )
    return jsonify({
        'events': [e.to_dict() for e in pag.items],
        'total':  pag.total,
        'pages':  pag.pages,
    })


@api_bp.route('/events/<string:event_id>/resolve', methods=['POST'])
@login_required
def resolve_event(event_id):
    from datetime import datetime
    event = Event.query.filter_by(event_id=event_id).first_or_404()
    event.is_resolved = True
    event.resolved_at = datetime.utcnow()
    from app import db
    db.session.commit()
    return jsonify({'success': True})


# ── Export ────────────────────────────────────────────────────

@api_bp.route('/export/csv')
@login_required
def export_csv():
    from app.services.report_service import export_events_csv
    evts = Event.query.order_by(Event.timestamp.desc()).all()
    csv_bytes = export_events_csv([e.to_dict() for e in evts])
    return send_file(
        io.BytesIO(csv_bytes), mimetype='text/csv',
        as_attachment=True, download_name='fallguard_events.csv'
    )


@api_bp.route('/export/pdf')
@login_required
def export_pdf():
    from app.services.report_service import generate_pdf_report
    evts = Event.query.order_by(Event.timestamp.desc()).limit(200).all()
    pdf  = generate_pdf_report([e.to_dict() for e in evts], current_user.to_dict())
    mime = 'application/pdf' if pdf[:4] == b'%PDF' else 'text/html'
    return send_file(
        io.BytesIO(pdf), mimetype=mime,
        as_attachment=True, download_name='fallguard_report.pdf'
    )


# ── Pipeline toggles ──────────────────────────────────────────

@api_bp.route('/detection/toggle', methods=['POST'])
@login_required
def toggle_pipeline_feature():
    import app.services.detection.detection_config as dcfg
    data    = request.get_json() or {}
    feature = data.get('feature')
    enabled = bool(data.get('enabled', False))
    labels  = {
        'object_detector': 'YOLO Object Detector',
        'floor_mapper':    'Floor Mapper',
        'pose_skip':       'Frame Skip',
    }
    if feature == 'object_detector':
        dcfg.ENABLE_OBJECT_DETECTOR = enabled
        det = current_app.config.get('_detector')
        if det and getattr(det, '_ready', False) and enabled:
            if hasattr(det, '_load_yolo') and getattr(det, '_yolo', None) is None:
                try:
                    det._load_yolo()
                except Exception as e:
                    return jsonify({'success': False, 'error': str(e)})
    elif feature == 'floor_mapper':
        dcfg.ENABLE_FLOOR_MAPPER = enabled
    elif feature == 'pose_skip':
        dcfg.POSE_SKIP_FRAMES = enabled
    else:
        return jsonify({'success': False, 'error': f'Unknown feature: {feature}'})
    return jsonify({
        'success': True,
        'label':   labels.get(feature, feature),
        'status':  {
            'object_detector': dcfg.ENABLE_OBJECT_DETECTOR,
            'floor_mapper':    dcfg.ENABLE_FLOOR_MAPPER,
            'pose_skip':       getattr(dcfg, 'POSE_SKIP_FRAMES', False),
        }
    })


@api_bp.route('/detection/pipeline_status')
@login_required
def pipeline_status():
    import app.services.detection.detection_config as dcfg
    return jsonify({
        'status': {
            'object_detector': dcfg.ENABLE_OBJECT_DETECTOR,
            'floor_mapper':    dcfg.ENABLE_FLOOR_MAPPER,
            'pose_skip':       getattr(dcfg, 'POSE_SKIP_FRAMES', False),
        }
    })


# ── Evaluation metrics ────────────────────────────────────────

@api_bp.route('/detection/metrics')
@login_required
def detection_metrics():
    detector = current_app.config.get('_detector')
    if detector is None or not getattr(detector, '_ready', False):
        return jsonify({'error': 'Detector not ready'})
    return jsonify(detector.get_metrics())


@api_bp.route('/detection/mark_fall', methods=['POST'])
@login_required
def mark_ground_truth_fall():
    detector = current_app.config.get('_detector')
    if detector:
        detector.mark_ground_truth_fall()
        return jsonify({'success': True})
    return jsonify({'success': False})


# ── Reports ──────────────────────────────────────────────────

@api_bp.route('/reports/daily')
@login_required
def daily_report():
    """Generate and return today's HTML report."""
    from app.services.daily_report import generate_daily_report
    html = generate_daily_report()
    from flask import Response
    return Response(html, mimetype='text/html')


@api_bp.route('/reports/daily/<date_str>')
@login_required
def daily_report_date(date_str):
    """Generate report for a specific date (YYYY-MM-DD)."""
    from app.services.daily_report import generate_daily_report
    from datetime import datetime
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
    html = generate_daily_report(date)
    from flask import Response
    return Response(html, mimetype='text/html')


@api_bp.route('/reports/weekly')
@login_required
def weekly_report():
    """Generate and return this week's HTML report."""
    from app.services.daily_report import generate_weekly_report
    html = generate_weekly_report()
    from flask import Response
    return Response(html, mimetype='text/html')


@api_bp.route('/reports/download/daily')
@login_required
def download_daily_report():
    """Download today's report as HTML file."""
    from app.services.daily_report import generate_daily_report, save_report
    from datetime import datetime
    html     = generate_daily_report()
    filename = f"fallguard_daily_{datetime.utcnow().strftime('%Y%m%d')}.html"
    path     = save_report(html, filename)
    return send_file(path, mimetype='text/html',
                     as_attachment=True, download_name=filename)


@api_bp.route('/reports/download/weekly')
@login_required
def download_weekly_report():
    """Download this week's report as HTML file."""
    from app.services.daily_report import generate_weekly_report, save_report
    from datetime import datetime
    html     = generate_weekly_report()
    filename = f"fallguard_weekly_{datetime.utcnow().strftime('%Y%m%d')}.html"
    path     = save_report(html, filename)
    return send_file(path, mimetype='text/html',
                     as_attachment=True, download_name=filename)


# ── Video clips ───────────────────────────────────────────────

@api_bp.route('/clips')
@login_required
def list_clips():
    """List all saved video clips."""
    import os, glob
    clip_dir = os.path.join('static', 'clips')
    if not os.path.exists(clip_dir):
        return jsonify({'clips': []})
    files = sorted(glob.glob(os.path.join(clip_dir, '*.mp4')) +
                   glob.glob(os.path.join(clip_dir, '*.avi')),
                   reverse=True)
    clips = []
    for f in files[:50]:   # max 50 clips
        name = os.path.basename(f)
        size = os.path.getsize(f) / 1_000_000
        mtime = os.path.getmtime(f)
        from datetime import datetime
        clips.append({
            'filename':  name,
            'path':      f'static/clips/{name}',
            'size_mb':   round(size, 1),
            'created':   datetime.fromtimestamp(mtime).isoformat(),
        })
    return jsonify({'clips': clips})


# ── Video quality control ────────────────────────────────────

@api_bp.route('/detection/quality', methods=['POST'])
@login_required
def set_quality():
    """Set broadcast resolution live — 280 / 360 / 480."""
    data  = request.get_json() or {}
    width = int(data.get('width', 360))
    if width not in (280, 360, 480):
        return jsonify({'success': False, 'message': 'Width must be 280, 360 or 480'})
    cam = _cam()
    if cam:
        cam._broadcast_width = width
    return jsonify({'success': True, 'width': width})


# ── Voice alert control ───────────────────────────────────────

@api_bp.route('/voice/toggle', methods=['POST'])
@login_required
def toggle_voice():
    """Enable or disable voice alerts."""
    cam = _cam()
    if not cam or not cam._voice:
        return jsonify({'success': False, 'message': 'Voice not available'})
    data    = request.get_json() or {}
    enabled = data.get('enabled', not cam._voice.enabled)
    if enabled:
        cam._voice.enable()
    else:
        cam._voice.disable()
    return jsonify({'success': True, 'enabled': cam._voice.enabled})


@api_bp.route('/voice/test', methods=['POST'])
@login_required
def test_voice():
    """Send a test voice alert."""
    cam = _cam()
    if not cam or not cam._voice:
        return jsonify({'success': False, 'message': 'Voice not available'})
    cam._voice.speak_custom("FallGuard AI voice alert test")
    return jsonify({'success': True})


# ── Detection status ──────────────────────────────────────────

@api_bp.route('/detection/status')
@login_required
def detection_status():
    detector = current_app.config.get('_detector')
    if detector is None:
        return jsonify({'ready': False, 'error': 'Detector not initialised'})
    backend = 'unknown'
    if detector.is_ready and getattr(detector, '_pose', None) is not None:
        backend = 'mediapipe'
    return jsonify({
        'ready':   detector.is_ready,
        'error':   detector.error_message,
        'fps':     detector.fps,
        'backend': backend,
    })


@api_bp.route('/detection/persons')
@login_required
def live_persons():
    detector = current_app.config.get('_detector')
    if detector is None or not getattr(detector, '_ready', False):
        return jsonify({'persons': [], 'fps': 0})
    last = getattr(detector, '_last_result', None)
    if last is None:
        return jsonify({'persons': [], 'fps': detector.fps})
    return jsonify({'persons': last.persons, 'fps': last.fps})


@api_bp.route('/detection/floor_mask', methods=['POST'])
@login_required
def toggle_floor_mask():
    detector = current_app.config.get('_detector')
    if detector and detector._ready and detector._renderer:
        detector._renderer.show_floor_mask = not detector._renderer.show_floor_mask
        state = detector._renderer.show_floor_mask
    else:
        state = False
    return jsonify({'floor_mask': state})
