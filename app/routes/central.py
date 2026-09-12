"""
central.py – Routes for multi-camera central dashboard and camera registry.
"""

from flask import Blueprint, render_template, request, jsonify, current_app
from flask_login import login_required
from app.models.database import db
from app.models.camera_node import CameraNode

central_bp = Blueprint('central', __name__)


def _mgr():
    return current_app.config.get('_multi_camera_manager')


# ── Pages ─────────────────────────────────────────────────────

@central_bp.route('/dashboard/central')
@login_required
def central_dashboard():
    """Main multi-camera monitoring dashboard."""
    nodes = CameraNode.query.filter_by(enabled=True).all()
    return render_template('pages/central_dashboard.html',
                           nodes=[n.to_dict() for n in nodes])


@central_bp.route('/dashboard/registry')
@login_required
def camera_registry():
    """Camera device registration page."""
    nodes = CameraNode.query.order_by(CameraNode.created_at).all()
    return render_template('pages/camera_registry.html',
                           nodes=[n.to_dict() for n in nodes])


# ── API ───────────────────────────────────────────────────────

@central_bp.route('/api/nodes', methods=['GET'])
@login_required
def get_nodes():
    nodes = CameraNode.query.order_by(CameraNode.created_at).all()
    return jsonify({'nodes': [n.to_dict() for n in nodes]})


@central_bp.route('/api/nodes', methods=['POST'])
@login_required
def add_node():
    data       = request.get_json() or {}
    name       = data.get('name', '').strip()
    is_local   = bool(data.get('is_local', False))
    stream_url = data.get('stream_url', '').strip()
    host       = data.get('host', '').strip()

    if not name:
        return jsonify({'success': False, 'message': 'Name is required'}), 400
    if is_local and not stream_url:
        return jsonify({'success': False,
                         'message': 'stream_url is required for a local camera '
                                    '(e.g. http://192.168.1.42:8080/video from '
                                    'the IP Webcam app on the phone)'}), 400
    if not is_local and not host:
        return jsonify({'success': False, 'message': 'Host is required for a remote node'}), 400

    # Max 4 cameras
    if CameraNode.query.count() >= 4:
        return jsonify({'success': False, 'message': 'Maximum 4 cameras allowed'}), 400

    node = CameraNode(
        name       = name,
        location   = data.get('location', ''),
        host       = host or '',
        port       = int(data.get('port', 5000)) if not is_local else 0,
        enabled    = True,
        is_local   = is_local,
        stream_url = stream_url,
    )
    db.session.add(node)
    db.session.commit()

    # Auto-start local cameras as soon as they're registered
    if is_local:
        mgr = _mgr()
        if mgr:
            res = mgr.start(node)
            node.is_running = bool(res.get('success'))
            db.session.commit()

    return jsonify({'success': True, 'node': node.to_dict()})


@central_bp.route('/api/nodes/<node_id>/camera/start', methods=['POST'])
@login_required
def start_local_camera(node_id):
    node = CameraNode.query.get_or_404(node_id)
    if not node.is_local:
        return jsonify({'success': False, 'message': 'Not a local camera node'}), 400
    mgr = _mgr()
    if mgr is None:
        return jsonify({'success': False, 'message': 'Camera manager unavailable'}), 503
    result = mgr.start(node)
    node.is_running = bool(result.get('success'))
    db.session.commit()
    return jsonify(result)


@central_bp.route('/api/nodes/<node_id>/camera/stop', methods=['POST'])
@login_required
def stop_local_camera(node_id):
    node = CameraNode.query.get_or_404(node_id)
    if not node.is_local:
        return jsonify({'success': False, 'message': 'Not a local camera node'}), 400
    mgr = _mgr()
    if mgr is None:
        return jsonify({'success': False, 'message': 'Camera manager unavailable'}), 503
    result = mgr.stop(node_id)
    node.is_running = False
    db.session.commit()
    return jsonify(result)


@central_bp.route('/api/nodes/<node_id>', methods=['DELETE'])
@login_required
def delete_node(node_id):
    node = CameraNode.query.get_or_404(node_id)
    if node.is_local:
        mgr = _mgr()
        if mgr:
            mgr.remove(node_id)
    db.session.delete(node)
    db.session.commit()
    return jsonify({'success': True})


@central_bp.route('/api/nodes/<node_id>/toggle', methods=['POST'])
@login_required
def toggle_node(node_id):
    node = CameraNode.query.get_or_404(node_id)
    node.enabled = not node.enabled
    db.session.commit()
    return jsonify({'success': True, 'enabled': node.enabled})