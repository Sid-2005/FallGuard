"""
multi_camera_manager.py - Runs one CameraService + FallDetector pair per
registered "local" CameraNode (e.g. a phone streaming video over Wi-Fi).

Each phone/room gets its own independent detector and its own PersonState,
so one room's activity can never affect another room's detection. All of
them share the SAME socketio server -- every event they emit is tagged with
camera_id (see CameraService._tag) so the dashboard can route each frame to
the right video tile.

This does NOT replace the existing single-camera page/flow (app.config
['_camera_service']) -- that keeps working exactly as before. This is an
additional, parallel set of camera+detector pairs for the central dashboard.
"""

import logging
from typing import Dict, Optional

from app.services.camera_service import CameraService
from app.services.fall_detector import FallDetector

logger = logging.getLogger(__name__)


class MultiCameraManager:
    def __init__(self, socketio, app, app_config: dict):
        self.socketio   = socketio
        self.app        = app
        self.app_config = app_config
        # node_id -> {'camera': CameraService, 'detector': FallDetector}
        self._cameras: Dict[str, dict] = {}

    def is_running(self, node_id: str) -> bool:
        entry = self._cameras.get(node_id)
        return bool(entry and entry['camera'].is_running)

    def start(self, node) -> dict:
        """node: a CameraNode row with is_local=True and a stream_url."""
        if not node.stream_url:
            return {'success': False, 'message': 'No stream URL set for this camera'}

        if self.is_running(node.id):
            return {'success': False, 'message': 'Already running'}

        # Reuse an existing detector for this node if we have one (keeps
        # PersonState continuity across stop/start), otherwise build fresh.
        entry = self._cameras.get(node.id)
        if entry is None:
            try:
                # YOLO (furniture/context detection) is the single most
                # expensive thing FallDetector does on CPU (~25-40ms/frame),
                # and it's the first thing that turns "a few cameras" into
                # "everything lags." Multicam nodes skip it and fall back to
                # lightweight background-subtraction person detection
                # instead -- fall confirmation itself (posture + stillness
                # over time) is unaffected; only the "ignore it, they're on
                # a bed/couch" suppression is less precise. Flip
                # multicam_enable_yolo=True in app_config if you'd rather
                # keep it and have fewer cameras running smoothly.
                enable_yolo = bool(self.app_config.get('multicam_enable_yolo', False))
                detector = FallDetector(config=self.app_config, socketio=self.socketio,
                                         enable_yolo=enable_yolo)
            except Exception as e:
                logger.error("Detector init failed for node %s: %s", node.id, e)
                return {'success': False, 'message': f'Detector init failed: {e}'}

            camera = CameraService(
                socketio    = self.socketio,
                detector    = detector,
                app_config  = self.app_config,
                app         = self.app,
                camera_id   = node.id,
                camera_name = node.name,
            )
            entry = {'camera': camera, 'detector': detector}
            self._cameras[node.id] = entry

        result = entry['camera'].start(node.stream_url)
        if result.get('success'):
            self._rebalance()
        return result

    def stop(self, node_id: str) -> dict:
        entry = self._cameras.get(node_id)
        if entry is None:
            return {'success': True, 'message': 'Not running'}
        result = entry['camera'].stop()
        self._rebalance()
        return result

    def _rebalance(self):
        """Scale how often each running camera does real inference based on
        how many are running at once, so total CPU load from detection
        stays roughly flat instead of growing with every camera you add
        (which is what causes lag to appear only once you have several
        cameras going). 1-2 cameras: inference every 2nd frame (smooth,
        responsive). 3+: back off further so each one still gets a fair,
        uncontended share of the CPU instead of all of them crawling."""
        running = [e['camera'] for e in self._cameras.values() if e['camera'].is_running]
        n = len(running)
        if n == 0:
            return
        every_n = 2 if n <= 2 else (3 if n <= 4 else 4)
        for cam in running:
            cam.set_detect_every_n(every_n)
        logger.info("Rebalanced %d active camera(s): detect_every_n=%d", n, every_n)

    def remove(self, node_id: str):
        """Fully tear down a camera (call this when a node is deleted)."""
        entry = self._cameras.pop(node_id, None)
        if entry:
            try:
                entry['camera'].stop()
            except Exception:
                pass
        self._rebalance()

    def status(self, node_id: str) -> Optional[dict]:
        entry = self._cameras.get(node_id)
        if entry is None:
            return None
        return entry['camera'].get_status()

    def start_all_enabled(self, nodes):
        """Call once at server startup to bring up every enabled local node."""
        for node in nodes:
            if node.is_local and node.enabled and node.stream_url:
                res = self.start(node)
                if not res.get('success'):
                    logger.warning("Could not auto-start camera %s: %s",
                                   node.name, res.get('message'))
        self._rebalance()