"""
FallGuard AI - Camera Service (Merged)

Uses Project 1's simple reliable synchronous capture loop
+ Project 22's voice alerts, video clips, SocketIO broadcast.
"""

import cv2
import threading
import time
import base64
import logging
import os
import numpy as np
from datetime import datetime
from typing import Optional, Union

logger = logging.getLogger(__name__)

UPLOAD_DIR = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _resize_frame(frame: np.ndarray, max_width: int = 480) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)),
                      interpolation=cv2.INTER_NEAREST)


def _looks_like_local_file(source) -> bool:
    """A phone/IP camera URL is a string too, but it's not a video file --
    it needs live-stream handling (no buffering, no FPS-based throttling,
    retry-on-hiccup instead of 'video ended'). Only treat a string source
    as a file if it doesn't look like a network stream address."""
    if not isinstance(source, str):
        return False
    return not source.lower().startswith(
        ('http://', 'https://', 'rtsp://', 'rtmps://', 'rtmp://', 'tcp://', 'udp://')
    )


class CameraService:
    def __init__(self, socketio, detector, app_config: dict, app=None,
                 camera_id: str = None, camera_name: str = None):
        self.socketio    = socketio
        self.detector    = detector
        self.config      = app_config
        self.app         = app
        self.camera_id   = camera_id     # None = legacy single-camera mode (unchanged behavior)
        self.camera_name = camera_name or 'Camera'

        self.cap              = None
        self.camera_index     = app_config.get('camera_index', 0)
        self.is_running       = False
        self._thread          = None
        self._lock            = threading.Lock()
        # Multi-camera nodes run several of these at once on the same CPU,
        # so give each one a lighter capture/broadcast profile than the
        # primary single-camera Monitor page. This is the main lever against
        # multi-camera lag -- it cuts per-camera CPU and network load
        # noticeably with only a modest drop in stream sharpness.
        self._is_multicam_node = camera_id is not None
        self._broadcast_width  = 240 if self._is_multicam_node else 360
        # Skip actual pose/YOLO inference on some frames for multi-camera
        # nodes -- this is the main lever, since inference (not resizing or
        # broadcasting) is what actually pegs the CPU when several cameras
        # run at once. Every camera still shows a live-ish feed (raw frames
        # in between), just with detection running less often.
        self._detect_every_n   = 3 if self._is_multicam_node else 1
        self._detect_max_width = 360 if self._is_multicam_node else 480
        self._last_light = {'activity': 'Starting...', 'is_fall': False, 'confidence': 0.0}

        self.alert_cooldown  = float(app_config.get('ALERT_COOLDOWN_SECONDS', 30))
        self.last_alert_time = 0.0
        self._prev_is_fall   = False   # tracks rising edge so we alert once per episode, not every frame
        self.frame_count     = 0
        self.start_time      = None
        self._source_is_file = False
        self._frame_delay    = 0.0

        self.event_image_dir = app_config.get('EVENT_IMAGE_DIR', 'static/events')
        os.makedirs(self.event_image_dir, exist_ok=True)

        # Voice alert
        try:
            from app.services.voice_alert import VoiceAlert
            self._voice = VoiceAlert()
        except Exception as e:
            self._voice = None
            logger.warning("Voice alerts unavailable: %s", e)

        # Video clip recorder
        try:
            from app.services.video_clip_recorder import VideoClipRecorder
            self._clipper = VideoClipRecorder(
                output_dir   = app_config.get('CLIP_DIR', 'static/clips'),
                pre_seconds  = 5,
                post_seconds = 8,
                fps          = 20.0,
            )
        except Exception as e:
            self._clipper = None
            logger.warning("Clip recorder unavailable: %s", e)

    # ── Public API ────────────────────────────────────────────

    def start(self, source: Union[int, str] = None):
        if self.is_running:
            return {'success': False, 'message': 'Already running — stop first'}
        if source is None:
            source = self.camera_index

        self._source_is_file = _looks_like_local_file(source)
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            return {'success': False,
                    'message': 'Cannot open camera. Check camera index or file path.'}

        if not self._source_is_file:
            if self._is_multicam_node:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  480)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
                self.cap.set(cv2.CAP_PROP_FPS,          12)
            else:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS,          20)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
            self._frame_delay = 0.0
        else:
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 20
            self._frame_delay = 1.0 / fps

        self.is_running  = True
        self.frame_count = 0
        self.start_time  = datetime.utcnow()
        self._prev_is_fall = False
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True
        )
        self._thread.start()
        logger.info("CameraService started source=%s", source)
        return {'success': True, 'message': 'Started'}

    def stop(self):
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        with self._lock:
            if self.cap:
                self.cap.release()
                self.cap = None
        self._thread = None
        return {'success': True, 'message': 'Stopped'}

    def set_detect_every_n(self, n: int):
        """Called by MultiCameraManager to scale how often this camera runs
        real inference, based on how many multicam cameras are running at
        once. Keeps total system-wide inference load roughly constant
        instead of it growing linearly (and lagging) with camera count."""
        self._detect_every_n = max(1, int(n))

    def switch_camera(self, index: int):
        self.stop()
        time.sleep(0.3)
        self.camera_index = index
        return self.start(index)

    def get_status(self) -> dict:
        uptime = 0
        if self.start_time:
            uptime = int((datetime.utcnow() - self.start_time).total_seconds())
        return {
            'is_running':     self.is_running,
            'camera_index':   self.camera_index,
            'fps':            self.detector.fps if self.detector else 0,
            'frame_count':    self.frame_count,
            'uptime_seconds': uptime,
            'source_is_file': self._source_is_file,
        }

    # ── Capture loop (Project 1 style — simple + reliable) ────

    def _capture_loop(self):
        consecutive_failures = 0
        warmup = 10 if not self._source_is_file else 0

        while self.is_running:
            t_start = time.time()

            with self._lock:
                if self.cap is None or not self.cap.isOpened():
                    break
                ret, frame = self.cap.read()

                # For live streams (phone/IP camera), the decoder can build
                # up a backlog faster than we consume it -- CAP_PROP_BUFFERSIZE
                # is ignored by most network stream backends, so lag quietly
                # grows over minutes instead of staying constant. If another
                # frame is already sitting there waiting (grab() returns
                # almost instantly), skip forward to it instead of processing
                # a frame that's already stale by the time we get to it.
                if ret and not self._source_is_file:
                    skipped = 0
                    while skipped < 4:
                        t0 = time.time()
                        ok = self.cap.grab()
                        if not ok or (time.time() - t0) > 0.004:
                            break
                        skipped += 1
                    if skipped:
                        ok2, newer = self.cap.retrieve()
                        if ok2:
                            frame = newer

            if not ret:
                if self._source_is_file:
                    self.socketio.emit('video_ended', self._tag({
                        'message': 'Video playback complete',
                        'frame_count': self.frame_count,
                    }))
                    break
                consecutive_failures += 1
                if consecutive_failures > 60:
                    self.socketio.emit('detection_error', self._tag({
                        'message': 'Camera stopped responding.'
                    }))
                    break
                time.sleep(0.05)
                continue

            consecutive_failures = 0

            # Warmup — show raw frames, skip detection
            if warmup > 0:
                warmup -= 1
                self._broadcast_raw(frame)
                time.sleep(0.05)
                continue

            self.frame_count += 1

            # For multi-camera nodes, only run the actual pose/YOLO
            # inference every Nth frame -- that's the expensive part. On
            # skipped frames we still broadcast the current raw frame (so
            # the tile doesn't visibly freeze) but reuse the last known
            # activity/confidence rather than resetting it, so the label on
            # the dashboard doesn't flicker back to "Starting..." every
            # other frame.
            if self._detect_every_n > 1 and (self.frame_count % self._detect_every_n != 0):
                self._broadcast_raw(frame, reuse_last=True)
                if self._source_is_file and self._frame_delay > 0:
                    elapsed = time.time() - t_start
                    sleep   = self._frame_delay - elapsed
                    if sleep > 0:
                        time.sleep(sleep)
                continue

            # Run detection — simple synchronous call (Project 1 style)
            try:
                proc   = _resize_frame(frame, max_width=self._detect_max_width)
                result = self.detector.process_frame(proc)
                self._last_light = {
                    'activity': result.activity, 'is_fall': result.is_fall,
                    'confidence': round(result.confidence, 3),
                }

                # Push to clip buffer
                if self._clipper and result.annotated_frame is not None:
                    self._clipper.push_frame(result.annotated_frame)

                self._broadcast_frame(result)

                # Alert once per fall episode (on the rising edge into
                # is_fall=True), not on every frame while the person stays
                # down. Repeated alerts every ~30s and confidence values
                # that don't match what actually triggered the fall both
                # came from firing this on every True frame instead of the
                # transition into it.
                if result.is_fall and not self._prev_is_fall:
                    self._handle_fall(result, proc)
                self._prev_is_fall = result.is_fall

            except Exception as e:
                logger.error("Frame processing error: %s", e, exc_info=True)
                self._broadcast_raw(frame)

            # Frame pacing
            if self._source_is_file and self._frame_delay > 0:
                elapsed = time.time() - t_start
                sleep   = self._frame_delay - elapsed
                if sleep > 0:
                    time.sleep(sleep)

        self.is_running = False
        with self._lock:
            if self.cap:
                self.cap.release()
                self.cap = None
        logger.info("Capture loop ended frames=%d", self.frame_count)

    # ── Broadcast ─────────────────────────────────────────────

    def _tag(self, payload: dict) -> dict:
        """Adds camera_id/camera_name to an emit payload. In legacy
        single-camera mode (camera_id is None) this is a no-op, so nothing
        about the existing single-camera page changes."""
        if self.camera_id is not None:
            payload['camera_id']   = self.camera_id
            payload['camera_name'] = self.camera_name
        return payload

    def _broadcast_raw(self, frame, reuse_last: bool = False):
        try:
            disp = _resize_frame(frame, max_width=self._broadcast_width)
            q    = {240: 30, 280: 35, 360: 45, 480: 60}.get(self._broadcast_width, 45)
            _, buf = cv2.imencode('.jpg', disp, [cv2.IMWRITE_JPEG_QUALITY, q])
            b64    = base64.b64encode(buf).decode('utf-8')
            # On a detection-skip frame (multi-camera perf mode), keep
            # showing the last known activity/fall state instead of
            # resetting to "Starting..." every other frame -- we simply
            # didn't run inference this frame, nothing actually changed.
            light = self._last_light if reuse_last else {
                'activity': 'Starting...', 'is_fall': False, 'confidence': 0.0,
            }
            self.socketio.emit('frame_update', self._tag({
                'frame': b64,
                'activity':   light['activity'],
                'is_fall':    light['is_fall'],
                'confidence': light['confidence'],
                'conf_posture': 0.0, 'conf_velocity': 0.0,
                'conf_height': 0.0, 'conf_inactivity': 0.0,
                'conf_context': 0.0, 'body_angle': 0.0,
                'velocity': 0.0, 'context_surface': 'unknown',
                'detected_objects': [], 'persons': [], 'fps': 0.0,
            }))
        except Exception as e:
            logger.debug("Raw broadcast error: %s", e)

    def _broadcast_frame(self, result):
        if result.annotated_frame is None:
            return
        try:
            disp = _resize_frame(result.annotated_frame,
                                  max_width=self._broadcast_width)
            q    = {280: 35, 360: 45, 480: 60}.get(self._broadcast_width, 45)
            _, buf = cv2.imencode('.jpg', disp, [cv2.IMWRITE_JPEG_QUALITY, q])
            b64    = base64.b64encode(buf).decode('utf-8')
            self.socketio.emit('frame_update', self._tag({
                'frame':            b64,
                'activity':         result.activity,
                'is_fall':          result.is_fall,
                'confidence':       round(result.confidence,     3),
                'conf_posture':     round(result.conf_posture,   3),
                'conf_velocity':    round(result.conf_velocity,  3),
                'conf_height':      round(result.conf_height,    3),
                'conf_inactivity':  round(result.conf_inactivity,3),
                'conf_context':     round(result.conf_context,   3),
                'body_angle':       round(result.body_angle,     1),
                'velocity':         round(result.velocity,       3),
                'context_surface':  result.context_surface,
                'detected_objects': result.detected_objects,
                'persons':          result.persons,
                'fps':              result.fps,
            }))
        except Exception as e:
            logger.debug("Broadcast error: %s", e)

    # ── Fall handling ─────────────────────────────────────────

    def _handle_fall(self, result, frame):
        now = time.time()
        cooldown = self._get_user_cooldown()
        if now - self.last_alert_time < cooldown:
            return
        self.last_alert_time = now

        ts    = datetime.utcnow()
        fname = f"fall_{ts.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        fpath = os.path.join(self.event_image_dir, fname)
        try:
            cv2.imwrite(fpath, frame)
        except Exception as e:
            logger.error("Could not save event image: %s", e)
            fname = None

        image_path = f"events/{fname}" if fname else None
        event_id   = self._save_event(result, image_path, ts)

        event_data = {
            'event_id':        event_id,
            'timestamp':       ts.isoformat(),
            'confidence':      result.confidence,
            'activity':        result.activity,
            'image_path':      image_path,
            'conf_posture':    result.conf_posture,
            'conf_velocity':   result.conf_velocity,
            'conf_height':     result.conf_height,
            'conf_inactivity': result.conf_inactivity,
            'conf_context':    result.conf_context,
            'persons':         result.persons,
        }
        self.socketio.emit('fall_alert', self._tag(event_data))
        self._dispatch_external_alerts(event_data)

        if self._voice:
            persons = result.persons
            pid     = persons[0]['track_id'] if persons else None
            self._voice.speak_fall(
                location  = self.config.get('ward_name', ''),
                person_id = pid,
            )

        if self._clipper:
            ward = self.config.get('ward_name', 'ward')
            clip = self._clipper.save_clip(label=ward)
            if clip:
                clip_path = clip.replace(os.sep, "/")
                self.socketio.emit('fall_clip_ready', self._tag({
                    'event_id':  event_id,
                    'clip_path': clip_path,
                }))

        logger.warning("FALL DETECTED conf=%.2f event_id=%s",
                       result.confidence, event_id)

    def _get_user_cooldown(self) -> float:
        if not self.app:
            return self.alert_cooldown
        try:
            with self.app.app_context():
                from app.models.settings_model import UserSettings
                s = UserSettings.query.first()
                if s: return float(s.alert_cooldown)
        except Exception:
            pass
        return self.alert_cooldown

    def _save_event(self, result, image_path, ts) -> str:
        import json
        if not self.app:
            logger.error("No app context available to save event")
            return 'unknown'
        try:
            with self.app.app_context():
                from app import db
                from app.models.event import Event
                event = Event(
                    event_type            = 'fall',
                    activity_label        = result.activity,
                    is_fall               = True,
                    confidence_total      = result.confidence,
                    confidence_posture    = result.conf_posture,
                    confidence_velocity   = result.conf_velocity,
                    confidence_height     = result.conf_height,
                    confidence_inactivity = result.conf_inactivity,
                    confidence_context    = result.conf_context,
                    body_angle            = result.body_angle,
                    velocity              = result.velocity,
                    detected_objects      = json.dumps(result.detected_objects),
                    image_path            = image_path,
                    timestamp             = ts,
                )
                db.session.add(event)
                db.session.commit()
                event_id = event.event_id
                return event_id
        except Exception as e:
            logger.error("DB save error: %s", e)
            return 'unknown'

    def _dispatch_external_alerts(self, event_data: dict):
        if not self.app:
            return
        try:
            with self.app.app_context():
                from app.models.settings_model  import UserSettings
                from app.models.contact         import EmergencyContact
                from app.services.alert_service import AlertService
                settings = UserSettings.query.first()
                if not settings: return
                contacts = [c.to_dict() for c in EmergencyContact.query.all()]
                if not contacts: return
                AlertService(self.config).dispatch_fall_alert(
                    event_data, contacts, settings.to_dict()
                )
        except Exception as e:
            logger.error("External alert dispatch error: %s", e)