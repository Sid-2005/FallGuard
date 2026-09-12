"""
FallGuard AI - Fall Detector (Merged)

Uses Project 1's working posture detection engine (IMAGE mode, proven accurate)
combined with Project 22's full web app features.

Key: RunningMode.IMAGE is used instead of VIDEO — no timestamp issues,
simpler, more reliable posture classification.
"""

import os

# Defensive thread caps -- normally set earlier by run.py, but this module
# can be imported directly (tests, scripts, a different entrypoint), and
# these env vars only take effect if set *before* torch/mediapipe load
# their native thread pools. See run.py for why this matters: without it,
# every camera's detector fights every other camera's detector for all
# CPU cores at once, which is the main cause of multi-camera lag.
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '2')

import cv2
import numpy as np
import time
import logging
import threading
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

cv2.setNumThreads(1)  # each camera already has its own thread; don't also
                       # let OpenCV fan resize/imencode work out to more
                       # threads per call -- with several cameras running
                       # that multiplies straight into oversubscription.

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Shared across every FallDetector instance (i.e. every camera). Caps how
# many cameras can be doing heavy pose/YOLO inference at the exact same
# instant. Without this, if N cameras' "run detection this frame" turns
# land on the same tick, all N fire at once and the CPU falls over even
# with the thread caps above. With it, extra cameras simply queue for a
# few milliseconds and take turns -- still real-time, just serialized
# instead of stacked on top of each other.
_MAX_CONCURRENT_INFERENCE = int(os.environ.get('FALLGUARD_MAX_CONCURRENT_INFERENCE', '2'))
_inference_gate = threading.Semaphore(_MAX_CONCURRENT_INFERENCE)

# ── Landmark indices ──────────────────────────────────────────
NOSE       = 0
L_SHOULDER = 11; R_SHOULDER = 12
L_HIP      = 23; R_HIP      = 24
L_KNEE     = 25; R_KNEE     = 26
L_ANKLE    = 27; R_ANKLE    = 28

# Vertical hip-to-knee gap (normalised by shoulder-to-hip torso length) below
# which the legs are considered "folded" -- i.e. seated -- rather than
# extended straight down as in standing. Used only for the display label
# (sitting vs standing); it plays no part in fall confirmation.
# Loosened from 0.55 -- that value under-caught sitting at typical desk/chair
# camera angles (knees not always fully level with hips), leaving people
# labelled "standing" while clearly seated.
SEATED_HIP_KNEE_RATIO = 0.72

POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(24,26),
    (25,27),(26,28),(27,29),(28,30),(29,31),(30,32)
]

HISTORY_LEN = 30

# ── Temporal fall-confirmation tuning ──────────────────────────
# A single bad frame must NOT be able to declare a fall. A fall is only
# confirmed once the person has been near-horizontal, at reasonably high
# confidence, AND roughly still for this many seconds in a row.
FALL_ANGLE_MAX         = 25     # body angle (deg from horizontal) below this = "lying-like"
                                 # (tightened from 35 -- a person leaning toward a screen
                                 # while seated is usually NOT this flat; only a genuinely
                                 # horizontal body should count)
FALL_CONFIRM_SECONDS   = 12     # must stay lying-like + still for this long to confirm
                                 # (no shortcuts -- every fall waits the full 12s, see
                                 # _evaluate_fall: the old "fast lane" for high-confidence
                                 # frames has been removed entirely)
FALL_STILLNESS_VELOCITY = 0.03  # below this = "not actively moving" (rules out a fast bend)
RECOVERY_ANGLE_MIN     = 45     # angle needed to cancel a candidate/confirmed fall (stood/sat back up)
ANKLE_VISIBILITY_MIN   = 0.5    # below this, ankles are considered occluded (e.g. hidden
                                 # under a desk) -- angle readings become unreliable, so we
                                 # don't let them count as fall evidence at all
POSE_LOST_TIMEOUT      = 20     # seconds of continuous tracking loss before we give up on a
                                 # confirmed fall and reset to "no person" (assume they left frame)

# ── Model — Full model for accuracy, Lite as fallback ─────────
MODEL_URLS = [
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
]


def _find_or_download_model() -> Optional[str]:
    """
    Look for model in project root (any name), then download if missing.
    Returns path to valid model file.
    """
    import glob, shutil

    # Search project root for any .task file
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    candidates = glob.glob(os.path.join(root, "*.task"))

    # Prefer full model (>50MB)
    for c in candidates:
        if os.path.getsize(c) > 50_000_000:
            logger.info("Using full model: %s (%.1f MB)", c, os.path.getsize(c)/1e6)
            return c

    # Accept any .task file
    if candidates:
        logger.info("Using model: %s", candidates[0])
        return candidates[0]

    # Also search Downloads/Desktop
    for search_root in [
        os.path.expanduser(os.path.join("~", "Downloads")),
        os.path.expanduser(os.path.join("~", "Desktop")),
    ]:
        for dirpath, _, files in os.walk(search_root):
            for f in files:
                if f.endswith('.task'):
                    src = os.path.join(dirpath, f)
                    dst = os.path.join(root, "pose_landmarker.task")
                    shutil.copy(src, dst)
                    logger.info("Found and copied model from %s", src)
                    return dst

    # Download lite model as last resort
    dst = os.path.join(root, "pose_landmarker.task")
    logger.info("Downloading pose model...")
    try:
        urllib.request.urlretrieve(MODEL_URLS[1], dst)
        logger.info("Model downloaded (%.1f MB)", os.path.getsize(dst)/1e6)
        return dst
    except Exception as e:
        logger.error("Model download failed: %s", e)
        return None


# ── Data classes ──────────────────────────────────────────────

@dataclass
class PersonState:
    center_history:   deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    height_history:   deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    angle_history:    deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    inactivity_start: float = 0.0
    is_inactive:      bool  = False
    prev_center:      Optional[float] = None
    prev_height:      Optional[float] = None
    # -- temporal fall confirmation (prevents single-frame flicker) --
    fall_candidate_start: Optional[float] = None
    fall_confirmed:       bool = False
    last_pose_seen_t:     float = 0.0
    seated_history:       deque = field(default_factory=lambda: deque(maxlen=5))


@dataclass
class DetectionResult:
    activity:         str   = 'unknown'
    is_fall:          bool  = False
    confidence:       float = 0.0
    conf_posture:     float = 0.0
    conf_velocity:    float = 0.0
    conf_height:      float = 0.0
    conf_inactivity:  float = 0.0
    conf_context:     float = 0.0
    body_angle:       float = 90.0
    velocity:         float = 0.0
    detected_objects: List[str] = field(default_factory=list)
    context_surface:  str   = 'floor'
    annotated_frame:  Optional[np.ndarray] = None
    persons:          List[dict] = field(default_factory=list)
    fps:              float = 0.0
    # aliases for Project 22 compatibility
    @property
    def is_ready(self): return True
    @property
    def error_message(self): return ""


class FallDetector:
    """
    Project 1's proven detection engine + Project 22 compatibility layer.
    Uses IMAGE mode (not VIDEO) — no timestamp issues, reliable posture detection.
    """

    def __init__(self, config: dict = None, socketio=None, enable_yolo: Optional[bool] = None):
        self.config    = config or {}
        self._socketio = socketio
        # None = follow the global detection_config.ENABLE_OBJECT_DETECTOR flag
        # (legacy/single-camera behavior). True/False = hard override for this
        # instance specifically -- used to turn YOLO off per multicam camera
        # without touching the global setting the single-camera page still uses.
        self._enable_yolo_override = enable_yolo

        self.confidence_threshold = self.config.get('FALL_CONFIDENCE_THRESHOLD',    0.85)
        self.inactivity_threshold = self.config.get('INACTIVITY_THRESHOLD_SECONDS', 5)
        self.velocity_threshold   = self.config.get('VELOCITY_THRESHOLD',           0.15)
        self.overlap_threshold    = self.config.get('OVERLAP_THRESHOLD',            0.5)

        self.w_posture    = self.config.get('WEIGHT_POSTURE',        0.30)
        self.w_velocity   = self.config.get('WEIGHT_VELOCITY',       0.25)
        self.w_height     = self.config.get('WEIGHT_HEIGHT_CHANGE',  0.20)
        self.w_inactivity = self.config.get('WEIGHT_INACTIVITY',     0.15)
        self.w_context    = self.config.get('WEIGHT_OBJECT_CONTEXT', 0.10)

        self._fps_tracker    = deque(maxlen=30)
        self._last_frame_t   = time.time()
        self._frame_count    = 0
        self._person_state   = PersonState()
        self._broadcast_width = 360
        self._last_result    = None
        self._ready          = False
        self._error_msg      = ""

        self._pose  = None
        self._yolo  = None
        self._bg    = cv2.createBackgroundSubtractorMOG2(
                          history=300, varThreshold=25, detectShadows=False)

        self._init_pose()
        self._init_yolo()

    # ── Init ─────────────────────────────────────────────────

    def _init_pose(self):
        model_path = _find_or_download_model()
        if not model_path:
            self._error_msg = (
                "Pose model not found. Place pose_landmarker.task in project root "
                "or run: python download_model.py"
            )
            logger.error(self._error_msg)
            if self._socketio:
                try: self._socketio.emit('detection_error', {'message': self._error_msg})
                except: pass
            return

        try:
            opts = mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=mp_vision.RunningMode.IMAGE,   # IMAGE mode — proven reliable
                num_poses=2,
                min_pose_detection_confidence=0.3,
                min_pose_presence_confidence=0.3,
                min_tracking_confidence=0.3,
                output_segmentation_masks=False,
            )
            self._pose  = mp_vision.PoseLandmarker.create_from_options(opts)
            self._ready = True
            logger.info("PoseLandmarker ready (IMAGE mode) model=%s", model_path)
        except Exception as e:
            self._error_msg = f"Pose init error: {e}"
            logger.error(self._error_msg)

    def _init_yolo(self):
        import app.services.detection.detection_config as dcfg
        if getattr(dcfg, 'ENABLE_OBJECT_DETECTOR', False):
            self._load_yolo()

    def _load_yolo(self):
        if self._yolo is not None:
            return True
        try:
            import torch
            torch.set_num_threads(2)  # see thread-cap note at top of file --
                                       # otherwise every camera's YOLO call
                                       # independently tries to use all cores
        except Exception:
            pass
        try:
            from ultralytics import YOLO
            self._yolo = YOLO('yolov8n.pt')
            logger.info("YOLOv8n loaded")
            return True
        except Exception:
            logger.info("YOLO unavailable — pose-only mode")
            return False

    # ── Properties (Project 22 compatibility) ─────────────────

    @property
    def fps(self) -> float:
        if not self._fps_tracker: return 0.0
        return round(sum(self._fps_tracker) / len(self._fps_tracker), 1)

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def error_message(self) -> str:
        return self._error_msg

    def update_config(self, new_cfg: dict):
        self.config.update(new_cfg)
        if 'FALL_CONFIDENCE_THRESHOLD' in new_cfg:
            self.confidence_threshold = float(new_cfg['FALL_CONFIDENCE_THRESHOLD'])

    def get_metrics(self) -> dict:
        return {'fps': self.fps, 'frame_count': self._frame_count}

    def mark_ground_truth_fall(self):
        pass  # stub for Project 22 compatibility

    # ── Main entry point ──────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        self._frame_count += 1
        now = time.time()
        dt  = now - self._last_frame_t
        self._last_frame_t = now
        if dt > 0:
            self._fps_tracker.append(1.0 / dt)

        h, w = frame.shape[:2]
        result = DetectionResult(fps=self.fps)

        # Serialize the heavy part (YOLO + pose inference) across every
        # camera's detector so they take turns on the CPU instead of all
        # landing on it at once. This is the main fix for multi-camera lag
        # -- see _inference_gate definition for why it's needed even with
        # the per-library thread caps in place.
        with _inference_gate:
            # 1. YOLO context detection
            person_box = None
            context_boxes: Dict[str, List] = {}
            import app.services.detection.detection_config as dcfg
            use_yolo = (dcfg.ENABLE_OBJECT_DETECTOR if self._enable_yolo_override is None
                        else self._enable_yolo_override)
            if use_yolo:
                if self._yolo is None:
                    self._load_yolo()
                if self._yolo:
                    person_box, context_boxes = self._run_yolo(frame)
                else:
                    person_box = self._detect_person_bg(frame)
            else:
                person_box = self._detect_person_bg(frame)

            # 2. Pose estimation (IMAGE mode)
            kp = None
            if getattr(dcfg, 'POSE_SKIP_FRAMES', False) and (self._frame_count % 2 == 0) and getattr(self, '_last_kp', None) is not None:
                kp = self._last_kp
            else:
                kp = self._run_pose(frame, w, h)
                self._last_kp = kp
        # (inference gate released here -- everything below is cheap
        # numpy/python scoring, no need to hold the CPU slot for it)
        if kp is None:
            ps = self._person_state
            if ps.fall_confirmed and (now - ps.last_pose_seen_t) < POSE_LOST_TIMEOUT:
                # We already confirmed a fall for this person and pose
                # tracking just momentarily lost them (very common while
                # someone is lying down -- it's a much rarer pose for the
                # model than standing). Losing tracking is NOT the same as
                # the person recovering, so keep reporting the fall instead
                # of silently flipping is_fall back to False. Without this,
                # every brief tracking dropout looked like "recovered, then
                # fell again" and re-triggered a brand new alert. Bounded by
                # POSE_LOST_TIMEOUT so a person who actually got up and
                # walked out of frame doesn't stay "fallen" forever.
                result.activity        = 'fallen'
                result.is_fall         = True
                result.confidence      = self._last_result.confidence if self._last_result else 0.6
                result.persons = [{
                    'track_id': 1, 'posture': 'fallen', 'status': 'fallen',
                    'fall_score': result.confidence, 'surface': 'unknown',
                    'is_resting': False, 'stage': 5,
                }]
                result.annotated_frame = self._annotate_empty(frame.copy())
                self._last_result = result
                return result
            if ps.fall_confirmed:
                # timed out -- give up and reset so a real re-entry starts clean
                ps.fall_confirmed = False
                ps.fall_candidate_start = None
            result.activity        = 'no_person'
            result.annotated_frame = self._annotate_empty(frame.copy())
            self._last_result = result
            return result

        # 3. Signals
        body_angle       = self._body_angle(kp)
        ankle_visibility = kp.get('ankle_visibility', 1.0)
        center_y         = kp['center'][1] / h
        body_height      = abs(kp['ankle_mid'][1] - kp['shoulder_mid'][1]) / max(h, 1)
        ps               = self._person_state
        ps.last_pose_seen_t = now
        velocity         = max(0.0, center_y - ps.prev_center) if ps.prev_center else 0.0

        ps.center_history.append(center_y)
        ps.height_history.append(body_height)
        ps.angle_history.append(body_angle)
        ps.prev_center = center_y
        ps.prev_height = body_height

        c_posture    = self._score_posture(body_angle)
        c_velocity   = self._score_velocity(velocity)
        c_height     = self._score_height(ps)
        c_inactivity = self._score_inactivity(velocity, now, ps)

        if person_box is None:
            xs = [v[0] for v in kp.values() if isinstance(v, tuple)]
            ys = [v[1] for v in kp.values() if isinstance(v, tuple)]
            person_box = (min(xs), min(ys), max(xs), max(ys)) if xs else (0,0,w,h)

        surface, c_context = self._score_context(person_box, context_boxes)

        conf = max(0.0, min(1.0,
            self.w_posture    * c_posture    +
            self.w_velocity   * c_velocity   +
            self.w_height     * c_height     +
            self.w_inactivity * c_inactivity +
            self.w_context    * c_context
        ))

        result.body_angle      = round(body_angle,  1)
        result.velocity        = round(velocity,    4)
        result.conf_posture    = round(c_posture,   3)
        result.conf_velocity   = round(c_velocity,  3)
        result.conf_height     = round(c_height,    3)
        result.conf_inactivity = round(c_inactivity,3)
        result.conf_context    = round(c_context,   3)
        result.confidence      = round(conf,        3)
        result.context_surface = surface
        result.detected_objects = list(context_boxes.keys())

        # Legs folded under the body (hip-knee gap small relative to torso
        # length) => seated, regardless of whether furniture was detected.
        # Smoothed over the last few frames (majority vote) so landmark
        # jitter doesn't flicker the label between "sitting" and "standing"
        # frame to frame while the person hasn't actually moved.
        torso_len = max(1.0, abs(kp['hip_mid'][1] - kp['shoulder_mid'][1]))
        hip_knee_gap = abs(kp['knee_mid'][1] - kp['hip_mid'][1])
        seated_now = (hip_knee_gap / torso_len) < SEATED_HIP_KNEE_RATIO
        ps.seated_history.append(seated_now)
        seated = sum(ps.seated_history) >= (len(ps.seated_history) / 2.0)

        activity = self._classify(body_angle, velocity, surface, conf, c_posture, seated)
        is_fall  = self._evaluate_fall(ps, now, body_angle, velocity, conf, surface, ankle_visibility)

        if is_fall:
            activity = 'fallen'
        elif ps.fall_candidate_start is not None:
            # still accumulating evidence -- not yet confirmed, but flag it
            # in the UI so a human can see something is being watched
            activity = 'possible_fall'

        result.activity = activity
        result.is_fall  = is_fall

        # Per-person data for Project 22 multi-person UI
        result.persons = [{
            'track_id':   1,
            'posture':    result.activity,
            'status':     result.activity,
            'fall_score': conf,
            'surface':    surface,
            'is_resting': surface in ('bed', 'couch', 'chair'),
            'stage':      5 if result.is_fall else (3 if activity == 'possible_fall' else 0),
        }]

        result.annotated_frame = self._annotate(
            frame.copy(), kp, result, person_box, context_boxes, w, h)

        self._last_result = result
        return result

    # ── Pose ─────────────────────────────────────────────────

    def _run_pose(self, frame, w, h):
        if self._pose is None:
            return None
        try:
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            det    = self._pose.detect(mp_img)
            if not det.pose_landmarks:
                return None
            lms = det.pose_landmarks[0]

            def px(i): return (int(lms[i].x * w), int(lms[i].y * h))

            ls, rs = px(L_SHOULDER), px(R_SHOULDER)
            lh, rh = px(L_HIP),      px(R_HIP)
            lk, rk = px(L_KNEE),     px(R_KNEE)
            la, ra = px(L_ANKLE),    px(R_ANKLE)
            return {
                'l_shoulder':  ls, 'r_shoulder': rs,
                'l_hip':       lh, 'r_hip':      rh,
                'l_knee':      lk, 'r_knee':     rk,
                'l_ankle':     la, 'r_ankle':    ra,
                'shoulder_mid':((ls[0]+rs[0])//2, (ls[1]+rs[1])//2),
                'hip_mid':     ((lh[0]+rh[0])//2, (lh[1]+rh[1])//2),
                'knee_mid':    ((lk[0]+rk[0])//2, (lk[1]+rk[1])//2),
                'ankle_mid':   ((la[0]+ra[0])//2, (la[1]+ra[1])//2),
                'center':      ((lh[0]+rh[0])//2, (lh[1]+rh[1])//2),
                'nose':        px(NOSE),
                'ankle_visibility': min(
                    getattr(lms[L_ANKLE], 'visibility', 1.0),
                    getattr(lms[R_ANKLE], 'visibility', 1.0)),
                '_landmarks':  lms,
            }
        except Exception as e:
            logger.debug("Pose error: %s", e)
            return None

    # ── YOLO ─────────────────────────────────────────────────

    def _run_yolo(self, frame):
        try:
            res = self._yolo(frame, verbose=False, conf=0.35)[0]
            person_box = None; best_conf = 0.0; ctx = {}
            for box in res.boxes:
                cls = int(box.cls[0]); cf = float(box.conf[0])
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                if cls == 0 and cf > best_conf:
                    best_conf = cf; person_box = (x1,y1,x2,y2)
                label = res.names.get(cls, str(cls))
                if label in ('bed','couch','chair'):
                    ctx.setdefault(label, []).append((x1,y1,x2,y2))
            return person_box, ctx
        except Exception as e:
            logger.debug("YOLO error: %s", e)
            return None, {}

    def _detect_person_bg(self, frame):
        fg = self._bg.apply(frame)
        k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
        cnts,_ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > 800:
                x,y,bw,bh = cv2.boundingRect(c)
                return (x,y,x+bw,y+bh)
        return None

    # ── Scoring ───────────────────────────────────────────────

    def _body_angle(self, kp):
        # Torso vector (shoulder -> hip), NOT shoulder -> ankle.
        #
        # Why: the old shoulder-to-ankle angle was the single biggest cause of
        # sitting/standing being flagged as a fall. When someone sits in a
        # chair, their ankles land forward of and below their shoulders (bent
        # knees), which drags that vector toward horizontal from a side-on
        # camera even though the person is fully upright. The torso itself
        # (shoulder to hip) stays near-vertical whether someone is standing,
        # sitting, or bending their knees -- it only goes horizontal when the
        # person's upper body is actually down, i.e. an actual fall. This is
        # unaffected by leg position or camera angle.
        sm, hm = kp['shoulder_mid'], kp['hip_mid']
        dx = hm[0]-sm[0]; dy = hm[1]-sm[1]
        return abs(np.degrees(np.arctan2(abs(dy), abs(dx)+1e-6)))

    def _score_posture(self, angle):
        if angle < 20: return 1.0
        if angle < 35: return 0.85
        if angle < 50: return 0.5
        if angle < 65: return 0.2
        return 0.0

    def _score_velocity(self, v):
        if v > 0.12: return 1.0
        if v > 0.07: return 0.7
        if v > 0.03: return 0.3
        return 0.0

    def _score_height(self, ps):
        if len(ps.height_history) < 10: return 0.0
        recent   = list(ps.height_history)
        baseline = np.mean(recent[:10])
        current  = np.mean(recent[-3:]) if len(recent) >= 3 else recent[-1]
        if baseline < 0.01: return 0.0
        drop = (baseline - current) / baseline
        if drop > 0.40: return 1.0
        if drop > 0.25: return 0.6
        if drop > 0.10: return 0.2
        return 0.0

    def _score_inactivity(self, velocity, now, ps):
        if velocity < 0.01:
            if not ps.is_inactive:
                ps.inactivity_start = now
                ps.is_inactive      = True
            return min(1.0, (now - ps.inactivity_start) / max(self.inactivity_threshold, 1))
        ps.is_inactive      = False
        ps.inactivity_start = 0.0
        return 0.0

    def _score_context(self, person_box, context_boxes):
        if not context_boxes:
            return 'floor', 0.5
        px1,py1,px2,py2 = person_box
        p_area = max(1, (px2-px1)*(py2-py1))
        # NOTE: 'chair' was previously missing here, so anyone sitting at a
        # desk/chair (laptop use, reading, etc.) was always scored as if they
        # were on the floor. This was the single biggest source of false
        # positives for desk/chair scenarios.
        for surface in ('bed', 'couch', 'chair'):
            for (ox1,oy1,ox2,oy2) in context_boxes.get(surface, []):
                ix1=max(px1,ox1); iy1=max(py1,oy1)
                ix2=min(px2,ox2); iy2=min(py2,oy2)
                if ix2 > ix1 and iy2 > iy1:
                    # chairs are smaller than the person, so a lower overlap
                    # ratio (relative to the chair, not the person) still counts
                    inter_area = (ix2-ix1)*(iy2-iy1)
                    chair_area = max(1, (ox2-ox1)*(oy2-oy1))
                    ratio = inter_area / p_area if surface != 'chair' else max(
                        inter_area / p_area, inter_area / chair_area)
                    if ratio >= self.overlap_threshold:
                        return surface, 0.0
        return 'floor', 0.8

    def _classify(self, angle, velocity, surface, conf, c_posture, seated=False):
        """Returns a human-readable activity label only. Whether an alert
        actually fires is decided separately by _evaluate_fall(), which
        requires the lying-like state to persist for FALL_CONFIRM_SECONDS —
        no single frame here can trigger an alert on its own."""
        if surface == 'bed':   return 'sleeping'
        if surface == 'couch': return 'lying_down'
        if surface == 'chair': return 'sitting'
        if angle > 65:
            if seated: return 'sitting'
            return 'walking' if velocity > 0.04 else 'standing'
        if angle > 45: return 'bending'
        if angle > 30: return 'sitting'
        if c_posture > 0.5 and velocity < 0.02: return 'lying'
        return 'falling' if conf > 0.6 else 'unknown'

    def _evaluate_fall(self, ps: 'PersonState', now: float, angle: float,
                        velocity: float, conf: float, surface: str,
                        ankle_visibility: float = 1.0) -> bool:
        """Gated, temporal fall confirmation. A fall is only ever confirmed
        once the person has been near-horizontal AND fairly still AND at
        reasonable confidence for FALL_CONFIRM_SECONDS in a row. This is
        what prevents a quick bend-to-pick-something-up, tying a shoe, or a
        one-frame pose glitch from ever firing an alert."""
        # Furniture is never a fall, regardless of angle/confidence, and
        # immediately cancels any in-progress candidate (e.g. sitting down).
        if surface in ('bed', 'couch', 'chair'):
            ps.fall_candidate_start = None
            ps.fall_confirmed = False
            return False

        # If the ankles are occluded (e.g. hidden under a desk while sitting),
        # the body angle is a guess and cannot be trusted as fall evidence.
        # Freeze the candidate timer rather than either advancing or clearing
        # it -- we simply don't have enough information this frame.
        if ankle_visibility < ANKLE_VISIBILITY_MIN:
            return ps.fall_confirmed

        lying_like = angle < FALL_ANGLE_MAX and conf >= 0.55

        if lying_like:
            if ps.fall_candidate_start is None:
                ps.fall_candidate_start = now
            elapsed = now - ps.fall_candidate_start
            motionless = velocity < FALL_STILLNESS_VELOCITY
            # No shortcuts: every fall must sit lying-like + still for the
            # FULL confirm window, no matter how high confidence is.
            if elapsed >= FALL_CONFIRM_SECONDS and motionless:
                ps.fall_confirmed = True
        elif angle >= RECOVERY_ANGLE_MIN:
            # person stood/sat back up -> cancel any candidate or confirmed fall
            ps.fall_candidate_start = None
            ps.fall_confirmed = False
        # else: ambiguous middle zone (bending range) -> leave state as-is,
        # don't cancel a genuine fall-in-progress but don't start one either

        return ps.fall_confirmed

    # ── Annotation ────────────────────────────────────────────

    def _annotate(self, frame, kp, result, person_box, context_boxes, w, h):
        lms = kp.get('_landmarks')
        if lms:
            for a, b in POSE_CONNECTIONS:
                try:
                    pa = (int(lms[a].x*w), int(lms[a].y*h))
                    pb = (int(lms[b].x*w), int(lms[b].y*h))
                    cv2.line(frame, pa, pb, (0,200,255), 2, cv2.LINE_AA)
                except: pass
            for lm in lms:
                try:
                    cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 4, (255,255,255), -1)
                except: pass

        if person_box:
            color = (0,0,255) if result.is_fall else (0,255,100)
            cv2.rectangle(frame, person_box[:2], person_box[2:], color, 2)

        for label, boxes in context_boxes.items():
            for box in boxes:
                cv2.rectangle(frame, box[:2], box[2:], (255,200,0), 1)
                cv2.putText(frame, label, (box[0], box[1]-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,200,0), 1)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (260,180), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        color = (0,0,255) if result.is_fall else (0,255,150)
        for i, (text, col) in enumerate([
            (f"Activity : {result.activity.upper()}", color),
            (f"Confidence: {result.confidence:.0%}",  (200,200,200)),
            (f"Angle    : {result.body_angle:.1f} deg",(200,200,200)),
            (f"Velocity : {result.velocity:.3f}",      (200,200,200)),
            (f"Surface  : {result.context_surface}",   (200,200,200)),
            (f"FPS      : {self.fps:.1f}",              (150,150,150)),
        ]):
            cv2.putText(frame, text, (8, 22+i*26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)

        if result.is_fall:
            cv2.rectangle(frame, (0,0), (w,h), (0,0,255), 4)
            cv2.putText(frame, "!! FALL DETECTED !!",
                        (w//2-140, h-20), cv2.FONT_HERSHEY_DUPLEX,
                        1.0, (0,0,255), 2, cv2.LINE_AA)
        return frame

    def _annotate_empty(self, frame):
        cv2.putText(frame, "No person detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,100,100), 1)
        return frame

    def release(self):
        if self._pose:
            self._pose.close()