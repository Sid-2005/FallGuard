"""
FallGuard AI - Fall Detector (YOLOv8-Pose + ST-GCN++)
"""

import os
import cv2
import numpy as np
import time
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import torch

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '2')

cv2.setNumThreads(1)

logger = logging.getLogger(__name__)

_MAX_CONCURRENT_INFERENCE = int(os.environ.get('FALLGUARD_MAX_CONCURRENT_INFERENCE', '2'))
_inference_gate = threading.Semaphore(_MAX_CONCURRENT_INFERENCE)

from ultralytics import YOLO
from app.services.detection.stgcn_classifier import STGCNClassifier

# COCO Keypoint Indices (YOLOv8-Pose)
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

SEATED_HIP_KNEE_RATIO = 0.72

POSE_CONNECTIONS = [
    (NOSE,1),(NOSE,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),(12,14),
    (13,15),(14,16)
]

HISTORY_LEN = 30
FALL_ANGLE_MAX = 25
FALL_CONFIRM_SECONDS = 3  # Changed by me Siddu from 12 sec(3 worked)
FALL_STILLNESS_VELOCITY = 0.03
RECOVERY_ANGLE_MIN = 45
ANKLE_VISIBILITY_MIN = 0.5
POSE_LOST_TIMEOUT = 20

@dataclass
class PersonState:
    center_history: deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    height_history: deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    angle_history: deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    kp_buffer: deque = field(default_factory=lambda: deque(maxlen=30))
    
    inactivity_start: float = 0.0
    is_inactive: bool = False
    prev_center: Optional[float] = None
    prev_height: Optional[float] = None
    
    fall_candidate_start: Optional[float] = None
    fall_confirmed: bool = False
    last_pose_seen_t: float = 0.0
    seated_history: deque = field(default_factory=lambda: deque(maxlen=5))

@dataclass
class DetectionResult:
    activity: str = 'unknown'
    is_fall: bool = False
    confidence: float = 0.0
    conf_posture: float = 0.0
    conf_velocity: float = 0.0
    conf_height: float = 0.0
    conf_inactivity: float = 0.0
    conf_context: float = 0.0
    body_angle: float = 90.0
    velocity: float = 0.0
    detected_objects: List[str] = field(default_factory=list)
    context_surface: str = 'floor'
    annotated_frame: Optional[np.ndarray] = None
    persons: List[dict] = field(default_factory=list)
    fps: float = 0.0
    
    @property
    def is_ready(self): return True
    @property
    def error_message(self): return ""

class FallDetector:
    def __init__(self, config: dict = None, socketio=None, enable_yolo: Optional[bool] = None):
        self.config = config or {}
        self._socketio = socketio
        self._enable_yolo_override = enable_yolo

        self.confidence_threshold = self.config.get('FALL_CONFIDENCE_THRESHOLD', 0.85)
        self.inactivity_threshold = self.config.get('INACTIVITY_THRESHOLD_SECONDS', 5)
        self.velocity_threshold = self.config.get('VELOCITY_THRESHOLD', 0.15)
        self.overlap_threshold = self.config.get('OVERLAP_THRESHOLD', 0.5)

        self.w_posture = self.config.get('WEIGHT_POSTURE', 0.30)
        self.w_velocity = self.config.get('WEIGHT_VELOCITY', 0.25)
        self.w_height = self.config.get('WEIGHT_HEIGHT_CHANGE', 0.20)
        self.w_inactivity = self.config.get('WEIGHT_INACTIVITY', 0.15)
        self.w_context = self.config.get('WEIGHT_OBJECT_CONTEXT', 0.10)

        self._fps_tracker = deque(maxlen=30)
        self._last_frame_t = time.time()
        self._frame_count = 0
        self._person_state = PersonState()
        self._broadcast_width = 360
        self._last_result = None
        self._ready = False
        self._error_msg = ""
        
        self._yolo = None
        self._pose_model = None
        self._bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)
        self._stgcn = STGCNClassifier()

        self._init_models()

    def _init_models(self):
        try:
            torch.set_num_threads(2)
        except Exception:
            pass
        
        try:
            self._pose_model = YOLO('yolov8n-pose.pt')
            self._ready = True
            logger.info("YOLOv8-Pose loaded for posture.")
        except Exception as e:
            self._error_msg = f"Pose model load error: {e}"
            logger.error(self._error_msg)

        import app.services.detection.detection_config as dcfg
        use_yolo = (dcfg.ENABLE_OBJECT_DETECTOR if self._enable_yolo_override is None else self._enable_yolo_override)
        if use_yolo:
            try:
                self._yolo = YOLO('yolov8n.pt')
                logger.info("YOLOv8n loaded for context.")
            except Exception as e:
                logger.info(f"YOLOv8n unavailable for context: {e}")

    @property
    def fps(self) -> float:
        if not self._fps_tracker: return 0.0
        return round(sum(self._fps_tracker) / len(self._fps_tracker), 1)

    @property
    def is_ready(self) -> bool: return self._ready
    @property
    def error_message(self) -> str: return self._error_msg

    def update_config(self, new_cfg: dict):
        self.config.update(new_cfg)
        if 'FALL_CONFIDENCE_THRESHOLD' in new_cfg:
            self.confidence_threshold = float(new_cfg['FALL_CONFIDENCE_THRESHOLD'])

    def get_metrics(self) -> dict:
        return {'fps': self.fps, 'frame_count': self._frame_count}

    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        self._frame_count += 1
        now = time.time()
        dt = now - self._last_frame_t
        self._last_frame_t = now
        if dt > 0: self._fps_tracker.append(1.0 / dt)

        h, w = frame.shape[:2]
        result = DetectionResult(fps=self.fps)

        with _inference_gate:
            person_box = None
            context_boxes: Dict[str, List] = {}
            if self._yolo:
                person_box, context_boxes = self._run_yolo(frame)
            else:
                person_box = self._detect_person_bg(frame)

            kp = self._run_pose(frame)
            
        if not kp:
            ps = self._person_state
            if ps.fall_confirmed and (now - ps.last_pose_seen_t) < POSE_LOST_TIMEOUT:
                result.activity = 'fallen'
                result.is_fall = True
                result.confidence = self._last_result.confidence if self._last_result else 0.6
                result.persons = [{'track_id': 1, 'posture': 'fallen', 'status': 'fallen', 'fall_score': result.confidence, 'surface': 'unknown', 'is_resting': False, 'stage': 5}]
                result.annotated_frame = self._annotate_empty(frame.copy())
                self._last_result = result
                return result
            if ps.fall_confirmed:
                ps.fall_confirmed = False
                ps.fall_candidate_start = None
            result.activity = 'no_person'
            result.annotated_frame = self._annotate_empty(frame.copy())
            self._last_result = result
            return result

        body_angle = self._body_angle(kp)
        ankle_visibility = kp.get('ankle_visibility', 1.0)
        center_y = kp['center'][1] / h
        body_height = abs(kp['ankle_mid'][1] - kp['shoulder_mid'][1]) / max(h, 1)
        ps = self._person_state
        ps.last_pose_seen_t = now
        velocity = max(0.0, center_y - ps.prev_center) if ps.prev_center else 0.0

        ps.center_history.append(center_y)
        ps.height_history.append(body_height)
        ps.angle_history.append(body_angle)
        ps.kp_buffer.append(kp)
        ps.prev_center = center_y
        ps.prev_height = body_height

        c_posture = self._score_posture(body_angle)
        c_velocity = self._score_velocity(velocity)
        c_height = self._score_height(ps)
        c_inactivity = self._score_inactivity(velocity, now, ps)

        if not person_box:
            xs = [v[0] for k, v in kp.items() if isinstance(v, tuple) and 'mid' not in k and k != '_landmarks']
            ys = [v[1] for k, v in kp.items() if isinstance(v, tuple) and 'mid' not in k and k != '_landmarks']
            person_box = (min(xs), min(ys), max(xs), max(ys)) if xs else (0,0,w,h)

        surface, c_context = self._score_context(person_box, context_boxes)

        conf = max(0.0, min(1.0, self.w_posture * c_posture + self.w_velocity * c_velocity + self.w_height * c_height + self.w_inactivity * c_inactivity + self.w_context * c_context))

        # Kinematic Heuristic Check for ST-GCN
        if velocity > self.velocity_threshold:
            # High acceleration detected, trigger ST-GCN buffer processing
            stgcn_prob = self._stgcn.predict(list(ps.kp_buffer))
            if stgcn_prob > 0.5:
                conf = max(conf, stgcn_prob)

        result.body_angle = round(body_angle, 1)
        result.velocity = round(velocity, 4)
        result.conf_posture = round(c_posture, 3)
        result.conf_velocity = round(c_velocity, 3)
        result.conf_height = round(c_height, 3)
        result.conf_inactivity = round(c_inactivity, 3)
        result.conf_context = round(c_context, 3)
        result.confidence = round(conf, 3)
        result.context_surface = surface
        result.detected_objects = list(context_boxes.keys())

        torso_len = max(1.0, abs(kp['hip_mid'][1] - kp['shoulder_mid'][1]))
        hip_knee_gap = abs(kp['knee_mid'][1] - kp['hip_mid'][1])
        seated_now = (hip_knee_gap / torso_len) < SEATED_HIP_KNEE_RATIO
        ps.seated_history.append(seated_now)
        seated = sum(ps.seated_history) >= (len(ps.seated_history) / 2.0)

        activity = self._classify(body_angle, velocity, surface, conf, c_posture, seated)
        is_fall = self._evaluate_fall(ps, now, body_angle, velocity, conf, surface, ankle_visibility)

        if is_fall:
            activity = 'fallen'
        elif ps.fall_candidate_start is not None:
            activity = 'possible_fall'

        result.activity = activity
        result.is_fall = is_fall
        result.persons = [{'track_id': 1, 'posture': result.activity, 'status': result.activity, 'fall_score': conf, 'surface': surface, 'is_resting': surface in ('bed', 'couch', 'chair'), 'stage': 5 if result.is_fall else (3 if activity == 'possible_fall' else 0)}]
        result.annotated_frame = self._annotate(frame.copy(), kp, result, person_box, context_boxes, w, h)
        self._last_result = result
        return result

    def _run_pose(self, frame):
        if not self._pose_model: return None
        try:
            res = self._pose_model(frame, verbose=False, conf=0.3)[0]
            if not len(res.boxes): return None
            # Get keypoints for the most confident person
            best_idx = res.boxes.conf.argmax().item()
            kps = res.keypoints.data[best_idx].cpu().numpy() # shape (17, 3) [x, y, conf]
            
            def px(i): return (int(kps[i, 0]), int(kps[i, 1]))
            
            ls, rs = px(L_SHOULDER), px(R_SHOULDER)
            lh, rh = px(L_HIP), px(R_HIP)
            lk, rk = px(L_KNEE), px(R_KNEE)
            la, ra = px(L_ANKLE), px(R_ANKLE)
            
            return {
                'l_shoulder': ls, 'r_shoulder': rs,
                'l_hip': lh, 'r_hip': rh,
                'l_knee': lk, 'r_knee': rk,
                'l_ankle': la, 'r_ankle': ra,
                'shoulder_mid': ((ls[0]+rs[0])//2, (ls[1]+rs[1])//2),
                'hip_mid': ((lh[0]+rh[0])//2, (lh[1]+rh[1])//2),
                'knee_mid': ((lk[0]+rk[0])//2, (lk[1]+rk[1])//2),
                'ankle_mid': ((la[0]+ra[0])//2, (la[1]+ra[1])//2),
                'center': ((lh[0]+rh[0])//2, (lh[1]+rh[1])//2),
                'nose': px(NOSE),
                'ankle_visibility': min(kps[L_ANKLE, 2], kps[R_ANKLE, 2]),
                '_landmarks': kps
            }
        except Exception as e:
            logger.debug(f"Pose error: {e}")
            return None

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
        except Exception:
            return None, {}

    def _detect_person_bg(self, frame):
        fg = self._bg.apply(frame)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
        cnts,_ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > 800:
                x,y,bw,bh = cv2.boundingRect(c)
                return (x,y,x+bw,y+bh)
        return None

    def _body_angle(self, kp):
        sm, hm = kp['shoulder_mid'], kp['hip_mid']
        dx, dy = hm[0]-sm[0], hm[1]-sm[1]
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
        recent = list(ps.height_history)
        baseline = np.mean(recent[:10])
        current = np.mean(recent[-3:]) if len(recent) >= 3 else recent[-1]
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
                ps.is_inactive = True
            return min(1.0, (now - ps.inactivity_start) / max(self.inactivity_threshold, 1))
        ps.is_inactive = False
        ps.inactivity_start = 0.0
        return 0.0

    def _score_context(self, person_box, context_boxes):
        if not context_boxes: return 'floor', 0.5
        px1,py1,px2,py2 = person_box
        p_area = max(1, (px2-px1)*(py2-py1))
        for surface in ('bed', 'couch', 'chair'):
            for (ox1,oy1,ox2,oy2) in context_boxes.get(surface, []):
                ix1, iy1 = max(px1,ox1), max(py1,oy1)
                ix2, iy2 = min(px2,ox2), min(py2,oy2)
                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2-ix1)*(iy2-iy1)
                    chair_area = max(1, (ox2-ox1)*(oy2-oy1))
                    ratio = inter_area / p_area if surface != 'chair' else max(inter_area / p_area, inter_area / chair_area)
                    if ratio >= self.overlap_threshold: return surface, 0.0
        return 'floor', 0.8

    def _classify(self, angle, velocity, surface, conf, c_posture, seated=False):
        if surface == 'bed': return 'sleeping'
        if surface == 'couch': return 'lying_down'
        if surface == 'chair': return 'sitting'
        if angle > 65:
            if seated: return 'sitting'
            return 'walking' if velocity > 0.04 else 'standing'
        if angle > 45: return 'bending'
        if angle > 30: return 'sitting'
        if c_posture > 0.5 and velocity < 0.02: return 'lying'
        return 'falling' if conf > 0.6 else 'unknown'

    def _evaluate_fall(self, ps, now, angle, velocity, conf, surface, ankle_visibility=1.0):
        if surface in ('bed', 'couch', 'chair'):
            ps.fall_candidate_start = None
            ps.fall_confirmed = False
            return False
        if ankle_visibility < ANKLE_VISIBILITY_MIN: return ps.fall_confirmed
        lying_like = angle < FALL_ANGLE_MAX and conf >= 0.55
        if lying_like:
            if ps.fall_candidate_start is None: ps.fall_candidate_start = now
            if now - ps.fall_candidate_start >= FALL_CONFIRM_SECONDS and velocity < FALL_STILLNESS_VELOCITY:
                ps.fall_confirmed = True
        elif angle >= RECOVERY_ANGLE_MIN:
            ps.fall_candidate_start = None
            ps.fall_confirmed = False
        return ps.fall_confirmed

    def _annotate(self, frame, kp, result, person_box, context_boxes, w, h):
        kps = kp.get('_landmarks')
        if kps is not None:
            for a, b in POSE_CONNECTIONS:
                try:
                    if kps[a, 2] > 0.3 and kps[b, 2] > 0.3:
                        pa = (int(kps[a, 0]), int(kps[a, 1]))
                        pb = (int(kps[b, 0]), int(kps[b, 1]))
                        cv2.line(frame, pa, pb, (0,200,255), 2, cv2.LINE_AA)
                except: pass
            for lm in kps:
                try:
                    if lm[2] > 0.3:
                        cv2.circle(frame, (int(lm[0]), int(lm[1])), 4, (255,255,255), -1)
                except: pass

        if person_box:
            color = (0,0,255) if result.is_fall else (0,255,100)
            cv2.rectangle(frame, person_box[:2], person_box[2:], color, 2)

        for label, boxes in context_boxes.items():
            for box in boxes:
                cv2.rectangle(frame, box[:2], box[2:], (255,200,0), 1)
                cv2.putText(frame, label, (box[0], box[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,200,0), 1)

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
            cv2.putText(frame, text, (8, 22+i*26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)

        if result.is_fall:
            cv2.rectangle(frame, (0,0), (w,h), (0,0,255), 4)
            cv2.putText(frame, "!! FALL DETECTED !!", (w//2-140, h-20), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
        return frame

    def _annotate_empty(self, frame):
        cv2.putText(frame, "No person detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,100,100), 1)
        return frame

    def release(self):
        pass