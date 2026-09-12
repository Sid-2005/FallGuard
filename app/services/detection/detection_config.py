"""
app/services/detection/detection_config.py

Runtime feature flags for the live detection pipeline (app/services/fall_detector.py).
These are toggled at runtime from Settings > Detection Pipeline and read on
every frame, so keep this module tiny and dependency-free.
"""

# Person/furniture context via YOLOv8n. When ON, a detected bed/couch/chair
# under the person suppresses fall alerts entirely (see FallDetector._score_context).
# Costs ~25-40ms/frame on CPU. If OFF, sitting/lying on furniture is judged by
# pose geometry alone, which is still guarded by the temporal confirmation window.
ENABLE_OBJECT_DETECTOR = True

# Skip pose inference on every other frame and reuse the last result.
# Roughly doubles throughput on CPU at the cost of slightly slower posture
# updates. Leave OFF unless you need the extra frame rate.
POSE_SKIP_FRAMES = False

# Kept only so the Settings UI toggle doesn't error — the standalone floor
# mapper module was removed as dead code (it was never wired into the live
# pipeline; furniture/floor context already comes from ENABLE_OBJECT_DETECTOR).
ENABLE_FLOOR_MAPPER = False
