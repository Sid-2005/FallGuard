"""
FallGuard AI — Single command startup
Run: python run.py
Auto-installs all dependencies, finds/downloads model, starts server.
"""

import os, sys, subprocess

# ── CPU thread-limiting (must happen before torch/mediapipe/cv2 are ever
# imported anywhere in the process) ─────────────────────────────────────
# PyTorch (used by YOLO) and MediaPipe/TFLite each default to grabbing
# *every* CPU core for a single inference call. That's fine with one
# camera. With several multi-camera threads each doing this at once, they
# all fight over the same cores simultaneously -- CPU context-switching
# overhead explodes and every camera's frame processing crawls, which is
# what actually causes "multicamera lags so much nothing is detected."
# Capping each library to a couple of threads lets several cameras run
# concurrently without starving each other.
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '2')

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def _find_model():
    import shutil

    dst = os.path.join(ROOT, 'pose_landmarker.task')

    # Check if model already exists in project directory
    if os.path.exists(dst):

        size = os.path.getsize(dst) / 1e6

        # MediaPipe Pose Landmarker Full model is approximately 9 MB
        if size >= 5:
            print(f"  [OK] Pose Landmarker Full model found ({size:.1f} MB)")
            return

        else:
            print(
                f"  ⚠ Model file appears too small "
                f"({size:.1f} MB)."
            )
            print("  Please replace it with the Full model.")

    # Search common download locations
    print("  Searching for pose_landmarker.task...")

    search_dirs = [
        os.path.expanduser(os.path.join('~', 'Downloads')),
        os.path.expanduser(os.path.join('~', 'Desktop')),
        os.path.expanduser(os.path.join('~', 'Documents')),
    ]

    for root_dir in search_dirs:

        if not os.path.exists(root_dir):
            continue

        for dirpath, _, files in os.walk(root_dir):

            if 'pose_landmarker.task' in files:

                src = os.path.join(
                    dirpath,
                    'pose_landmarker.task'
                )

                size = os.path.getsize(src) / 1e6

                # Accept Full model
                if size >= 5:

                    shutil.copy(src, dst)

                    print(
                        f"  [OK] Full model found and copied "
                        f"({size:.1f} MB)"
                    )

                    return

    # Model not found
    print("  [Error] Full Pose Landmarker model not found.")

    print(
        "  Download the Full model from:"
    )

    print(
        "  https://storage.googleapis.com/"
        "mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/"
        "pose_landmarker_full.task"
    )

    print(
        "  Rename to: pose_landmarker.task"
    )

    print(
        f"  Place in: {ROOT}"
    )

def _yolo_object():
    # yolov8n.pt is the general-purpose object detector used for furniture
    # context (bed/couch/chair) — see FallDetector._run_yolo / _score_context.
    # Pose estimation itself is done by the MediaPipe model (pose_landmarker.task),
    # not YOLO — this is a separate, smaller model just for furniture detection.
    import shutil
    dst = os.path.join(ROOT, 'yolov8n.pt')
    if os.path.exists(dst): return
    try:
        from ultralytics import YOLO
        print("  Downloading YOLOv8n object detector (~6 MB)...")
        YOLO('yolov8n.pt')
        if os.path.exists('yolov8n.pt'):
            shutil.move('yolov8n.pt', dst)
        print("  [OK] YOLOv8n ready")
    except Exception as e:
        print(f"  YOLOv8n object detector skipped ({e}) — will run in pose-only mode")


print("\n" + "="*55)
print("  FallGuard AI — Starting up")
print("="*55)
print("\n  [1/3] Checking dependencies...")
print("  [2/3] Checking model...")
_find_model()
print("  [3/3] Preparing YOLOv8...")
_yolo_object()


import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)

from app import create_app, socketio


def main():
    env = os.getenv('FLASK_ENV', 'development')
    app = create_app(env)

    with app.app_context():
        from app.services.fall_detector  import FallDetector
        from app.services.camera_service import CameraService

        det_config = {
            'FALL_CONFIDENCE_THRESHOLD':    app.config.get('FALL_CONFIDENCE_THRESHOLD', 0.85),
            'ALERT_COOLDOWN_SECONDS':       app.config.get('ALERT_COOLDOWN_SECONDS',    30),
            'INACTIVITY_THRESHOLD_SECONDS': app.config.get('INACTIVITY_THRESHOLD_SECONDS', 5),
            'VELOCITY_THRESHOLD':           app.config.get('VELOCITY_THRESHOLD',        0.15),
            'OVERLAP_THRESHOLD':            app.config.get('OVERLAP_THRESHOLD',         0.5),
        }

        detector = FallDetector(config=det_config, socketio=socketio)

        cam_config = {
            'ALERT_COOLDOWN_SECONDS': app.config.get('ALERT_COOLDOWN_SECONDS', 30),
            'EVENT_IMAGE_DIR':        app.config.get('EVENT_IMAGE_DIR', 'static/events'),
            'CLIP_DIR':               'static/clips',
            'ward_name':              '',
            'camera_index':           0,
        }
        camera_service = CameraService(socketio, detector, cam_config, app=app)

        app.config['_camera_service'] = camera_service
        app.config['_detector']       = detector

        # ── Multi-camera (phones/room cameras) ──────────────────
        from app.services.multi_camera_manager import MultiCameraManager
        from app.models.camera_node import CameraNode

        multi_cam_config = dict(cam_config)  # same weights/thresholds, separate instance
        mgr = MultiCameraManager(socketio, app, multi_cam_config)
        app.config['_multi_camera_manager'] = mgr

        try:
            local_nodes = CameraNode.query.filter_by(is_local=True, enabled=True).all()
            mgr.start_all_enabled(local_nodes)
        except Exception as e:
            logging.getLogger(__name__).warning(
                "Could not auto-start local cameras (DB not migrated yet?): %s", e)

    host = os.getenv('HOST',  '0.0.0.0')
    port = int(os.getenv('PORT', 5000))

    print(f"\n{'='*55}")
    print(f"  FallGuard AI  ->  http://localhost:{port}")
    print(f"  Network       ->  http://<your-ip>:{port}")
    print(f"  Login         ->  admin / admin123")
    print(f"{'='*55}\n")

    socketio.run(app, host=host, port=port,
                 debug=False, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main()