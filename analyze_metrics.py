import os
import cv2
import sys
import numpy as np
from collections import deque

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.services.fall_detector import FallDetector
from app.services.camera_service import _resize_frame

DATASET_ROOT = r"D:\Major Project (Git)\11 Sept 2026\New Dataset"

def analyze_video(dataset, camera, detector):
    video_path = os.path.join(DATASET_ROOT, dataset, f"{camera}.avi")
    if not os.path.exists(video_path): return
    
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    max_drop = 0
    max_ar = 0
    
    print(f"\nAnalyzing {dataset} - {camera}")
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        proc = _resize_frame(frame, max_width=480)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        detector.process_frame(proc, is_file=True, video_fps=video_fps)
        ps = detector._person_state
        
        if len(ps.center_history) >= 10:
            current_center = ps.center_history[-1]
            center_10_ago = ps.center_history[-10]
            drop = current_center - center_10_ago
            
            aspect_ratio = 0.5
            if ps.kp_buffer:
                kp = ps.kp_buffer[-1]
                if 'shoulder_mid' in kp:
                    xs = [v[0] for k, v in kp.items() if isinstance(v, tuple) and 'mid' not in k and k != '_landmarks']
                    ys = [v[1] for k, v in kp.items() if isinstance(v, tuple) and 'mid' not in k and k != '_landmarks']
                    if xs and max(ys) > min(ys):
                        aspect_ratio = (max(xs) - min(xs)) / (max(ys) - min(ys))
            
            if drop > max_drop: max_drop = drop
            if aspect_ratio > max_ar: max_ar = aspect_ratio
            
            # Print frames where they look like they fell
            if aspect_ratio > 1.2 and ps.center_history[-1] > 0.3:
                angle = ps.angle_history[-1] if ps.angle_history else 90
                print(f"  Frame {frame_count:4d}: Drop/10f: {drop:.3f} | AR: {aspect_ratio:.2f} | Angle: {angle:.1f} | Center: {current_center:.2f}")

    print(f"Max Drop/10f: {max_drop:.3f} | Max AR: {max_ar:.2f}")
    cap.release()

if __name__ == "__main__":
    det = FallDetector()
    analyze_video('chute01', 'cam7', det)
    analyze_video('chute23', 'cam7', det)
    analyze_video('chute24', 'cam7', det)
