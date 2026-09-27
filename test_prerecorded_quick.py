import os
import cv2
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.services.fall_detector import FallDetector
from app.services.camera_service import _resize_frame

DATASET_ROOT = r"D:\Major Project (Git)\11 Sept 2026\New Dataset"

def test_video(dataset, camera, expected):
    video_path = os.path.join(DATASET_ROOT, dataset, f"{camera}.avi")
    if not os.path.exists(video_path): return f"File missing: {video_path}"
    
    cap = cv2.VideoCapture(video_path)
    detector = FallDetector()
    fall_detected_frame = -1
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        proc = _resize_frame(frame, max_width=480)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        result = detector.process_frame(proc, is_file=True, video_fps=video_fps)
        if result.is_fall and fall_detected_frame == -1:
            fall_detected_frame = frame_count
            break
        if frame_count % 1000 == 0:
            print(f"Processed {frame_count} frames of {dataset} {camera}...")
            
    cap.release()
    actual = "FALL DETECTED" if fall_detected_frame != -1 else "NO FALL"
    status = "✅ Pass" if actual == expected else "❌ Fail"
    return f"{dataset} | {camera} | {expected} | {actual} | {status} | Frame {fall_detected_frame if fall_detected_frame != -1 else '-'}"

if __name__ == "__main__":
    print(test_video('chute23', 'cam7', "NO FALL"))
    print(test_video('chute01', 'cam7', "FALL DETECTED"))
