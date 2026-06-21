

import cv2, numpy as np, sys, argparse
from pathlib import Path


K = np.array([
    [595.8036047830891,  0.0,               924.65430795264774],
    [0.0,               598.33827405037948, 580.41919770485049],
    [0.0,               0.0,               1.0]
], dtype=np.float64)
D = np.array([
    -0.015349419086740696, -0.053676477252104900,
     0.061315407683887907, -0.026142516909791854
], dtype=np.float64)
FOV_SCALE = 0.75

def undistort_frame(frame):
    h, w = frame.shape[:2]
    Knew = K.copy()
    Knew[0, 0] *= FOV_SCALE
    Knew[1, 1] *= FOV_SCALE
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), Knew, (w, h), cv2.CV_16SC2)
    return cv2.remap(frame, m1, m2, cv2.INTER_LINEAR)

def find_positive_video():
    """Search for a positive video automatically."""
    for path in sorted(Path("../input").rglob("positive_1.mp4")):
        return str(path)
    for path in sorted(Path("../input").rglob("positive_*.mp4")):
        return str(path)
    return None
def main():
    # Keep the argparse parser so you can change frames/videos via CLI if needed
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None)
    parser.add_argument("--frame", type=int, default=50)
    args = parser.parse_args()

    video_path = args.video or find_positive_video()
    if not video_path:
        sys.exit("[ERROR] No positive video found. Pass --video path/to/video.mp4")

    print(f"[INFO] Opening {video_path} headlessly, extracting frame {args.frame}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        sys.exit(f"[ERROR] Cannot read frame {args.frame}")

    # Undistort and save immediately
    undist = undistort_frame(frame)
    Path("debug").mkdir(exist_ok=True)
    
    output_path = "debug/find_coordinates_here1.png"
    cv2.imwrite(output_path, undist)
    
    print(f"\n[SUCCESS] Image saved to: {output_path}")
    print("Download this image to your local machine, open it in an editor to find ")
    print("the (X, Y) pixel positions for the 4 corners, and fill in config.hpp manually.")
if __name__ == "__main__":
    main()
