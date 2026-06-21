

import cv2
import numpy as np
import argparse
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

CANDIDATE_ROI = np.array([
    [620,  185],
    [1280, 185],
    [1300, 330],
    [600,  330],
], dtype=np.int32)


def undistort(frame):
    h, w = frame.shape[:2]
    Knew = K.copy()
    Knew[0, 0] *= FOV_SCALE
    Knew[1, 1] *= FOV_SCALE
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), Knew, (w, h), cv2.CV_16SC2)
    return cv2.remap(frame, m1, m2, cv2.INTER_LINEAR)


def draw_grid(img, step=100, minor=50):
    """Draw grid LINES ONLY (no fill). Major lines every `step`, minor every `minor`."""
    h, w = img.shape[:2]
    overlay = img.copy()

    # Minor lines — dim
    for x in range(0, w, minor):
        cv2.line(overlay, (x, 0), (x, h), (0, 90, 0), 1)
    for y in range(0, h, minor):
        cv2.line(overlay, (0, y), (w, y), (0, 90, 0), 1)

    # Major lines — bright green
    for x in range(0, w, step):
        cv2.line(overlay, (x, 0), (x, h), (0, 220, 0), 1)
    for y in range(0, h, step):
        cv2.line(overlay, (0, y), (w, y), (0, 220, 0), 1)

    # Blend so grid is semi-transparent and image stays visible
    out = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)

    # Labels (drawn solid on top, with dark outline for readability)
    for x in range(0, w, step):
        _label(out, str(x), (x + 3, 22))
    for y in range(0, h, step):
        _label(out, str(y), (3, y + 18))

    return out


def _label(img, text, org):
    # dark outline then bright text
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0), 1, cv2.LINE_AA)


def draw_candidate(img):
    if CANDIDATE_ROI is None:
        return img
    cv2.polylines(img, [CANDIDATE_ROI], True, (0, 0, 255), 2, cv2.LINE_AA)
    for i, (px, py) in enumerate(CANDIDATE_ROI):
        cv2.circle(img, (px, py), 6, (0, 0, 255), -1)
        _label(img, f"{i+1}:({px},{py})", (px + 8, py - 8))
    return img


def process_one(video, frame_no, tag=""):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"  [skip] cannot read frame {frame_no}")
        return
    u = undistort(frame)
    u = draw_grid(u)
    u = draw_candidate(u)
    Path("debug").mkdir(exist_ok=True)
    out = f"debug/grid_{tag}{frame_no:04d}.png"
    cv2.imwrite(out, u)
    print(f"  saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, default=120)
    ap.add_argument("--scan", action="store_true",
                    help="save every 15th frame to hunt for feet-up posture")
    args = ap.parse_args()

    if not Path(args.video).exists():
        raise FileNotFoundError(args.video)

    if args.scan:
        cap = cv2.VideoCapture(args.video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"[scan] {total} frames — saving every 15th")
        for fno in range(0, total, 15):
            process_one(args.video, fno, tag="scan_")
        print("\n[done] Browse debug/grid_scan_*.png — find the frame where")

    else:
        process_one(args.video, args.frame)



if __name__ == "__main__":
    main()