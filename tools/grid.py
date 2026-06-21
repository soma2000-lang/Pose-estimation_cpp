

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

# Matching the config.hpp in FOV_SCALE
FOV_SCALE = 0.75


CANDIDATE_ROI = np.array([
    [700,  580],   # top-left  — inboard
    [1920, 580],   # top-right — outboard
    [1920, 1080],  # bottom-right
    [700,  1080],  # bottom-left
], dtype=np.int32)


def undistort(frame):
    h, w = frame.shape[:2]
    Knew = K.copy()
    Knew[0, 0] *= FOV_SCALE
    Knew[1, 1] *= FOV_SCALE
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), Knew, (w, h), cv2.CV_16SC2)
    return cv2.remap(frame, m1, m2, cv2.INTER_LINEAR)


def _label(img, text, org, scale=0.55, thick=1):
    """Draw text with dark outline so it's readable on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 255, 0), thick, cv2.LINE_AA)


def draw_grid(img, major=100, minor=50):
    """Draw grid LINES ONLY (no fill). Semi-transparent so image stays visible."""
    h, w = img.shape[:2]
    overlay = img.copy()

    # Minor lines first (dim)
    for x in range(0, w, minor):
        if x % major != 0:
            cv2.line(overlay, (x, 0), (x, h), (0, 80, 0), 1)
    for y in range(0, h, minor):
        if y % major != 0:
            cv2.line(overlay, (0, y), (w, y), (0, 80, 0), 1)

    # Major lines (bright)
    for x in range(0, w, major):
        cv2.line(overlay, (x, 0), (x, h), (0, 200, 0), 1)
    for y in range(0, h, major):
        cv2.line(overlay, (0, y), (w, y), (0, 200, 0), 1)

    # Blend with original — grid stays visible but image readable
    out = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

    # Labels (drawn on top, fully opaque with outline)
    for x in range(0, w, major):
        _label(out, str(x), (x + 3, 22))
    for y in range(0, h, major):
        _label(out, str(y), (3, y + 18))

    return out


def draw_candidate(img):
    if CANDIDATE_ROI is None:
        return img
    cv2.polylines(img, [CANDIDATE_ROI], True, (0, 0, 255), 2, cv2.LINE_AA)
    for i, (px, py) in enumerate(CANDIDATE_ROI):
        cv2.circle(img, (px, py), 6, (0, 0, 255), -1)
        # Red labels for the candidate corners
        txt = f"{i+1}:({px},{py})"
        cv2.putText(img, txt, (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
    return img


def add_header(img, video_name, frame_no):
    """Top-right banner with video name and frame number."""
    h, w = img.shape[:2]
    txt = f"{Path(video_name).name}  frame {frame_no}"
    cv2.rectangle(img, (w - 600, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(img, txt, (w - 590, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


def process_one(video, frame_no, tag=""):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"  [skip] cannot read frame {frame_no}")
        return False
    u = undistort(frame)
    u = draw_grid(u)
    u = draw_candidate(u)
    add_header(u, video, frame_no)

    Path("debug").mkdir(exist_ok=True)
    out = f"debug/grid_{tag}{frame_no:04d}.png"
    cv2.imwrite(out, u)
    print(f"  saved {out}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, default=120)
    ap.add_argument("--scan", action="store_true",
                    help="save every Nth frame to find feet-up posture")
    ap.add_argument("--step", type=int, default=15,
                    help="frame step for --scan mode (default 15)")
    args = ap.parse_args()

    if not Path(args.video).exists():
        raise FileNotFoundError(args.video)

    if args.scan:
        cap = cv2.VideoCapture(args.video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"[scan] {args.video}: {total} frames, saving every {args.step}th")
        count = 0
        for fno in range(0, total, args.step):
            if process_one(args.video, fno, tag="scan_"):
                count += 1
        print(f"\n[done] Saved {count} frames to debug/grid_scan_*.png")
        
    else:
        process_one(args.video, args.frame)



if __name__ == "__main__":
    main()