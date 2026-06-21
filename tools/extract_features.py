
import csv, math, sys
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("Install ultralytics first: pip install ultralytics")

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

# ── COCO keypoint indices ──────────────────────────────────────────────────────
KP_L_SHLDR, KP_R_SHLDR = 5,  6
KP_L_HIP,   KP_R_HIP   = 11, 12
KP_L_KNEE,  KP_R_KNEE  = 13, 14
KP_L_ANKLE, KP_R_ANKLE = 15, 16

# ── Passenger seat filter ─────────────────────────────────────────────────────
# MANIPULATION: Aligned to 900.0 to perfectly isolate the passenger side 
# and ignore center console elements/driver artifacts.
SEAT_XMIN, SEAT_XMAX = 600.0, 1920.0

# ─────────────────────────────────────────────────────────────────────────────
def build_maps(h, w):
    Knew = K.copy()
    Knew[0, 0] *= FOV_SCALE
    Knew[1, 1] *= FOV_SCALE
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), Knew, (w, h), cv2.CV_16SC2)
    return m1, m2

def knee_angle_norm(h_pt, k_pt, a_pt):
    """Angle at the knee (hip-knee-ankle), normalised to [0, 1]."""
    bk = h_pt - k_pt
    bc = a_pt - k_pt
    denom = np.linalg.norm(bk) * np.linalg.norm(bc)
    if denom < 1e-6:
        return 0.0
    cos_val = np.dot(bk, bc) / denom
    angle_deg = math.degrees(math.acos(float(np.clip(cos_val, -1, 1))))
    return angle_deg / 180.0

def extract_features(kp):
    """
    kp: numpy array shape (17, 3) — [x, y, confidence] for each keypoint.
    Returns list of 10 floats in the same order as C++ extractFeatures().
    """
    sh = np.array([(kp[KP_L_SHLDR][0] + kp[KP_R_SHLDR][0]) / 2,
                   (kp[KP_L_SHLDR][1] + kp[KP_R_SHLDR][1]) / 2])
    hp = np.array([(kp[KP_L_HIP][0] + kp[KP_R_HIP][0]) / 2,
                   (kp[KP_L_HIP][1] + kp[KP_R_HIP][1]) / 2])
    torso = max(np.linalg.norm(sh[:2] - hp[:2]), 1.0)  # x,y only — match C++

    feats = []
    for (ai, ki, hi) in [(KP_L_ANKLE, KP_L_KNEE, KP_L_HIP),
                          (KP_R_ANKLE, KP_R_KNEE, KP_R_HIP)]:
        ax, ay, ac = kp[ai][0], kp[ai][1], kp[ai][2]
        kx, ky     = kp[ki][0], kp[ki][1]
        hx, hy     = kp[hi][0], kp[hi][1]

        feats.append((ay - hp[1]) / torso)                      # height norm
        feats.append((ax - hp[0]) / torso)                      # lateral norm
        feats.append(knee_angle_norm(
            np.array([hx, hy]),
            np.array([kx, ky]),
            np.array([ax, ay])))                                 # angle norm
        feats.append(1.0 if ay < ky else 0.0)                   # above knee
        feats.append(float(ac))                                  # confidence
    return feats   # length 10

def select_passenger(result):
    """Return the keypoints (17,3) for the front passenger, or None."""
    if result.keypoints is None or result.boxes is None:
        return None
    boxes = result.boxes.xyxy.cpu().numpy()   # (N, 4)
    confs = result.boxes.conf.cpu().numpy()   # (N,)
    kps   = result.keypoints.data.cpu().numpy()  # (N, 17, 3)

    best_kp, best_cx = None, -1.0
    for i in range(len(boxes)):
        if confs[i] < 0.25:
            continue
        cx = (boxes[i][0] + boxes[i][2]) / 2.0
        if SEAT_XMIN < cx < SEAT_XMAX and cx > best_cx:
            best_cx = cx
            best_kp = kps[i]
    return best_kp

# ─────────────────────────────────────────────────────────────────────────────
def main():
    out_dir  = Path("data")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "features.csv"


    video_dir = Path("/home/smajumder/solutions/Solution/solution/input")
    videos = sorted(video_dir.rglob("*.mp4"))
    
    if not videos:
        sys.exit(f"[ERROR] No videos found under {video_dir}")

    print(f"[INFO] Found {len(videos)} videos in target folder.")
    print("[INFO] Loading YOLOv8m-pose (inference only)…")
    model = YOLO("models/yolov8m-pose.pt")

    header = ["video", "profile", "frame", "video_label"] + \
             [f"f{i}" for i in range(10)]

    total_rows = 0
    h0, w0, m1, m2 = None, None, None, None

    with open(out_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for vid in videos:
            if "verify_" in vid.name:
                continue
                
            vid_label = 1 if "positive" in vid.name else 0
            profile   = vid.parent.name
            cap       = cv2.VideoCapture(str(vid))
            fidx      = 0

            print(f"\n  {vid.name}  (label={vid_label})")
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                h, w = frame.shape[:2]
                if h != h0 or w != w0:
                    h0, w0 = h, w
                    m1, m2 = build_maps(h, w)

                undist  = cv2.remap(frame, m1, m2, cv2.INTER_LINEAR)
                results = model(undist, verbose=False)
                kp      = select_passenger(results[0])

                if kp is not None:
                    feats = extract_features(kp)
                    writer.writerow([vid.name, profile, fidx, vid_label] + feats)
                    total_rows += 1

                if fidx % 100 == 0:
                    print(f"    frame {fidx}\r", end="", flush=True)
                fidx += 1

            cap.release()
            print(f"    Done: {fidx} frames, {total_rows} rows total")

    print(f"\n[DONE] Wrote {total_rows} feature rows → {out_path}")
    print("Next step:  python tools/4_auto_label.py")


if __name__ == "__main__":
    main()
