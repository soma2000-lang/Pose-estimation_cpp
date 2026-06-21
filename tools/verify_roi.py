

import cv2
import numpy as np
import sys
from pathlib import Path



DASHBOARD_ROI = np.array([
    [700,  580],   # top-left  — inboard (toward centre console)
    [1920, 580],   # top-right — outboard (toward passenger door)
    [1920, 1080],  # bottom-right
    [700,  1080]   # bottom-left
], dtype=np.int32)

# ── Camera parameters ─────────────────────────────────────────────────────────
K = np.array([
    [595.8036047830891,  0.0,               924.65430795264774],
    [0.0,               598.33827405037948, 580.41919770485049],
    [0.0,               0.0,               1.0]
], dtype=np.float64)
D = np.array([
    -0.015349419086740696, -0.053676477252104900,
     0.061315407683887907, -0.026142516909791854
], dtype=np.float64)

def main():
    videos = sorted(Path("/home/smajumder/solutions/").rglob("positive_*.mp4"))
    if not videos:
        sys.exit("[ERROR] No videos found under /home/smajumder/solutions/")

    Path("debug").mkdir(exist_ok=True)
    h0, w0 = None, None
    m1, m2 = None, None

    for vid in videos:

        if "verify_" in vid.name:
            continue
            
        print(f"\n[PROCESSING] Rendering verification layer for: {vid.name}")
        cap = cv2.VideoCapture(str(vid))
        
     
        fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        

        max_frames_to_verify = min(total_frames, 300) 
        
        output_path = f"debug/verify_{vid.stem}5.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = None

        frame_count = 0
        while frame_count < max_frames_to_verify:
            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]
            if h != h0 or w != w0:
                h0, w0 = h, w
                Knew = K.copy(); Knew[0,0] *= 0.6; Knew[1,1] *= 0.6
                m1, m2 = cv2.fisheye.initUndistortRectifyMap(
                    K, D, np.eye(3), Knew, (w, h), cv2.CV_16SC2)

            undist = cv2.remap(frame, m1, m2, cv2.INTER_LINEAR)
            overlay = undist.copy()

            cv2.polylines(overlay, [DASHBOARD_ROI], True, (0, 220, 0), 2)
            cv2.fillPoly(overlay, [DASHBOARD_ROI], (0, 200, 0))
            cv2.addWeighted(overlay, 0.15, undist, 0.85, 0, undist)
            cv2.polylines(undist, [DASHBOARD_ROI], True, (0, 220, 0), 2)

            # Label metadata on frame
            cv2.putText(undist, f"Verifying: {vid.name}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2)

            # Lazy initialize video writer using actual frame sizes
            if out_writer is None:
                out_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

            out_writer.write(undist)
            frame_count += 1

        cap.release()
        if out_writer is not None:
            out_writer.release()
            print(f"[SUCCESS] Saved verification clip to: {output_path}")

    print("\n[DONE] Headless ROI verification video generation complete.")
    print("Next step: View files in 'debug/' then run: python tools/3_extract_features.py")

if __name__ == "__main__":
    main()