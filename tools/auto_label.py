"""
tools/4_auto_label.py
━━━━━━━━━━━━━━━━━━━━━
Convert video-level ground truth to per-frame labels using weak supervision.

Strategy:
  negative_*.mp4 → ALL frames NEGATIVE (100% certain)
  positive_*.mp4 → use the geometric rule as a frame-level proposer,
                   then apply a short minimum-duration filter to remove
                   single-frame spikes that are likely noise.

The rule proposer uses:
  - ankle_above_knee     (feature f3 or f8)
  - ankle height norm    (feature f0 or f5) — threshold for OOP height


Usage:
    cd solution/
    python tools/4_auto_label.py

Input:   data/features.csv
Output:  data/labeled.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path


HEIGHT_THRESH = 0.3     # ankle_y significantly BELOW hip (positive = lower)
LATERAL_THRESH = 0.2    # ankle far to the outboard side
MIN_POSITIVE_RUN = 3

def rule_propose(row):
    """Return 1 if geometric features suggest OOP in this camera's geometry."""
    # In this camera: OOP ankle is BELOW hip (large positive height norm)
    # f0 = (ankle_y - hip_y) / torso  → large positive = ankle far below hip
    left_low  = float(row["f0"]) > HEIGHT_THRESH
    right_low = float(row["f5"]) > HEIGHT_THRESH

    # Knee angle: when legs are raised onto dash, knee angle is very different
    # f2 = knee angle norm (0=straight, 1=fully bent)
    left_bent  = float(row["f2"]) > 0.5
    right_bent = float(row["f7"]) > 0.5

    # Either ankle is far below hip (on dash from camera's perspective)
    return int((left_low and left_bent) or (right_low and right_bent))

def smooth_run_filter(labels, min_run):
    """Remove positive runs shorter than min_run frames (noise)."""
    arr = np.array(labels, dtype=int)
    out = arr.copy()
    i = 0
    while i < len(arr):
        if arr[i] == 1:
            j = i
            while j < len(arr) and arr[j] == 1:
                j += 1
            run_len = j - i
            if run_len < min_run:
                out[i:j] = 0
            i = j
        else:
            i += 1
    return out.tolist()

def main():
    in_path  = Path("data/features.csv")
    out_path = Path("data/labeled.csv")

    if not in_path.exists():
        raise FileNotFoundError(f"{in_path} not found. Run 3_extract_features.py first.")

    df = pd.read_csv(in_path)
    print(f"[INFO] Loaded {len(df)} rows from {in_path}")

    df["auto_label"] = 0   # default NEGATIVE

    # ── Negatives: all frames are NEGATIVE (ground truth) ────────────────────
    neg_mask = df["video"].str.contains("negative")
    df.loc[neg_mask, "auto_label"] = 0
    print(f"[INFO] Negatives: {neg_mask.sum()} frames → all label=0 (certain)")

    # ── Positives: rule proposer + noise filter ───────────────────────────────
    pos_mask = df["video"].str.contains("positive")
    for vid_name in df.loc[pos_mask, "video"].unique():
        vid_df = df[df["video"] == vid_name].copy()
        proposed = vid_df.apply(rule_propose, axis=1).tolist()
        filtered  = smooth_run_filter(proposed, MIN_POSITIVE_RUN)
        df.loc[df["video"] == vid_name, "auto_label"] = filtered

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n[LABEL DISTRIBUTION]")
    for profile in df["profile"].unique():
        sub = df[df["profile"] == profile]
        pos = (sub["auto_label"] == 1).sum()
        neg = (sub["auto_label"] == 0).sum()
        print(f"  {profile}: POSITIVE={pos}  NEGATIVE={neg}")

    total_pos = (df["auto_label"] == 1).sum()
    total_neg = (df["auto_label"] == 0).sum()
    ratio = total_pos / max(total_neg, 1)
    print(f"\n  Total: POSITIVE={total_pos}  NEGATIVE={total_neg}  ratio={ratio:.3f}")
    if ratio < 0.05:
        print("  [WARN] Very few positives — check HEIGHT_THRESH or your ROI calibration.")

    # ── Save transitions for manual review ───────────────────────────────────
    debug_dir = Path("debug/transitions")
    debug_dir.mkdir(parents=True, exist_ok=True)
    transitions = df[df["auto_label"] != df["auto_label"].shift(1).fillna(0)]
    transitions[["video","profile","frame","auto_label"]].to_csv(
        debug_dir / "transition_frames.csv", index=False)
    print(f"\n[INFO] Transition frames saved → {debug_dir}/transition_frames.csv")
    print("  RECOMMENDATION: open this CSV and spot-check the frame indices")
    print("  against your videos using verify_roi.py to confirm transitions look right.")

    df.to_csv(out_path, index=False)
    print(f"\n[DONE] Wrote {len(df)} labeled rows → {out_path}")
    print("Next step:  python tools/5_train_head.py")
    # Add this at the top of main() in 4_auto_label.py, before any processing:

    pos_df = df[df["video"].str.contains("positive")]
    print(f"  f0 (L ankle height): min={pos_df.f0.min():.3f}  max={pos_df.f0.max():.3f}  mean={pos_df.f0.mean():.3f}")
    print(f"  f5 (R ankle height): min={pos_df.f5.min():.3f}  max={pos_df.f5.max():.3f}  mean={pos_df.f5.mean():.3f}")
    print(f"  f2 (L knee angle):   min={pos_df.f2.min():.3f}  max={pos_df.f2.max():.3f}")
    print(f"  f7 (R knee angle):   min={pos_df.f7.min():.3f}  max={pos_df.f7.max():.3f}")
    print(f"  f3 (L above knee):   sum={pos_df.f3.sum()}")
    print(f"  f8 (R above knee):   sum={pos_df.f8.sum()}")

if __name__ == "__main__":
    main()
