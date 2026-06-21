"""

━━━━━━━━━━━━━━━━━━━━━
Train a logistic-regression classifier on the body-size-normalized pose
features.  Trains on ALL labeled data (all 3 profiles combined).

The classifier is intentionally tiny:
  - 10 input features
  - Logistic regression (linear boundary)
  - class_weight='balanced' to handle the NEGATIVE-heavy imbalance
  - StandardScaler pre-processing (folded into weights by export_head.py)

Why logistic regression?
  - Generalizes better than MLP on small datasets (< 50 000 rows)
  - Zero inference cost in C++: one dot product
  - Interpretable: weight signs tell you which features matter


Input:   data/labeled.csv
Output:  models/head.pkl   (sklearn Pipeline: StandardScaler + LogReg)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score)
FEAT_DISPLAY_NAMES = [
    "L_ankle_height", "L_ankle_lateral", "L_knee_angle",
    "L_above_knee",   "L_ankle_conf",
    "R_ankle_height", "R_ankle_lateral", "R_knee_angle",
    "R_above_knee",   "R_ankle_conf",
]
FEAT_NAMES = [f"f{i}" for i in range(10)]

def main():
    in_path  = Path("data/labeled.csv")
    out_path = Path("models/head.pkl")
    out_path.parent.mkdir(exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"{in_path} not found. Run 4_auto_label.py first.")

    df = pd.read_csv(in_path).dropna()
    df = df[df['profile'].isin(['profile1', 'profile2'])]
    # Per-profile baseline normalization using negative frames only
    neg_mask = df['auto_label'] == 0

    feature_cols = FEAT_NAMES 
   
    for prof in df['profile'].unique():
        prof_neg = df[(df['profile'] == prof) & neg_mask]
        baseline = prof_neg[feature_cols].mean()
        df.loc[df['profile'] == prof, feature_cols] -= baseline.values
  
    X = df[feature_cols].values.astype(np.float32)
    y = df["auto_label"].astype(int).values

    pos = y.sum(); neg = len(y) - pos
    print(f"[INFO] Dataset: {len(df)} rows | POSITIVE={pos} | NEGATIVE={neg}")

    # ── Train on all data ─────────────────────────────────────────────────────
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight = "balanced",
            max_iter     = 2000,
            C            = 0.1,          # regularisation (1.0 = moderate)
            solver       = "lbfgs",
            random_state = 42))

   
    X_mirror = X.copy()
    X_mirror[:, [0,1,2,3,4, 5,6,7,8,9]] = X[:, [5,6,7,8,9, 0,1,2,3,4]]
    X_mirror[:, 1] *= -1   # L_lateral sign flip
    X_mirror[:, 6] *= -1   # R_lateral sign flip

    X_aug = np.vstack([X, X_mirror])
    y_aug = np.concatenate([y, y])

    print(f"[INFO] Augmented dataset: {len(X_aug)} rows (original + L/R mirror)")
    clf.fit(X_aug, y_aug)
    joblib.dump(clf, out_path)
    print(f"[INFO] Model saved → {out_path}")

    # ── In-sample report (just to confirm training worked) ───────────────────
    pred = clf.predict(X)
    print("\n[TRAINING METRICS — in-sample]")
    print(classification_report(y, pred,
                                 target_names=["NEGATIVE","POSITIVE"]))
    print("Confusion matrix:")
    print(confusion_matrix(y, pred))

    scaler = clf.named_steps["standardscaler"]
    lr     = clf.named_steps["logisticregression"]
    # Weights in original (unscaled) feature space
    w_orig = lr.coef_[0] / scaler.scale_
    print("\n[FEATURE IMPORTANCE  (unscaled weight magnitude)]")
    for name, w in sorted(zip(FEAT_DISPLAY_NAMES, w_orig),
                          key=lambda x: abs(x[1]), reverse=True):
        bar = "█" * min(int(abs(w) * 20), 30)
        sign = "+" if w > 0 else "-"
        print(f"  {name:<22} {sign}{abs(w):6.3f}  {bar}")

    print(f"\n[DONE] Head trained.  F1 (in-sample) = {f1_score(y, pred):.4f}")
    print("Next step:  python tools/6_validate_head.py")


if __name__ == "__main__":
    main()

