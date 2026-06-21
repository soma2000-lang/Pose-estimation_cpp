import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             confusion_matrix, classification_report,
                             precision_recall_curve)

def main():
    in_path = Path("data/labeled.csv")
    if not in_path.exists():
        raise FileNotFoundError(f"{in_path} not found. Run 4_auto_label.py first.")

    df = pd.read_csv(in_path).dropna()
    feat_cols = [f"f{i}" for i in range(10)]
    X = df[feat_cols].values.astype(np.float32)
    y = df["auto_label"].astype(int).values
    
    # Adapt to available data: prefer profile-out, fall back to video-out
    profiles = sorted(df["profile"].unique())
    videos = sorted(df["video"].unique())

    if len(profiles) >= 2:
        groups = df["profile"].values
        group_list = profiles
        mode = "LEAVE-ONE-PROFILE-OUT"
    else:
        groups = df["video"].values
        group_list = videos
        mode = "LEAVE-ONE-VIDEO-OUT (only 1 profile available)"

    print("=" * 62)
    print(f"  {mode} CROSS-VALIDATION")
    print("=" * 62)

    all_f1 = []
    for held in group_list:
        tr_mask = groups != held
        te_mask = groups == held

        X_tr, y_tr = X[tr_mask], y[tr_mask]
        X_te, y_te = X[te_mask], y[te_mask]

        # ── Symmetric L/R augmentation on training set only ──────────────
        X_mirror = X_tr.copy()
        X_mirror[:, [0,1,2,3,4, 5,6,7,8,9]] = X_tr[:, [5,6,7,8,9, 0,1,2,3,4]]
        X_mirror[:, 1] *= -1   # L_lateral sign flip
        X_mirror[:, 6] *= -1   # R_lateral sign flip
        X_tr_aug = np.vstack([X_tr, X_mirror])
        y_tr_aug = np.concatenate([y_tr, y_tr])

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced",
                               max_iter=2000, C=0.1,
                               solver="lbfgs", random_state=42))
        clf.fit(X_tr_aug, y_tr_aug)
        pred = clf.predict(X_te)

        f1   = f1_score(y_te, pred, zero_division=0)
        prec = precision_score(y_te, pred, zero_division=0)
        rec  = recall_score(y_te, pred, zero_division=0)
        cm   = confusion_matrix(y_te, pred)
        all_f1.append(f1)

        print(f"\n  Held-out: {held}  (train on: {[g for g in group_list if g != held]})")
        print(f"  F1={f1:.3f}  Precision={prec:.3f}  Recall={rec:.3f}")
        print(f"  Confusion matrix (rows=actual, cols=pred):")
        print(f"    TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
        print(f"    FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

        # Flag worst error frames for manual review
        wrong_mask = pred != y_te
        if wrong_mask.sum() > 0:
            wrong_df = df[te_mask][wrong_mask][["video","frame","auto_label"]].copy()
            wrong_df["predicted"] = pred[wrong_mask]
            wrong_path = Path(f"debug/errors_{held}.csv")
            wrong_path.parent.mkdir(exist_ok=True)
            wrong_df.head(50).to_csv(wrong_path, index=False)
            print(f"  Error frames → {wrong_path}  (first 50)")

    # ── 1. Train final model on full data for global threshold evaluation ──
    X_mirror_full = X.copy()
    X_mirror_full[:, [0,1,2,3,4, 5,6,7,8,9]] = X[:, [5,6,7,8,9, 0,1,2,3,4]]
    X_mirror_full[:, 1] *= -1
    X_mirror_full[:, 6] *= -1
    X_full_aug = np.vstack([X, X_mirror_full])
    y_full_aug = np.concatenate([y, y])

    clf_final = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced",
                           max_iter=2000, C=0.1,
                           solver="lbfgs", random_state=42))
    clf_final.fit(X_full_aug, y_full_aug)

    # ── 2. Global Threshold Tuning Block ───────────────────────────────────
    print("\n====== GLOBAL THRESHOLD TUNING ======")
    all_proba, all_y = [], []
    for prof in profiles:
        test_mask = df['profile'] == prof
        X_test = df[test_mask][feat_cols].values
        y_test = df[test_mask]['auto_label'].values
        proba = clf_final.predict_proba(X_test)[:, 1]
        all_proba.extend(proba)
        all_y.extend(y_test)

    precisions, recalls, thresholds = precision_recall_curve(all_y, all_proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = f1s.argmax()
    
    # Safe guard index bounds check for threshold assignment
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    
    print(f"  Best global threshold = {best_thresh:.3f}  →  F1 = {f1s[best_idx]:.3f}")

    # ── 3. Final Summary Reporting ──────────────────────────────────────────
    mean_f1 = np.mean(all_f1)
    print("\n" + "=" * 62)
   
    valid_f1 = [f for f, g in zip(all_f1, group_list) if df[df.video == g].auto_label.sum() > 0]
    if valid_f1:
        print(f"  Mean F1 over {len(valid_f1)} held-out sets WITH POSITIVES: {np.mean(valid_f1):.3f}")
    print(f"  Mean F1 over all {len(group_list)} held-out sets: {mean_f1:.3f} (includes 0s from negative-only sets)")
    print("=" * 62)

    if valid_f1:
        judge_f1 = float(np.mean(valid_f1))
    else:
        judge_f1 = mean_f1

    if judge_f1 >= 0.90:
        verdict = "EXCELLENT — ship the learned head, report proudly"
    elif judge_f1 >= 0.75:
        verdict = "GOOD — acceptable; note limitations in report"
    elif judge_f1 >= 0.60:
        verdict = "WEAK — inspect error frames; consider rule fallback"
    else:
        verdict = "POOR — check label quality and feature normalization"
    print(f"  Verdict: {verdict}  (based on F1={judge_f1:.3f} over sets with positives)")

if __name__ == "__main__":
    main()
