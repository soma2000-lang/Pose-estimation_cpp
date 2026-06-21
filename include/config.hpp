#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <array>

namespace Config {


inline const cv::Matx33d K = {
    595.8036047830891,  0.0,                924.65430795264774,
    0.0,                598.33827405037948, 580.41919770485049,
    0.0,                0.0,                1.0
};
inline const cv::Vec4d D = {
    -0.015349419086740696, -0.053676477252104900,
     0.061315407683887907, -0.026142516909791854
};

inline constexpr double FOV_SCALE = 0.75; 
inline const std::vector<cv::Point> DASHBOARD_ROI = {
    {700,  580},   // top-left  — inboard (toward centre console)
    {1920, 580},   // top-right — outboard (toward passenger door)
    {1920, 1080},  // bottom-right
    {700,  1080}   // bottom-left
};

static const float PROFILE_BASELINE[3][10] = {
    { 0.393195f, 0.401971f, 0.261412f, 0.760331f, 0.092303f, 0.479572f, -0.019560f, 0.324930f, 0.772137f, 0.123406f },
    { 0.498809f, 0.610054f, 0.192195f, 0.971429f, 0.140912f, 0.519389f, 0.070539f, 0.307370f, 0.985034f, 0.127795f },
    { 0.784182f, 0.681085f, 0.632715f, 0.033992f, 0.430188f, 1.061210f, 0.118400f, 0.706213f, 0.037945f, 0.436478f },
};

inline constexpr int ROI_DILATE_PX = 12;
inline constexpr int ROI_ERODE_PX  = 12;

inline constexpr float SEAT_XMIN = 600.0f;
inline constexpr float SEAT_XMAX = 1920.0f;


inline const char* MODEL_PATH   = "models/yolov8m-pose.onnx";
inline constexpr float PERSON_CONF = 0.45f;   // min bbox confidence
inline constexpr float KP_CONF     = 0.25f;   // min keypoint confidence
inline constexpr int   INPUT_SIZE  = 640;      // YOLO input resolution


inline constexpr int WINDOW_SIZE    = 7;   // majority-vote window (frames)
inline constexpr int OCCLUSION_HOLD = 5;   // hold last ankle pos (frames)

// ═══════════════════════════════════════════════════════════════════════════
// COCO-17 KEYPOINT INDICES  (fixed )
// ═══════════════════════════════════════════════════════════════════════════
inline constexpr int KP_L_SHLDR =  5, KP_R_SHLDR =  6;
inline constexpr int KP_L_HIP   = 11, KP_R_HIP   = 12;
inline constexpr int KP_L_KNEE  = 13, KP_R_KNEE  = 14;
inline constexpr int KP_L_ANKLE = 15, KP_R_ANKLE = 16;
inline constexpr int N_KP       = 17;

// ═══════════════════════════════════════════════════════════════════════════
// LEARNED CLASSIFIER HEAD
// ──────────────────────────────────────────────────────────────────────────
// Feature vector (10 values, 5 per leg, same order as extract_features.py):
//   [0] left  ankle height norm  (ankle_y - hip_mid_y) / torso_len
//   [1] left  ankle lateral norm (ankle_x - hip_mid_x) / torso_len
//   [2] left  knee angle (degrees / 180, normalised)
//   [3] left  ankle_above_knee   (1.0 or 0.0)
//   [4] left  ankle confidence
//   [5] right ankle height norm
//   [6] right ankle lateral norm
//   [7] right knee angle
//   [8] right ankle_above_knee
//   [9] right ankle confidence
//

inline constexpr int N_FEATURES = 10;



inline const std::array<float, N_FEATURES> HEAD_WEIGHTS = {
  0.51583318f, -0.85875940f, 2.30870652f, -0.75555756f, 5.59127656f,
  0.51583318f, 0.85875940f, 2.30870652f, -0.75555756f, 5.59127656f
};
inline constexpr float HEAD_BIAS = -2.73810635f;
inline constexpr bool  USE_LEARNED_HEAD = true;


inline constexpr float HEAD_THRESHOLD =  0.421f;

} // namespace Config
