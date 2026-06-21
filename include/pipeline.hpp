#pragma once
#include "detector_base.hpp"
#include "smoother.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <array>

// ─────────────────────────────────────────────────────────────────────────────
// pipeline.hpp
//
// Runs the full per-video pipeline:
//   1. Read raw fisheye frame from VideoCapture
//   2. Undistort via precomputed remap tables (Kannala-Brandt fisheye)
//   3. Detect → extract keypoints (via IDetector)
//   4. Select front passenger detection
//   5. Extract body-size-normalized features
//   6. Classify: learned head (primary) or rule-based (fallback)
//   7. Temporal majority-vote smoothing
//   8. Draw annotations → write to output VideoWriter
// ─────────────────────────────────────────────────────────────────────────────


void buildUndistortMaps(int W, int H,
                        cv::Mat& map1, cv::Mat& map2);

// Dilate or erode the dashboard ROI polygon by px pixels.
std::vector<cv::Point> morphROI(const std::vector<cv::Point>& roi, int px);

// Select the best passenger detection from a list.
// Returns nullptr if no valid detection found in the seat region.
const Detection* selectPassenger(const std::vector<Detection>& dets);

std::array<float, 10> extractFeatures(
    const std::vector<std::array<float,3>>& kp);

// Learned logistic-regression head (folded scaler — pure dot product).
bool classifyLearned(const std::array<float, 10>& feats);

// Rule-based fallback (ankle inside eroded/dilated ROI + hysteresis).
bool classifyRule(const Detection* psg, bool prev_state,
                  AnkleMemory& mem);

// Top-level per-frame decision: uses learned head when available.
bool classifyFrame(const Detection* psg, bool prev_state,
                   AnkleMemory& mem);

// Draw all annotations onto the undistorted frame.
void drawAnnotations(cv::Mat& frame,
                     const std::vector<Detection>& dets,
                     const Detection* passenger,
                     bool label);

// Process one video file end-to-end.
void processVideo(IDetector& detector,
                  const std::string& input_path,
                  const std::string& output_path);
