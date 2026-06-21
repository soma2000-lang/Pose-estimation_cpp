#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <array>

// ─────────────────────────────────────────────────────────────────────────────
// detector_base.hpp
//
// Abstract interface for the inference backend.
// Both OnnxDetector (development) and TrtDetector (Jetson edge) implement
// this — all downstream code depends only on IDetector, never on internals.
// ─────────────────────────────────────────────────────────────────────────────

struct Detection {
    cv::Rect2f bbox;                           // pixel coords in undist frame
    float      conf;                           // object confidence 0-1
    // 17 COCO keypoints, each [x, y, confidence] in undist frame coords
    std::vector<std::array<float, 3>> keypoints;
};

class IDetector {
public:
    virtual ~IDetector() = default;

    // Detect people and return their keypoints.
    // Input:  undistorted BGR frame (1920x1080 after fisheye correction)
    // Output: vector of detections; each has bbox + 17 keypoints
    virtual std::vector<Detection> detect(const cv::Mat& frame) = 0;
};
