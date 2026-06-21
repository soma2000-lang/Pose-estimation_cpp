#pragma once
#include "detector_base.hpp"
#include <onnxruntime_cxx_api.h>
#include <string>
#include <memory>

// ─────────────────────────────────────────────────────────────────────────────
// detector_onnx.hpp
//
// Inference backend using ONNX Runtime with CUDA execution provider.
// Wraps YOLOv8m-pose exported as yolov8m-pose.onnx (opset 12, non-dynamic).
// ─────────────────────────────────────────────────────────────────────────────

class OnnxDetector : public IDetector {
public:
    explicit OnnxDetector(const std::string& model_path);
    ~OnnxDetector() override = default;

    std::vector<Detection> detect(const cv::Mat& frame) override;

private:
    Ort::Env            env_;
    Ort::SessionOptions session_opts_;
    Ort::Session        session_{nullptr};
    Ort::MemoryInfo     mem_info_{nullptr};

    // Preprocess: BGR frame → 1×3×640×640 float32 tensor
    // Returns the float data buffer, fills scale/pad for coord reversal.
    std::vector<float> preprocess(const cv::Mat& frame,
                                  float& scale,
                                  int&   pad_x,
                                  int&   pad_y) const;

    // Postprocess: raw tensor → Detection list in original-frame coords
    std::vector<Detection> postprocess(const float* data,
                                       int    n_anchors,
                                       float  scale,
                                       int    pad_x,
                                       int    pad_y,
                                       int    orig_w,
                                       int    orig_h) const;
};
