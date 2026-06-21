#include "detector_onnx.hpp"
#include "config.hpp"
#include <algorithm>
#include <stdexcept>
#include <iostream>

// ─────────────────────────────────────────────────────────────────────────────
// detector_onnx.cpp
//
// YOLOv8-pose ONNX Runtime inference.
//
// Input tensor:   name="images"   shape=[1, 3, 640, 640]   float32  RGB/255
// Output tensor:  name="output0"  shape=[1, 56, 8400]      float32
//   Features per anchor: [cx, cy, w, h, obj_conf, kp0_x, kp0_y, kp0_c, ...]
//   Access: value(feature_f, anchor_i) = data[f * 8400 + i]
// ─────────────────────────────────────────────────────────────────────────────

OnnxDetector::OnnxDetector(const std::string& model_path) {
    // Enable all graph optimisations
    session_opts_.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    session_opts_.SetIntraOpNumThreads(1);

    // CUDA execution provider — runs model on GPU 0
    OrtCUDAProviderOptions cuda_opts{};
    cuda_opts.device_id = 0;
    session_opts_.AppendExecutionProvider_CUDA(cuda_opts);

    // Load model
    session_ = Ort::Session(env_, model_path.c_str(), session_opts_);

    // CPU memory info for tensor allocation
    mem_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::cout << "[OnnxDetector] Model loaded from " << model_path << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
std::vector<float> OnnxDetector::preprocess(const cv::Mat& frame,
                                             float& scale,
                                             int&   pad_x,
                                             int&   pad_y) const {
    const int S = Config::INPUT_SIZE;      // 640
    const int W = frame.cols;             // 1920
    const int H = frame.rows;             // 1080

    // Letterbox: scale so the LONGER side fits in S
    scale = std::min(static_cast<float>(S) / W,
                     static_cast<float>(S) / H);
    int nw = static_cast<int>(W * scale);
    int nh = static_cast<int>(H * scale);
    pad_x  = (S - nw) / 2;
    pad_y  = (S - nh) / 2;

    // Build letterboxed image (grey padding 114)
    cv::Mat padded(S, S, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
    resized.copyTo(padded(cv::Rect(pad_x, pad_y, nw, nh)));

    // BGR → RGB
    cv::cvtColor(padded, padded, cv::COLOR_BGR2RGB);

    // Split channels, normalise to [0,1], pack as CHW float32
    padded.convertTo(padded, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> chans(3);
    cv::split(padded, chans);

    std::vector<float> tensor(3 * S * S);
    for (int c = 0; c < 3; ++c)
        std::memcpy(tensor.data() + c * S * S,
                    chans[c].data,
                    S * S * sizeof(float));
    return tensor;
}

// ─────────────────────────────────────────────────────────────────────────────
std::vector<Detection> OnnxDetector::postprocess(const float* data,
                                                  int   n,
                                                  float scale,
                                                  int   pad_x,
                                                  int   pad_y,
                                                  int   orig_w,
                                                  int   orig_h) const {
    // data layout: feature f at anchor i → data[f * n + i]
    // Features: 0-3 bbox(cx,cy,w,h), 4 obj_conf, 5.. keypoints(x,y,c)*17

    std::vector<Detection> dets;
    dets.reserve(64);

    for (int i = 0; i < n; ++i) {
        float conf = data[4 * n + i];
        if (conf < Config::PERSON_CONF) continue;

        // ── Bounding box in original-frame coordinates ──────────────────
        float cx = data[0 * n + i];
        float cy = data[1 * n + i];
        float bw = data[2 * n + i];
        float bh = data[3 * n + i];

        // Reverse letterbox: subtract padding, divide by scale
        float x1 = (cx - bw * 0.5f - pad_x) / scale;
        float y1 = (cy - bh * 0.5f - pad_y) / scale;
        float x2 = (cx + bw * 0.5f - pad_x) / scale;
        float y2 = (cy + bh * 0.5f - pad_y) / scale;

        // Clamp to original frame
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(orig_w)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(orig_h)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(orig_w)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(orig_h)));

        Detection det;
        det.conf = conf;
        det.bbox = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);

        // ── Keypoints in original-frame coordinates ──────────────────────
        det.keypoints.resize(Config::N_KP);
        for (int k = 0; k < Config::N_KP; ++k) {
            float kx = (data[(5 + k * 3    ) * n + i] - pad_x) / scale;
            float ky = (data[(5 + k * 3 + 1) * n + i] - pad_y) / scale;
            float kc =  data[(5 + k * 3 + 2) * n + i];
            det.keypoints[k] = {kx, ky, kc};
        }

        dets.push_back(std::move(det));
    }
    return dets;
}


std::vector<Detection> OnnxDetector::detect(const cv::Mat& frame) {
    float scale; int pad_x, pad_y;
    std::vector<float> tensor = preprocess(frame, scale, pad_x, pad_y);

    const int S = Config::INPUT_SIZE;
    std::vector<int64_t> input_shape = {1, 3, S, S};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info_,
        tensor.data(), tensor.size(),
        input_shape.data(), input_shape.size());

    const char* in_names[]  = {"images"};
    const char* out_names[] = {"output0"};

    auto outputs = session_.Run(
        Ort::RunOptions{nullptr},
        in_names,  &input_tensor, 1,
        out_names, 1);

    // Output shape: [1, 56, 8400]
    auto info  = outputs[0].GetTensorTypeAndShapeInfo();
    auto dims  = info.GetShape();
    int n_anchors = static_cast<int>(dims[2]);   // 8400

    const float* out_data = outputs[0].GetTensorData<float>();
    return postprocess(out_data, n_anchors,
                       scale, pad_x, pad_y,
                       frame.cols, frame.rows);
}
