#include "pipeline.hpp"
#include "config.hpp"
#include "smoother.hpp"
#include <cmath>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// Fisheye undistortion maps (built once per video resolution)
// ─────────────────────────────────────────────────────────────────────────────
void buildUndistortMaps(int W, int H, cv::Mat& map1, cv::Mat& map2) {
    cv::Matx33d Knew = Config::K;
    Knew(0, 0) *= Config::FOV_SCALE;
    Knew(1, 1) *= Config::FOV_SCALE;
    cv::fisheye::initUndistortRectifyMap(
        Config::K, Config::D,
        cv::Matx33d::eye(), Knew,
        cv::Size(W, H), CV_16SC2,
        map1, map2);
}

// ─────────────────────────────────────────────────────────────────────────────
// ROI morphology helpers
// ─────────────────────────────────────────────────────────────────────────────
std::vector<cv::Point> morphROI(const std::vector<cv::Point>& roi, int px) {
    if (px == 0) return roi;
    // Compute centroid, move each point toward/away by px
    cv::Point2f centroid(0, 0);
    for (auto& p : roi) centroid += cv::Point2f(p);
    centroid *= 1.0f / roi.size();
    std::vector<cv::Point> out;
    out.reserve(roi.size());
    for (auto& p : roi) {
        cv::Point2f d = cv::Point2f(p) - centroid;
        float len = std::max(std::hypot(d.x, d.y), 1.0f);
        float factor = 1.0f + px / len;
        out.emplace_back(static_cast<int>(centroid.x + d.x * factor),
                         static_cast<int>(centroid.y + d.y * factor));
    }
    return out;
}

static bool insideROI(const std::vector<cv::Point>& roi, float x, float y) {
    return cv::pointPolygonTest(roi, cv::Point2f(x, y), false) >= 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Passenger selection: rightmost bbox centre in [SEAT_XMIN, SEAT_XMAX]
// ─────────────────────────────────────────────────────────────────────────────
const Detection* selectPassenger(const std::vector<Detection>& dets) {
    const Detection* best = nullptr;
    float best_cx = -1.0f;
    for (const auto& d : dets) {
        if (d.conf < Config::PERSON_CONF) continue;
        float cx = d.bbox.x + d.bbox.width * 0.5f;
        if (cx > Config::SEAT_XMIN && cx < Config::SEAT_XMAX && cx > best_cx) {
            best_cx = cx;
            best    = &d;
        }
    }
    return best;
}

// ─────────────────────────────────────────────────────────────────────────────
// Feature extraction  — MUST match tools/3_extract_features.py exactly
// ─────────────────────────────────────────────────────────────────────────────
static float kneeAngle(float hx, float hy,
                       float kx, float ky,
                       float ax, float ay) {
    float bkx = hx - kx, bky = hy - ky;   // bone: knee→hip
    float bcx = ax - kx, bcy = ay - ky;   // bone: knee→ankle
    float dot  = bkx*bcx + bky*bcy;
    float denom = std::hypot(bkx, bky) * std::hypot(bcx, bcy);
    if (denom < 1e-6f) return 0.0f;
    float angle_deg = std::acos(std::max(-1.0f, std::min(1.0f, dot/denom)))
                      * (180.0f / M_PI);
    return angle_deg / 180.0f;   // normalise to [0,1]
}

std::array<float, 10> extractFeatures(
    const std::vector<std::array<float, 3>>& kp) {

    // Mid-shoulder and mid-hip
    float sh_x = (kp[Config::KP_L_SHLDR][0] + kp[Config::KP_R_SHLDR][0]) * 0.5f;
    float sh_y = (kp[Config::KP_L_SHLDR][1] + kp[Config::KP_R_SHLDR][1]) * 0.5f;
    float hp_x = (kp[Config::KP_L_HIP][0]   + kp[Config::KP_R_HIP][0])   * 0.5f;
    float hp_y = (kp[Config::KP_L_HIP][1]   + kp[Config::KP_R_HIP][1])   * 0.5f;
    float torso = std::max(std::hypot(sh_x - hp_x, sh_y - hp_y), 1.0f);

    std::array<float, 10> f{};
    // Leg pair: (ankle_idx, knee_idx, hip_idx), slot offset 0 and 5
    const int LEGS[2][3] = {
        {Config::KP_L_ANKLE, Config::KP_L_KNEE, Config::KP_L_HIP},
        {Config::KP_R_ANKLE, Config::KP_R_KNEE, Config::KP_R_HIP}
    };
    for (int leg = 0; leg < 2; ++leg) {
        int ai = LEGS[leg][0], ki = LEGS[leg][1], hi = LEGS[leg][2];
        float ax = kp[ai][0], ay = kp[ai][1], ac = kp[ai][2];
        float kx = kp[ki][0], ky = kp[ki][1];
        float hx = kp[hi][0], hy = kp[hi][1];

        int slot = leg * 5;
        f[slot + 0] = (ay - hp_y) / torso;              // height norm
        f[slot + 1] = (ax - hp_x) / torso;              // lateral norm
        f[slot + 2] = kneeAngle(hx, hy, kx, ky, ax, ay); // angle norm
        f[slot + 3] = (ay < ky) ? 1.0f : 0.0f;          // above knee
        f[slot + 4] = ac;                                 // conf
    }
    return f;
}

// ─────────────────────────────────────────────────────────────────────────────
// Learned logistic-regression head (dot product — no ML library needed)
// ─────────────────────────────────────────────────────────────────────────────
bool classifyLearned(const std::array<float, 10>& feats) {
    float logit = Config::HEAD_BIAS;
    for (int i = 0; i < Config::N_FEATURES; ++i)
        logit += Config::HEAD_WEIGHTS[i] * feats[i];
    float prob = 1.0f / (1.0f + std::exp(-logit));
    return prob > Config::HEAD_THRESHOLD;
}

// ─────────────────────────────────────────────────────────────────────────────
// Rule-based classifier with hysteresis ROI and occlusion memory
// ─────────────────────────────────────────────────────────────────────────────
bool classifyRule(const Detection* psg, bool prev_state, AnkleMemory& mem) {
    if (!psg) return false;

    // Build hysteresis ROIs
    static auto enter_roi = morphROI(Config::DASHBOARD_ROI, -Config::ROI_ERODE_PX);
    static auto exit_roi  = morphROI(Config::DASHBOARD_ROI,  Config::ROI_DILATE_PX);

    const auto& kp  = psg->keypoints;
    bool oop = false;

    for (int leg = 0; leg < 2; ++leg) {
        int ai = (leg == 0) ? Config::KP_L_ANKLE : Config::KP_R_ANKLE;
        float ax, ay;

        if (kp[ai][2] > Config::KP_CONF) {
            ax = kp[ai][0]; ay = kp[ai][1];
            mem.update(leg, ax, ay);
        } else if (mem.get(leg, ax, ay)) {
            // Use last known position during brief occlusion
        } else {
            continue;
        }

        // Hysteresis: entering requires being inside eroded ROI;
        //             exiting requires being outside dilated ROI
        const auto& roi = prev_state ? exit_roi : enter_roi;
        if (insideROI(roi, ax, ay)) oop = true;
    }

    // Secondary: knee above hip (raised-leg posture) — both legs
    for (int leg = 0; leg < 2; ++leg) {
        int ki = (leg == 0) ? Config::KP_L_KNEE : Config::KP_R_KNEE;
        int hi = (leg == 0) ? Config::KP_L_HIP  : Config::KP_R_HIP;
        if (kp[ki][2] > Config::KP_CONF && kp[hi][2] > Config::KP_CONF)
            if (kp[ki][1] < kp[hi][1] - 20.0f) oop = true;
    }

    return oop;
}

// ─────────────────────────────────────────────────────────────────────────────
// Top-level per-frame decision
// ─────────────────────────────────────────────────────────────────────────────
bool classifyFrame(const Detection* psg, bool prev_state, AnkleMemory& mem) {
    if (!psg) return false;

    if (Config::USE_LEARNED_HEAD) {
        // Use learned head when ankle confidence is sufficient
        float ac = std::max(psg->keypoints[Config::KP_L_ANKLE][2],
                            psg->keypoints[Config::KP_R_ANKLE][2]);
        if (ac > Config::KP_CONF) {
            auto feats = extractFeatures(psg->keypoints);
            return classifyLearned(feats);
        }
        // Fall through to rule if both ankles low-confidence
    }
    return classifyRule(psg, prev_state, mem);
}

// ─────────────────────────────────────────────────────────────────────────────
// Drawing helpers
// ─────────────────────────────────────────────────────────────────────────────
static const cv::Scalar GREEN(0, 220, 90);
static const cv::Scalar RED  (0, 50, 220);
static const cv::Scalar ORNG (0, 140, 255);
static const cv::Scalar BLUE (200, 80, 0);
static const cv::Scalar WHITE(220, 220, 220);

void drawAnnotations(cv::Mat& frame,
                     const std::vector<Detection>& dets,
                     const Detection* passenger,
                     bool label) {

    // Dashboard ROI polygon (dilated = full visible zone)
    auto vis_roi = morphROI(Config::DASHBOARD_ROI, Config::ROI_DILATE_PX);
    cv::polylines(frame, {vis_roi}, true, GREEN, 2, cv::LINE_AA);

    // All detections (thin box)
    for (const auto& d : dets) {
        cv::Rect r(d.bbox);
        cv::rectangle(frame, r, BLUE, 1);
    }

    // Passenger box and keypoints
    if (passenger) {
        cv::Rect r(passenger->bbox);
        cv::rectangle(frame, r, label ? RED : GREEN, 2);

        // Draw key keypoints
        static const int KEY_KP[] = {
            Config::KP_L_ANKLE, Config::KP_R_ANKLE,
            Config::KP_L_KNEE,  Config::KP_R_KNEE,
            Config::KP_L_HIP,   Config::KP_R_HIP
        };
        for (int ki : KEY_KP) {
            const auto& kp = passenger->keypoints[ki];
            if (kp[2] < 0.2f) continue;
            int g = static_cast<int>(kp[2] * 220);
            cv::circle(frame, cv::Point(kp[0], kp[1]), 6,
                       cv::Scalar(0, g, 255 - g), -1, cv::LINE_AA);
        }
    }

    // Label banner
    std::string txt = label
        ? "POSITIVE  -  Feet on Dashboard"
        : "NEGATIVE";
    cv::Scalar bar_col = label ? cv::Scalar(0, 0, 160) : cv::Scalar(20, 100, 20);
    cv::rectangle(frame, cv::Point(18, 18), cv::Point(570, 72), bar_col, -1);
    cv::putText(frame, txt, cv::Point(28, 56),
                cv::FONT_HERSHEY_SIMPLEX, 1.15,
                label ? RED : GREEN, 2, cv::LINE_AA);
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-video pipeline
// ─────────────────────────────────────────────────────────────────────────────
void processVideo(IDetector& detector,
                  const std::string& input_path,
                  const std::string& output_path) {

    cv::VideoCapture cap(input_path);
    if (!cap.isOpened())
        throw std::runtime_error("Cannot open: " + input_path);

    int    W   = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int    H   = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int    total= static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    // Build undistort maps once per video (same W/H)
    cv::Mat map1, map2;
    buildUndistortMaps(W, H, map1, map2);

    // Create output directory if needed
    fs::create_directories(fs::path(output_path).parent_path());

    cv::VideoWriter writer(output_path,
                           cv::VideoWriter::fourcc('m','p','4','v'),
                           fps, cv::Size(W, H));
    if (!writer.isOpened())
        throw std::runtime_error("Cannot write: " + output_path);

    Smoother    smoother(Config::WINDOW_SIZE);
    AnkleMemory mem(Config::OCCLUSION_HOLD);

    cv::Mat raw, undist;
    bool prev_state = false;
    int  frame_idx  = 0;

    std::cout << "  [" << fs::path(input_path).filename().string()
              << "] frames=" << total << " fps=" << fps << "\n";

    while (cap.read(raw)) {
        // 1. Undistort
        cv::remap(raw, undist, map1, map2, cv::INTER_LINEAR);

        // 2. Detect
        auto dets = detector.detect(undist);

        // 3. Select passenger
        mem.tick();
        const Detection* psg = selectPassenger(dets);

        // 4. Classify
        bool raw_label  = classifyFrame(psg, prev_state, mem);

        // 5. Smooth
        bool smooth_label = smoother.update(raw_label);
        prev_state = smooth_label;

        // 6. Annotate + write
        drawAnnotations(undist, dets, psg, smooth_label);
        writer.write(undist);

        if (++frame_idx % 100 == 0)
            std::cout << "    frame " << frame_idx << "/" << total << "\r"
                      << std::flush;
    }
    std::cout << "    Done: " << total << " frames → " << output_path << "\n";
    cap.release();
    writer.release();
}
