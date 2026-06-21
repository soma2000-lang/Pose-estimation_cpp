#include "detector_onnx.hpp"
#include "pipeline.hpp"
#include "config.hpp"
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// main.cpp
//
// Usage:  ./oop_detect <input_dir>  <output_dir>
//
//   input_dir  must contain profile1/, profile2/, profile3/
//   output_dir will be created automatically if it does not exist
//
// Example (run from the solution/ directory):
//   ./build/oop_detect ../input ../output
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_dir> <output_dir>\n";
        return 1;
    }

    fs::path input_root  = argv[1];
    fs::path output_root = argv[2];

    if (!fs::exists(input_root)) {
        std::cerr << "Input directory not found: " << input_root << "\n";
        return 1;
    }
    fs::create_directories(output_root);

    // Print active config summary
    std::cout << "═══════════════════════════════════════════\n";
    std::cout << "  OOP Feet-on-Dashboard Detector\n";
    std::cout << "  Model:   " << Config::MODEL_PATH << "\n";
    std::cout << "  Smoother window: " << Config::WINDOW_SIZE << " frames\n";
    std::cout << "  Learned head:    "
              << (Config::USE_LEARNED_HEAD ? "ON" : "OFF (rule-based)") << "\n";
    std::cout << "═══════════════════════════════════════════\n";

    OnnxDetector detector(Config::MODEL_PATH);

    int n_processed = 0;

    // Walk profile1/, profile2/, profile3/
    for (const auto& profile : fs::directory_iterator(input_root)) {
        if (!profile.is_directory()) continue;

        std::string prof_name = profile.path().filename().string();
        fs::path out_prof = output_root / prof_name;
        fs::create_directories(out_prof);

        // Walk *.mp4 inside each profile
        for (const auto& entry : fs::directory_iterator(profile.path())) {
            if (entry.path().extension() != ".mp4") continue;

            fs::path out_vid = out_prof / entry.path().filename();
            std::cout << "\nProcessing: " << entry.path().string() << "\n";

            try {
                processVideo(detector,
                             entry.path().string(),
                             out_vid.string());
                ++n_processed;
            } catch (const std::exception& e) {
                std::cerr << "  ERROR: " << e.what() << "\n";
            }
        }
    }

    std::cout << "\n[Done] Processed " << n_processed << " video(s).\n";
    std::cout << "Output written to: " << output_root << "\n";
    return 0;
}
