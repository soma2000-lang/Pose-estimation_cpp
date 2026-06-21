#pragma once
#include <deque>
#include <array>
#include <numeric>

//
// Temporal majority-vote smoother:  outputs POSITIVE only when more than
// half of the last WINDOW_SIZE frames are POSITIVE.
// Also provides AnkleMemory: holds last N confident ankle positions so that
// brief dashboard-occlusion of the ankle does not kill the detection.
// ─────────────────────────────────────────────────────────────────────────────

class Smoother {
public:
    explicit Smoother(int window_size) : N_(window_size) {}

    // Call once per frame.  Returns smoothed POSITIVE/NEGATIVE.
    bool update(bool raw) {
        window_.push_back(raw);
        if (static_cast<int>(window_.size()) > N_) window_.pop_front();
        int pos = std::count(window_.begin(), window_.end(), true);
        return pos > static_cast<int>(window_.size()) / 2;
    }

    void reset() { window_.clear(); }

private:
    int N_;
    std::deque<bool> window_;
};


// Remembers the last confident (x,y) position of each ankle for
// OCCLUSION_HOLD frames.
class AnkleMemory {
public:
    explicit AnkleMemory(int hold_frames) : hold_(hold_frames) {
        age_.fill(999);
        pos_.fill({0.0f, 0.0f});
    }

    // Record a new good observation for leg index (0=left,1=right).
    void update(int idx, float x, float y) {
        pos_[idx] = {x, y};
        age_[idx] = 0;
    }

    // Tick time forward each frame for all tracked ankles.
    void tick() {
        for (auto& a : age_) if (a < 999) ++a;
    }

    // Returns true and fills (x,y) if the ankle was seen recently.
    bool get(int idx, float& x, float& y) const {
        if (age_[idx] >= hold_) return false;
        x = pos_[idx][0];
        y = pos_[idx][1];
        return true;
    }

private:
    int hold_;
    std::array<int, 2>                  age_;
    std::array<std::array<float,2>, 2>  pos_;
};
