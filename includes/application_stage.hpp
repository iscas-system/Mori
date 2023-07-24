#pragma once

namespace mori {

enum struct ApplicationStage {
    all, forward, backward
};  // enum struct ApplicationStage

namespace utils {

static std::string get_application_stage_str(ApplicationStage stage) {
    switch (stage) {
        case ApplicationStage::all:      return "all";
        case ApplicationStage::forward:  return "forward";
        case ApplicationStage::backward: return "backward";
    }
    return "";
}

}   // namespace utils
}   // namespace mori