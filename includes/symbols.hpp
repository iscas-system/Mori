#pragma once

namespace mori {

enum struct ApplicationStage {
    all, forward, backward, update
};  // enum struct ApplicationStage

enum struct Direction {
    prev, post
};  // enum struct Direction

namespace utils {

static std::string get_application_stage_str(ApplicationStage stage) {
    switch (stage) {
        case ApplicationStage::all:      return "all";
        case ApplicationStage::forward:  return "forward";
        case ApplicationStage::backward: return "backward";
        case ApplicationStage::update:   return "update";
    }
    return "";
}

}   // namespace utils
}   // namespace mori