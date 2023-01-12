#pragma once

namespace mori {

enum class ApplicationStage {
    all, forward, backward
};  // enum class ApplicationStage

namespace util {

static std::string get_application_stage_str(ApplicationStage stage) {
    switch (stage) {
        case ApplicationStage::all:      return "all";
        case ApplicationStage::forward:  return "forward";
        case ApplicationStage::backward: return "backward";
    }
    return "";
}

}   // namespace util
}   // namespace mori