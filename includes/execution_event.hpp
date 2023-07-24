#pragma once

#include <string>
#include <chrono>

#include "includes/application_stage.hpp"
#include "includes/event_utils.hpp"

namespace mori {
namespace events {

enum struct ExecutionEventType {
    request, release, execution
};  // enum struct ExecutionEventType

namespace utils {
    static std::string get_event_type_str(ExecutionEventType type) {
        switch (type) {
            case ExecutionEventType::request:
                return "request";
            case ExecutionEventType::release:
                return "release";
            default:
                return "execution";
        }
    }
}   // namespace utils

struct ExecutionEvent {
    std::string op;
    ExecutionEventType type;
    ApplicationStage   stage;
    std::chrono::steady_clock::time_point timestamp;

    ExecutionEvent() {
        op = "";
        type = ExecutionEventType::execution;
        stage = ApplicationStage::all;
        timestamp = std::chrono::steady_clock::now();
    }

    ExecutionEvent(const std::string& _op, ExecutionEventType _type, ApplicationStage _stage, const std::chrono::steady_clock::time_point& _timestamp) {
        op = _op;
        type = _type;
        stage = _stage;
        timestamp = _timestamp;
    }

    ExecutionEvent(const std::string& _op, ExecutionEventType _type, ApplicationStage _stage) {
        op = _op;
        type = _type;
        stage = _stage;
        timestamp = std::chrono::steady_clock::now();
    }

    ExecutionEvent(const ExecutionEvent& event) = default;
    ExecutionEvent& operator=(const ExecutionEvent& event) = default;

    bool operator<(const ExecutionEvent& event) const {return timestamp < event.timestamp;}

    operator std::string() const {
        std::stringstream ss;
        ss<<"Timestamp: "<<mori::utils::get_timestamp_val(timestamp)<<" operator: "<<op<<" type: "<<utils::get_event_type_str(type)<<" stage: "<<mori::utils::get_application_stage_str(stage);
        return ss.str();
    }
};  // struct ExecutionEvent

static Logger& operator<<(Logger& logger, const ExecutionEvent& event) {
    logger << static_cast<std::string>(event);
    return logger;
}

}   // namespace events
}   // namespace mori