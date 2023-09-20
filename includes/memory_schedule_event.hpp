#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <sstream>

#include "includes/memory_layout.hpp"

namespace mori {
namespace events {

enum struct ScheduleEventType {
    allocate, 
    copyin, copyout, 
    swapin, swapout, 
    freedev, freehost, 
    free
};  // enum struct ScheduleEventType

namespace utils {

static std::string get_schedule_event_type_str(ScheduleEventType type) {
    switch (type) {
        case ScheduleEventType::allocate:
            return "allocate";
        case ScheduleEventType::copyin:
            return "copyin";
        case ScheduleEventType::copyout:
            return "copyout";
        case ScheduleEventType::swapin:
            return "swapin";
        case ScheduleEventType::swapout:
            return "swapout";
        case ScheduleEventType::freedev:
            return "freedev";
        case ScheduleEventType::freehost:
            return "freehost";
        case ScheduleEventType::free:
            return "free";
        default:
            break;
    }
    assert(0);
    return "";
}

}   // namespace utils

struct ScheduleEvent final {
    std::string operator_name = "";
    std::string tensor_name   = "";
    size_t      size          = 0;

    ScheduleEventType type  = ScheduleEventType::allocate;
    std::string postop = "";    // For execution-triggered events, the event should be executed after executing postop.
    long timepoint = 0;          // For timepoing-triggered events, the event should be executed after specificied timepoint.

    bool instant = false;

    ScheduleEvent() = default;
    ScheduleEvent(const std::string& _op_name, const std::string& _tensor_name, size_t _size): operator_name(_op_name), tensor_name(_tensor_name), size(_size) {}
    ScheduleEvent(const std::string& _op_name, const std::string& _tensor_name, size_t _size, ScheduleEventType _event_type, const std::string& _postop, bool _instant = false): operator_name(_op_name), tensor_name(_tensor_name), size(_size), type(_event_type), postop(_postop), instant(_instant) {}
    ScheduleEvent(const std::string& _op_name, const std::string& _tensor_name, size_t _size, ScheduleEventType _event_type, long _timepoint, bool _instant = false): operator_name(_op_name), tensor_name(_tensor_name), size(_size), type(_event_type), timepoint(_timepoint), instant(_instant) {}
};  // struct ScheduleEvent

struct StageScheduleEvents {
    std::unordered_map<std::string, std::vector<ScheduleEvent>> execution;
    std::vector<ScheduleEvent> timepoint;
};  // struct StageScheduleEvents

struct ScheduleEvents {
    layout::MemoryMap memory_map;
    StageScheduleEvents forward_schedule_events;
    StageScheduleEvents backward_schedule_events;
};  // struct ScheduleEvents

}   // namespace events
}   // namespace mori