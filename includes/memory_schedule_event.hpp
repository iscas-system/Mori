#pragma once

#include <vector>

namespace mori {
namespace events {

enum class ScheduleEventType {
    allocate, 
    copyin, copyout, 
    swapin, swapout, 
    freedev, freehost, 
    free
};  // enum class ScheduleEventType

struct ScheduleEvent {
    std::string operator_name = "";
    std::string tensor_name   = "";
    size_t      size          = 0;

    ScheduleEventType type  = ScheduleEventType::allocate;
    std::string postop = "";    // For execution-triggered events, the event should be executed after executing postop.
    int timepoint = 0;          // For timepoing-triggered events, the event should be executed after specificied timepoint.

    ScheduleEvent() = default;
    ScheduleEvent(const std::string& _op_name, const std::string& _tensor_name, size_t _size): operator_name(_op_name), tensor_name(_tensor_name), size(_size) {}
    ScheduleEvent(const std::string& _op_name, const std::string& _tensor_name, size_t _size, ScheduleEventType _event_type, const std::string& _postop): operator_name(_op_name), tensor_name(_tensor_name), size(_size), type(_event_type), postop(_postop) {}
    ScheduleEvent(const std::string& _op_name, const std::string& _tensor_name, size_t _size, ScheduleEventType _event_type, int _timepoint): operator_name(_op_name), tensor_name(_tensor_name), size(_size), type(_event_type), timepoint(_timepoint) {}
};  // struct ScheduleEvent

struct StageScheduleEvents {
    std::vector<ScheduleEvent> execution;
    std::vector<ScheduleEvent> timepoint;
};  // struct StageScheduleEvents

struct ScheduleEvents {
    StageScheduleEvents forward_schedule_events;
    StageScheduleEvents backward_schedule_events;
};  // struct ScheduleEvents

}   // namespace events
}   // namespace mori