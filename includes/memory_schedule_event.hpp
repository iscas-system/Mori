#pragma once

#include "includes/stdlibs.hpp"

namespace mori {

enum class ScheduleEventType {
    allocate, 
    copyin, copyout, 
    swapin, swapout, 
    freedev, freehost, 
    free
};  // enum ScheduleEventType

struct ScheduleEvent {
    std::string operator_name;
    std::string tensor_name;

    int interval;
    ScheduleEventType type;

    ScheduleEvent(): operator_name(""), tensor_name(""), interval(0), type(ScheduleEventType::allocate) {}
    ScheduleEvent(const std::string& _operator_name, const std::string& _tensor_name, int _interval, ScheduleEventType _type): operator_name(_operator_name), tensor_name(_tensor_name), interval(_interval), type(_type) {}
};  // struct ScheduleEvent

}   // namespace mori