#pragma once

#include "memory_event.hpp"
#include "memory_schedule_event.hpp"
#include "memory_status.hpp"

namespace mori {

// Abstract struct
struct Backend {
    virtual void init() = 0;
    virtual void registerOperator(const OperatorStatus& operator_status) = 0;
    virtual void submitEvent(const MemoryEvent& event) = 0;
    virtual std::vector<ScheduleEvent> getScheduleEvents() = 0;
    virtual void unregisterOperator(const std::string& op) = 0;
    virtual void terminate() = 0;
    virtual ~Backend() {};
};  // struct Backend

}   // namespace mori