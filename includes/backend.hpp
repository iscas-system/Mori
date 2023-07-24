#pragma once

#include "includes/memory_event.hpp"
#include "includes/execution_event.hpp"
#include "includes/memory_schedule_event.hpp"
#include "includes/memory_status.hpp"

namespace mori {

// Abstract struct
struct Backend {
    virtual void init() = 0;

    virtual void submitMemoryStatus(const status::MemoryStatus& status) = 0;

    virtual void start() {}

    virtual void setIteration(int _iteration) = 0;
    virtual void newIteration() = 0;
    virtual void halfIteration() = 0;

    virtual void submitEvent(const events::MemoryEvent& event) = 0;
    virtual void submitEvent(const events::ExecutionEvent& event) = 0;
    virtual events::ScheduleEvents getScheduleEvents() = 0;

    virtual void stop() {}

    virtual void terminate() = 0;
    
    virtual ~Backend() {};
};  // struct Backend

}   // namespace mori