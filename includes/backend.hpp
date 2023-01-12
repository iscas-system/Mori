#pragma once

#include "includes/memory_event.hpp"
#include "includes/memory_schedule_event.hpp"
#include "includes/memory_status.hpp"

namespace mori {

// Abstract struct
struct Backend {
    virtual void init() = 0;

    virtual void registerTensor(const status::Tensor&) = 0;
    virtual void registerOperator(const status::Operator& operator_status) {}
    virtual void setEntry(const std::string& _op) {}

    virtual void start() {}

    virtual void setIteration(int _iteration) = 0;
    virtual void newIteration() = 0;
    virtual void halfIteration() = 0;

    virtual void submitEvent(const events::MemoryEvent& event) = 0;
    virtual events::ScheduleEvents getScheduleEvents() = 0;

    virtual void stop() {}

    virtual void unregisterTensor(const std::string&) = 0;
    virtual void unregisterOperator(const std::string& op) {}

    virtual void terminate() = 0;
    
    virtual ~Backend() {};
};  // struct Backend

}   // namespace mori