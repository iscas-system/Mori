#pragma once

#include <string>
#include <atomic>
#include <mutex>
#include <iostream>
#include <condition_variable>

#include "../includes/backend.hpp"
#include "../includes/memory_event.hpp"
#include "../includes/memory_schedule_event.hpp"

namespace mori {

struct MemoryScheduler {
    Backend* backend;

    bool inited = false;

    MemoryScheduler() {}

    void setBackend(Backend* _backend) {
        if (inited) return;
        backend = _backend;
    }

    virtual void init() {
        if (inited) throw std::exception();
        inited = true;
    }

    /**
     * isActiveScheduler
     * Inform that if the scheduler activelly schedule the memory swapping
     * Active scheduler realtime automatically triggers memory swapping, while proactive scheduler only responds to memory events.
     * @return if the scheduler is an active scheduler.
     */
    virtual bool isActiveScheduler() {
        return false;
    }

    virtual void schedule() {
        if (false) {
            int curr_sched_iteration = 0;
            {
            }

            // Do scheduling here.
        }
    }

    /**
     * onMemoryEvent
     * Action when new memory event is submitted.
     */
    virtual void onMemoryEvent(const MemoryEvent& event) {
        switch(event.type) {
            case allocate:
                // Proactive memory scheduler only need to schedule when memory is insufficient.
                break;
            default:
                break;
        }
    }

    virtual void onIncreaseIteration() {
    }
    
    virtual std::vector<ScheduleEvent> getScheduleEvents() {
        return std::vector<ScheduleEvent>();
    }

    virtual void submitEvent(MemoryEvent event) {
        if (!inited) throw std::exception();
        onMemoryEvent(event);
    }

    virtual void increaseIteration() {
        if (!inited) throw std::exception();
        onIncreaseIteration();
    }

    virtual void terminate() {
        if (!inited) throw std::exception();
        inited = false;
    }

    virtual ~MemoryScheduler() {
        if (inited) terminate();
        backend = nullptr;
    }
    
};  // struct MemoryScheduler

struct FIFOMemoryScheduler : public MemoryScheduler {
    virtual void init() {MemoryScheduler::init();}

    virtual void schedule() {}

    virtual void terminate() {}

    virtual ~FIFOMemoryScheduler() = default;

};  // struct FIFOMemoryScheduler

struct DependencyAwareMemoryScheduler: public MemoryScheduler {
    virtual void init() {MemoryScheduler::init();}

    virtual void schedule() {}

    virtual void terminate() {}

    virtual ~DependencyAwareMemoryScheduler() = default;

};  // struct DependencyAwareMemoryScheduler

struct MaximumSizePriorityMemoryScheduler: public MemoryScheduler {
    virtual void init() {MemoryScheduler::init();}

    virtual void schedule() {}

    virtual void terminate() {}

    virtual ~MaximumSizePriorityMemoryScheduler() = default;
};  // struct MaximumSizePriorityMemoryScheduler

struct RWAwareMemoryScheduler: public MemoryScheduler {
    virtual void init() {MemoryScheduler::init();}

    virtual void schedule() {}

    virtual void terminate() {}

    virtual ~RWAwareMemoryScheduler() = default;

};  // struct RWAwareMemoryScheduler

}   // namespace mori