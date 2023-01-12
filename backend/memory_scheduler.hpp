#pragma once

#include <string>
#include <vector>
#include <atomic>
#include <mutex>

#include "includes/context.hpp"
#include "includes/backend.hpp"
#include "includes/memory_status.hpp"
#include "includes/memory_event.hpp"
#include "includes/memory_schedule_event.hpp"
#include "includes/exceptions.hpp"
#include "backend/events.hpp"

namespace mori {

struct MemoryScheduler {
protected:
    const Context& context;
    status::MemoryStatus& status;
    events::Events& events;

    events::ScheduleEvents schedule_events;

public:
    MemoryScheduler(const Context& _context, status::MemoryStatus& _status, events::Events& _events): context(_context), status(_status), events(_events) {}

    /**
     * Action when the scheduling is triggered.
    */
    virtual void onSchedule() = 0;

    /**
     * Action when new memory event is submitted.
     * @param event The submitted event
     */
    virtual void onMemoryEvent(const events::MemoryEvent& event) = 0;

    /**
     * Action when an iteration starts.
    */
    virtual void onNewIteration() = 0;
    
    inline events::ScheduleEvents getScheduleEvents() { return schedule_events; }

    void submitEvent(events::MemoryEvent event) { onMemoryEvent(event); }

    void newIteration() { onNewIteration(); }

    virtual ~MemoryScheduler() = default;
    
};  // struct MemoryScheduler

struct FIFOMemoryScheduler : public MemoryScheduler {
    bool decided = false;

    FIFOMemoryScheduler(const Context& _context, status::MemoryStatus& _status, events::Events& _events): MemoryScheduler(_context, _status, _events) {}

    virtual void onSchedule() override {

    }
    virtual void onMemoryEvent(const events::MemoryEvent& event) override {}
    virtual void onNewIteration() override {
        if (events.getIteration() == 1) return;
        auto iter_1_res = events.select().where([](const events::EventSet::item& item) {
            return item.first == 1 && item.second.stage == ApplicationStage::forward;
        }).get();
        auto swapout_res = iter_1_res.select().where([](const events::EventSet::item& item) {
            return item.second.type == events::MemoryEventType::swapout;
        }).get();

        if (!decided) {
            for (auto &x : swapout_res.ref()) {
                // Create schedule event here;
                std::string target_tensor = x->second.tensor;
                // Get the last access of this tensor
                auto target_tensor_res = iter_1_res.select().where([x](const events::EventSet::item& item) {
                    return item.second.tensor == x->second.tensor;
                }).get();

                auto op = (*--target_tensor_res.ref().end())->second.op;
                schedule_events.forward_schedule_events.execution.emplace_back("", x->second.tensor, x->second.size, events::ScheduleEventType::swapout, "o1");
            }

            decided = true;
        }
    }

    virtual ~FIFOMemoryScheduler() = default;

};  // struct FIFOMemoryScheduler

struct DependencyAwareMemoryScheduler: public MemoryScheduler {
    DependencyAwareMemoryScheduler(const Context& _context, status::MemoryStatus& _status, events::Events& _events): MemoryScheduler(_context, _status, _events) {}

    virtual void onSchedule() override {}
    virtual void onMemoryEvent(const events::MemoryEvent& event) override {}
    virtual void onNewIteration() override {}

    virtual ~DependencyAwareMemoryScheduler() = default;

};  // struct DependencyAwareMemoryScheduler

struct MaximumSizePriorityMemoryScheduler: public MemoryScheduler {
    MaximumSizePriorityMemoryScheduler(const Context& _context, status::MemoryStatus& _status, events::Events& _events): MemoryScheduler(_context, _status, _events) {}

    virtual void onSchedule() override {}
    virtual void onMemoryEvent(const events::MemoryEvent& event) override {}
    virtual void onNewIteration() override {}

    virtual ~MaximumSizePriorityMemoryScheduler() = default;
};  // struct MaximumSizePriorityMemoryScheduler

}   // namespace mori