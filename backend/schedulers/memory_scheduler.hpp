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
#include "backend/events.hpp"
#include "backend/decisions/model.hpp"

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
private:
    bool model_decided = false;
    bool event_decided = false;

    decisions::Model model;

protected:
    void analyze() {
        model.setMemoryInfo(status.getMemoryInfo());
        model.analyze(status);
        schedule_events.memory_map = model.getMemoryMap();

        model_decided = true;
    }

public:
    FIFOMemoryScheduler(const Context& _context, status::MemoryStatus& _status, events::Events& _events): MemoryScheduler(_context, _status, _events) {}

    virtual void onSchedule() override {
        return;
        if (!model_decided) analyze();
        if (model.getLayerCount() == 1) return;

        // Prepare memory events
        auto iter_1_res = events.select().where([](const events::EventSet::item& item) {
            return item.first == 1;
        }).get();

        if (iter_1_res.size() == 0) return;

        auto iter_1_forward_res = iter_1_res.select().where([](const events::EventSet::item& item) {
            return item.second.stage == ApplicationStage::forward;
        }).get();

        // Generate swapout events based on analysis model.
        std::vector<status::TensorPres> tensors_swap;
        std::unordered_map<std::string, std::vector<events::ScheduleEvent>> events_swapout;
        for (auto &l : model.getLayers()) {
            for (auto &s : l.regions) {
                const decisions::Node& node = model.getMemoryNode(s);
                if (node.posts.empty()) continue;   // No need to swapout tensor.
                status::TensorPres tensor = status.referenceTensor(s);
                tensors_swap.emplace_back(std::move(tensor));
                for (auto &x : node.region.sections) {
                    // Generate copyout and freehost events.
                    auto iter_1_tensor_forward_res = iter_1_forward_res.select().where([s](const events::EventSet::item& item) {
                        return item.second.tensor == s;
                    }).get();

                    std::string last_acquired;
                    std::string last_assigned;
                    for (auto &y : iter_1_tensor_forward_res.ref()) {
                        switch (y->second.type) {
                            case events::MemoryEventType::read:
                                last_acquired = y->second.op;
                                break;
                            case events::MemoryEventType::write:
                            case events::MemoryEventType::access:
                                last_assigned = y->second.op;
                                break;
                            default:
                                break;
                        }
                    }
                    
                    // events_swapout[last_assigned].emplace_back(tensor.getOperatorName(), s, x, events::ScheduleEventType::copyout, last_assigned);
                    events_swapout[last_acquired].emplace_back(tensor.getOperatorName(), s, x, events::ScheduleEventType::swapout, last_acquired);
                }
            }
        }

        for (auto &s : status.getExecutionOrder()) {
            if (events_swapout.find(s) == events_swapout.end()) continue;
            for (auto &x : events_swapout.at(s)) {
                schedule_events.forward_schedule_events.execution.push_back(x);
            }
        }

        auto iter_1_backward_res = iter_1_res.select().where([](const events::EventSet::item& item) {
            return item.second.stage == ApplicationStage::backward;
        }).get();

        for (auto &x : tensors_swap) {
            // Get the first access of this tensor in backword stage
            auto target_tensor_backward_res = iter_1_backward_res.select().where([&x](const events::EventSet::item& item) {
                return item.second.tensor == x.getName() && item.second.type != events::MemoryEventType::swapin && item.second.type != events::MemoryEventType::swapout;
            }).get();

            auto opb = (*target_tensor_backward_res.ref().begin())->second.op;
            for (int i = 0; i < 1; ++i) {
                opb = status.getExecutionPost(opb);
                if (opb == "") break;
            }

            // Generate swapin event
            if (opb != "")
                schedule_events.backward_schedule_events.execution.emplace(schedule_events.backward_schedule_events.execution.begin(), x.getOperatorName(), x.getName(), x.getSize(), events::ScheduleEventType::swapin, opb); 
        }

        event_decided = true;
    }

    virtual void onMemoryEvent(const events::MemoryEvent& event) override {}

    virtual void onNewIteration() override {
        onSchedule();
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