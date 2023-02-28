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
        // Do not schedule in the first iteration
        if (events.getIteration() == 1 || decided) return;
        auto iter_1_res = events.select().where([](const events::EventSet::item& item) {
            return item.first == 1;
        }).get();
        
        // The total unmet memory requirement.
        auto swapout_res = iter_1_res.select().where([](const events::EventSet::item& item) {
            return item.second.type == events::MemoryEventType::swapout;
        }).get();

        size_t unmet_memory_requirement = 0;
        size_t released_memory_requirement = 0;
        for (auto &x : swapout_res.ref()) {
            // Create schedule event here;
            unmet_memory_requirement += x->second.size;
        }

        // Tensors to swapout.
        std::vector<status::TensorPres> tensors_swap;
        for (auto &op_name : status.getExecutionOrder()) {
            status::OperatorPres op_pres = status.referenceOperator(op_name);
            // Forward propagation and backward propagation share the same set of operators.
            if (op_pres.isBackwardPropagation()) continue;

            for (auto &tensor_name : op_pres.getTensors()) { 
                status::TensorPres tensor_pres = status.referenceTensor(tensor_name);
                // Do not swap out persistant tensors.
                if (tensor_pres.isPersistant()) continue;
                released_memory_requirement += tensor_pres.getSize();
                tensors_swap.emplace_back(std::move(tensor_pres));
                if (unmet_memory_requirement <= released_memory_requirement) break;
            }
            if (unmet_memory_requirement <= released_memory_requirement) break;
        }

        // Generate swap events.
        released_memory_requirement = 0;
        for (auto &x : tensors_swap) {
            // Get the last access of this tensor in forward stage.
            auto target_tensor_forward_res = iter_1_res.select().where([&x](const events::EventSet::item& item) {
                return item.second.stage == ApplicationStage::forward && item.second.tensor == x.getName() && item.second.type != events::MemoryEventType::swapin && item.second.type != events::MemoryEventType::swapout;
            }).get();

            auto opf = (*target_tensor_forward_res.ref().rbegin())->second.op;
            size_t tensor_swap_size = x.getSize();
            // Decide the swapout size of each tensor.
            if (released_memory_requirement + x.getSize() > unmet_memory_requirement)
                tensor_swap_size = unmet_memory_requirement - released_memory_requirement;

            // Generate swapout event
            schedule_events.forward_schedule_events.execution.emplace_back("", x.getName(), tensor_swap_size, events::ScheduleEventType::swapout, opf);
            released_memory_requirement += tensor_swap_size;

            // Get the first access of this tensor in backword stage
            auto target_tensor_backward_res = iter_1_res.select().where([&x](const events::EventSet::item& item) {
                return item.second.stage == ApplicationStage::backward && item.second.tensor == x.getName() && item.second.type != events::MemoryEventType::swapin && item.second.type != events::MemoryEventType::swapout;
            }).get();

            auto opb = (*target_tensor_backward_res.ref().begin())->second.op;
            for (int i = 0; i < 1; ++i) {
                opb = status.getExecutionPost(opb);
                if (opb == "") break;
            }

            // Generate swapin event
            if (opb != "")
                schedule_events.backward_schedule_events.execution.emplace(schedule_events.backward_schedule_events.execution.begin(), "", x.getName(), tensor_swap_size, events::ScheduleEventType::swapin, opb); 

            if (unmet_memory_requirement <= released_memory_requirement) break;
        }

        decided = true;
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