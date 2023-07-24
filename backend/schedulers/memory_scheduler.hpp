#pragma once

#include <algorithm>
#include <string>
#include <vector>
#include <atomic>
#include <mutex>

#include "includes/context.hpp"
#include "includes/backend.hpp"
#include "includes/memory_status.hpp"
#include "includes/memory_event.hpp"
#include "includes/execution_event.hpp"
#include "includes/memory_schedule_event.hpp"
#include "backend/events.hpp"
#include "backend/decisions/layout_model.hpp"
#include "backend/decisions/time_model.hpp"

namespace mori {

struct MemoryScheduler {
protected:
    const Context& context;
    status::MemoryStatus& status;
    events::Events& events;

    events::ScheduleEvents schedule_events;

    std::atomic<int> current_iteration = 0;

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
     * Action when new execution event is submitted.
     * @param event The submitted event
     */
    virtual void onMemoryEvent(const events::ExecutionEvent& event) = 0;

    /**
     * Action when an iteration starts.
    */
    virtual void onNewIteration() = 0;
    
    inline events::ScheduleEvents getScheduleEvents() {
        onSchedule();
        return schedule_events;
    }

    void submitEvent(events::MemoryEvent event) { onMemoryEvent(event); }
    void submitEvent(events::ExecutionEvent event) { onMemoryEvent(event); }

    void newIteration() {
        ++current_iteration;
        onNewIteration();
    }

    virtual ~MemoryScheduler() = default;
    
};  // struct MemoryScheduler

struct FIFOMemoryScheduler : public MemoryScheduler {
private:
    bool layout_model_decided = false;
    bool event_decided        = false;

    decisions::LayoutModel layout_model;
    decisions::TimeModel   time_model;

protected:
    void analyzeLayoutModel() {
        layout_model.setMemoryInfo(status.getMemoryInfo());
        layout_model.analyze(status);
        schedule_events.memory_map = layout_model.getMemoryMap();

        layout_model_decided = true;
    }

    void initTimeModel() {
        auto iter_1_backward_res = events.from_execution_events().where([](const events::EventSet<events::ExecutionEvent>::item& item) {
            // return item.first == (current_iteration - 1);
            return item.first == 1 && item.second.stage == ApplicationStage::backward;
        }).get();
        // No memory events.
        if (iter_1_backward_res.empty()) return;

        auto iter_1_backward_request_res = iter_1_backward_res.select().where([](const events::EventSet<events::ExecutionEvent>::item& item) {
            return item.second.type == events::ExecutionEventType::request;
        }).get();
        auto iter_1_backward_release_res = iter_1_backward_res.select().where([](const events::EventSet<events::ExecutionEvent>::item& item) {
            return item.second.type == events::ExecutionEventType::release;
        }).get();

        std::unordered_map<std::string, long> request_timepoints;
        std::unordered_map<std::string, long> release_timepoints;

        for (auto &x : iter_1_backward_request_res.ref()) request_timepoints[x->second.op] = utils::get_timestamp_val(x->second.timestamp);
        for (auto &x : iter_1_backward_release_res.ref()) release_timepoints[x->second.op] = utils::get_timestamp_val(x->second.timestamp);
        assert(request_timepoints.size() == release_timepoints.size());

        for (auto &so : status.getExecutionOrder()) {
            status::OperatorPres operator_pres = status.referenceOperator(so);
            if (!operator_pres.isBackwardPropagation()) continue;
            if (request_timepoints.count(so) != 1 || release_timepoints.count(so) != 1) continue;
            
            decisions::TimeModel::Timespan timespan(so, release_timepoints.at(so) - request_timepoints.at(so));
            time_model.submitExecutionSynchronization(so);
            time_model.submitExecutionTimespan(so, timespan);
        }
    }

    void analyzeEvent() {
        // Step 1: Prepare events
        // Prepare memory events
        auto iter_1_mem_res = events.from_memory_events().where([](const events::EventSet<events::MemoryEvent>::item& item) {
            // return item.first == (current_iteration - 1);
            return item.first == 1;
        }).get();
        // No memory events.
        if (iter_1_mem_res.empty()) return;
        
        // Prepare execution events
        auto iter_1_exec_res = events.from_execution_events().where([](const events::EventSet<events::ExecutionEvent>::item& item) {
            // return item.first == (current_iteration - 1);
            return item.first == 1;
        }).get();
        // No memory events.
        if (iter_1_exec_res.empty()) return;
        
        // Step 2: Analyze swapout events
        auto iter_1_forward_mem_res = iter_1_mem_res.select().where([](const events::EventSet<events::MemoryEvent>::item& item) {
            return item.second.stage == ApplicationStage::forward;
        }).get();

        auto iter_1_forward_swapout_res = iter_1_forward_mem_res.select().where([](const events::EventSet<events::MemoryEvent>::item& item) {
            return item.second.type == events::MemoryEventType::swapout;
        }).get();
        // No need to swap.
        if (iter_1_forward_swapout_res.empty()) return;

        // Generate swapout events based on analysis model.
        // All tensors are accessed in forward propagation.
        std::unordered_set<std::string> tensors_swap;
        for (auto &l : layout_model.getLayers()) {
            for (auto &s : l.regions) {
                const decisions::LayoutModel::Node& node = layout_model.getMemoryNode(s);
                if (node.posts.empty()) continue;   // No need to swapout tensor.
                
                // Generate copyout and freehost events.
                auto iter_1_tensor_forward_res = iter_1_forward_mem_res.select().where([s](const events::EventSet<events::MemoryEvent>::item& item) {
                    return item.second.tensor == s;
                }).get();

                bool forward_event_generated = false;
                std::string last_acquired;
                std::string last_assigned;
                for (auto &y : iter_1_tensor_forward_res.ref()) {
                    switch (y->second.type) {
                        case events::MemoryEventType::allocate:
                            // Tensor allocated in forward propagation, swapout event should be generated.
                            forward_event_generated = true;
                        case events::MemoryEventType::read:
                            last_acquired = y->second.op;
                            break;
                        case events::MemoryEventType::write:
                        case events::MemoryEventType::access:
                            last_assigned = y->second.op;
                            break;
                        case events::MemoryEventType::free:
                            // Tensor released in forward propagation, swapout event should not be generated.
                            forward_event_generated = false;
                        default:
                            break;
                    }
                }

                if (!forward_event_generated) continue;

                tensors_swap.insert(s);                
                for (auto &x : node.region.sections) {
                    status::TensorPres pres = status.referenceTensor(s);
                    schedule_events.forward_schedule_events.execution[last_assigned].emplace_back(pres.getOperatorName(), s, x, events::ScheduleEventType::copyout, last_acquired);
                    schedule_events.forward_schedule_events.execution[last_acquired].emplace_back(pres.getOperatorName(), s, x, events::ScheduleEventType::swapout, last_acquired);
                }
            }
        }
        // Output: swapout events, tensors_swap

        // Step 3: Analyze swapin events
        auto iter_1_backward_mem_res = iter_1_mem_res.select().where([](const events::EventSet<events::MemoryEvent>::item& item) {
            return item.second.stage == ApplicationStage::backward;
        }).get();
        auto iter_1_backward_access_res = iter_1_backward_mem_res.select().where([](const events::EventSet<events::MemoryEvent>::item& item) {
            if (item.second.type == events::MemoryEventType::allocate) return false;
            if (item.second.type == events::MemoryEventType::free) return false;
            return item.second.type != events::MemoryEventType::swapin && item.second.type != events::MemoryEventType::swapout;
        }).get();

        initTimeModel();
        decisions::TransferringModel transferring_model;

        // for (auto &x : iter_1_backward_access_res.ref()) {
        //     if (tensors_swap.find(x->second.tensor) == tensors_swap.end()) continue;
            
        //     // Located the first access of a tensor
        //     for
        // }

        for (auto &s : status.getExecutionOrder()) {
            auto op_pres = status.referenceOperator(s);
            if (!op_pres.isBackwardPropagation()) continue;

            auto target_operator_backward_res = iter_1_backward_access_res.select().where([&s, &tensors_swap](const events::EventSet<events::MemoryEvent>::item& item) {
                if (item.second.op != s) return false;
                return tensors_swap.find(item.second.tensor) != tensors_swap.end();
            }).get();
            if (target_operator_backward_res.empty()) continue;

            for (auto &x : target_operator_backward_res.ref()) {
                status::TensorPres tensor_pres = status.referenceTensor(x->second.tensor);
                decisions::TimeModel::Timespan timespan(x->second.tensor, transferring_model.analyze(tensor_pres.getSize()));
                time_model.submitTransferringTimespan(s, timespan);
            }
            time_model.submitTransferringSynchronization(s);
            time_model.setSynchronizationEnabled(s);
        }

        time_model.analyze();

        for (auto &x : time_model.transferring_lane.timespans) {
            if (x.second.synchronization) continue;
            status::TensorPres pres = status.referenceTensor(x.second.target);
            // Generate swapin event
            schedule_events.backward_schedule_events.timepoint.emplace_back(pres.getOperatorName(), pres.getName(), pres.getSize(), events::ScheduleEventType::swapin, x.second.timepoint);
        }

        event_decided = true;
    }

public:
    FIFOMemoryScheduler(const Context& _context, status::MemoryStatus& _status, events::Events& _events): MemoryScheduler(_context, _status, _events) {}

    virtual void onSchedule() override {
        if (!layout_model_decided) analyzeLayoutModel();
        if (layout_model.getLayerCount() == 1) return;  // No need of memory swapping.
        if (!event_decided) analyzeEvent();
    }

    virtual void onMemoryEvent(const events::MemoryEvent& event) override {}
    virtual void onMemoryEvent(const events::ExecutionEvent& event) override {}

    virtual void onNewIteration() override {}

    virtual ~FIFOMemoryScheduler() = default;

};  // struct FIFOMemoryScheduler

// struct DependencyAwareMemoryScheduler: public MemoryScheduler {
//     DependencyAwareMemoryScheduler(const Context& _context, status::MemoryStatus& _status, events::Events& _events): MemoryScheduler(_context, _status, _events) {}

//     virtual void onSchedule() override {}
//     virtual void onMemoryEvent(const events::MemoryEvent& event) override {}
//     virtual void onMemoryEvent(const events::ExecutionEvent& event) override {}
//     virtual void onNewIteration() override {}

//     virtual ~DependencyAwareMemoryScheduler() = default;

// };  // struct DependencyAwareMemoryScheduler

// struct MaximumSizePriorityMemoryScheduler: public MemoryScheduler {
//     MaximumSizePriorityMemoryScheduler(const Context& _context, status::MemoryStatus& _status, events::Events& _events): MemoryScheduler(_context, _status, _events) {}

//     virtual void onSchedule() override {}
//     virtual void onMemoryEvent(const events::MemoryEvent& event) override {}
//     virtual void onMemoryEvent(const events::ExecutionEvent& event) override {}
//     virtual void onNewIteration() override {}

//     virtual ~MaximumSizePriorityMemoryScheduler() = default;
// };  // struct MaximumSizePriorityMemoryScheduler

}   // namespace mori