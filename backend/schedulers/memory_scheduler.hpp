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
    const Context::View& context;
    status::MemoryStatus& status;
    events::Events& events;

    events::ScheduleEvents schedule_events;

    std::atomic<int> current_iteration = 0;

protected:
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

public:
    MemoryScheduler(const Context::View& _context, status::MemoryStatus& _status, events::Events& _events): context(_context), status(_status), events(_events) {}
    
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

struct EventBasedMemoryScheduler : public MemoryScheduler {
protected:
    bool event_decided = false;

    virtual void preAnalyzeEvents()  = 0;
    virtual std::unordered_set<std::string> analyzeForwardEvents(const events::EventSet<events::MemoryEvent>&)               = 0;
    virtual void analyzeBackwardEvents(const events::EventSet<events::MemoryEvent>&, const std::unordered_set<std::string>&) = 0;
    virtual void postAnalyzeEvents() = 0;

    virtual void onSchedule() override {
        if (event_decided) return;

        auto iter_1_mem_res = events.from_memory_events().where([](const events::EventSet<events::MemoryEvent>::item& item) {
            // return item.first == (current_iteration - 1);
            return item.first == 1;
        }).get();
        if (iter_1_mem_res.empty()) return;

        auto iter_1_forward_mem_res = iter_1_mem_res.select().where([](const events::EventSet<events::MemoryEvent>::item& item) {
            return item.second.stage == ApplicationStage::forward;
        }).get();

        auto iter_1_backward_mem_res = iter_1_mem_res.select().where([](const events::EventSet<events::MemoryEvent>::item& item) {
            return item.second.stage == ApplicationStage::backward;
        }).get();

        preAnalyzeEvents();
        auto tensors_swapped = analyzeForwardEvents(iter_1_forward_mem_res);
        analyzeBackwardEvents(iter_1_backward_mem_res, tensors_swapped);
        postAnalyzeEvents();

        event_decided = true;
    }

    virtual void onMemoryEvent(const events::MemoryEvent& event) override {
        if (!event_decided) return;
        
        if (event.stage == ApplicationStage::forward && event.type == events::MemoryEventType::swapin) {
            // Indicate schedule error.
            assert(true);
        }
    }

    virtual void onMemoryEvent(const events::ExecutionEvent& event) override {}
    virtual void onNewIteration() override {}
public:
    EventBasedMemoryScheduler(const Context::View& _context, status::MemoryStatus& _status, events::Events& _events): MemoryScheduler(_context, _status, _events) {}
    virtual ~EventBasedMemoryScheduler() = default;
};  // struct EventBasedMemoryScheduler

struct FIFOMemoryScheduler : public EventBasedMemoryScheduler {
protected:
    virtual void preAnalyzeEvents() override  {}
    virtual std::unordered_set<std::string> analyzeForwardEvents(const events::EventSet<events::MemoryEvent>& iter_1_forward_mem_res) override {
        auto iter_1_forward_swapout_res = iter_1_forward_mem_res.select().where([](const events::EventSet<events::MemoryEvent>::item& item) {
            return item.second.type == events::MemoryEventType::swapout;
        }).get();
        // No need to swap.
        if (iter_1_forward_swapout_res.empty()) return std::unordered_set<std::string>();

        size_t unmet_memory_requirement = 0;
        size_t released_memory_requirement = 0;
        for (auto &x : iter_1_forward_swapout_res.ref()) unmet_memory_requirement += x->second.size;

        std::unordered_set<std::string> tensors_swapped;
        // Tensors to swapout.
        for (auto &s : status.getExecutionOrder()) {
            status::OperatorPres op_pres = status.referenceOperator(s);
            // Forward propagation and backward propagation share the same set of operators.
            if (op_pres.isBackwardPropagation()) continue;

            for (auto &tensor_name : op_pres.getTensors()) { 
                status::TensorPres tensor_pres = status.referenceTensor(tensor_name);
                // Do not swap out persistant tensors.
                if (tensor_pres.isPersistent() || tensor_pres.isTransient()) continue;

                // Get the last access of this tensor in forward stage.
                auto iter_1_tensor_forward_res = iter_1_forward_mem_res.select().where([&tensor_name](const events::EventSet<events::MemoryEvent>::item& item) {
                    return item.second.tensor == tensor_name && item.second.type != events::MemoryEventType::swapin && item.second.type != events::MemoryEventType::swapout;
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

                tensors_swapped.insert(tensor_name);
                // Generate swapout event
                // schedule_events.forward_schedule_events.execution[last_assigned].emplace_back(tensor_pres.getOperatorName(), tensor_pres.getName(), tensor_pres.getSize(), events::ScheduleEventType::copyout, last_assigned);
                schedule_events.forward_schedule_events.execution[last_acquired].emplace_back(tensor_pres.getOperatorName(), tensor_pres.getName(), tensor_pres.getSize(), events::ScheduleEventType::swapout, last_acquired);
                released_memory_requirement += tensor_pres.getSize();

                // if (unmet_memory_requirement <= released_memory_requirement) break;
            }
        }

        return tensors_swapped;
    }
    virtual void postAnalyzeEvents() override {}

public:
    FIFOMemoryScheduler(const Context::View& _context, status::MemoryStatus& _status, events::Events& _events): EventBasedMemoryScheduler(_context, _status, _events) {}
};  // struct FIFOMemoryScheduler

struct ExecutionTimeAwareMemoryScheduler : public FIFOMemoryScheduler {
protected:
    decisions::TimeModel         time_model;
    decisions::TransferringModel transferring_model;

    std::unordered_map<std::string, long> execution_timespans;

protected:
    virtual void preAnalyzeEvents() override {
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

        for (auto &s : status.getExecutionOrder()) {
            status::OperatorPres operator_pres = status.referenceOperator(s);
            if (!operator_pres.isBackwardPropagation()) continue;
            if (request_timepoints.count(s) != 1 || release_timepoints.count(s) != 1) continue;
            
            execution_timespans.emplace(s, release_timepoints.at(s) - request_timepoints.at(s));
            decisions::TimeModel::Timespan timespan(s, release_timepoints.at(s) - request_timepoints.at(s));
            time_model.submitExecutionSynchronization(s);
            time_model.submitExecutionTimespan(s, timespan);
        }
    }

public:
    ExecutionTimeAwareMemoryScheduler(const Context::View& _context, status::MemoryStatus& _status, events::Events& _events): FIFOMemoryScheduler(_context, _status, _events) { time_model.setStrongSynchronization(false); }
    virtual ~ExecutionTimeAwareMemoryScheduler() = default;
};  // struct ExecutionTimeAwareMemoryScheduler

struct SectionAwareMemoryScheduler : public ExecutionTimeAwareMemoryScheduler {
private:
    bool layout_model_decided = false;

    decisions::LayoutModel layout_model;

protected:
    void analyzeLayoutModel() {
        layout_model.setMemoryInfo(status.getMemoryInfo());
        layout_model.analyze(status);
        schedule_events.memory_map = layout_model.getMemoryMap();

        layout_model_decided = true;
    }

    virtual void preAnalyzeEvents() override {
        if (!layout_model_decided) analyzeLayoutModel();
        if (layout_model.getLayerCount() == 1) return;  // No need of memory swapping.
        ExecutionTimeAwareMemoryScheduler::preAnalyzeEvents();
    }

    virtual std::unordered_set<std::string> analyzeForwardEvents(const events::EventSet<events::MemoryEvent>& iter_1_forward_mem_res) override {
        std::unordered_set<std::string> tensors_swapped;

        auto iter_1_forward_swapout_res = iter_1_forward_mem_res.select().where([](const events::EventSet<events::MemoryEvent>::item& item) {
            return item.second.type == events::MemoryEventType::swapout;
        }).get();
        // No need to swap.
        if (iter_1_forward_swapout_res.empty()) return tensors_swapped;

        size_t unmet_memory_requirement = 0;
        size_t released_memory_requirement = 0;
        for (auto &x : iter_1_forward_swapout_res.ref()) unmet_memory_requirement += x->second.size;

        // Generate swapout events based on analysis model.
        // All tensors are accessed in forward propagation.
        
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

                tensors_swapped.insert(s);
                status::TensorPres pres = status.referenceTensor(s);
                for (auto &x : node.region.sections) {
                    // schedule_events.forward_schedule_events.execution[last_assigned].emplace_back(pres.getOperatorName(), s, x, events::ScheduleEventType::copyout, last_acquired);
                    schedule_events.forward_schedule_events.execution[last_acquired].emplace_back(pres.getOperatorName(), s, x, events::ScheduleEventType::swapout, last_acquired);
                    released_memory_requirement += x;
                }

                // if (unmet_memory_requirement <= released_memory_requirement) break;
            }
        }
        return tensors_swapped;
    }
    virtual void analyzeBackwardEvents(const events::EventSet<events::MemoryEvent>& iter_1_backward_mem_res, const std::unordered_set<std::string>& tensors_swapped) override {
        auto iter_1_backward_access_res = iter_1_backward_mem_res.select().where([](const events::EventSet<events::MemoryEvent>::item& item) {
            if (item.second.type == events::MemoryEventType::allocate) return false;
            if (item.second.type == events::MemoryEventType::free) return false;
            return item.second.type != events::MemoryEventType::swapin && item.second.type != events::MemoryEventType::swapout;
        }).get();

        for (auto &s : status.getExecutionOrder()) {
            auto op_pres = status.referenceOperator(s);
            if (!op_pres.isBackwardPropagation()) continue;

            auto target_operator_backward_res = iter_1_backward_access_res.select().where([&s, &tensors_swapped](const events::EventSet<events::MemoryEvent>::item& item) {
                if (item.second.op != s) return false;
                return tensors_swapped.find(item.second.tensor) != tensors_swapped.end();
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
            schedule_events.backward_schedule_events.timepoint.emplace_back(pres.getOperatorName(), pres.getName(), pres.getSize(), events::ScheduleEventType::copyin, x.second.timepoint);
        }
    }

public:
    SectionAwareMemoryScheduler(const Context::View& _context, status::MemoryStatus& _status, events::Events& _events): ExecutionTimeAwareMemoryScheduler(_context, _status, _events) { time_model.setStrongSynchronization(true); }
    virtual ~SectionAwareMemoryScheduler() = default;
};  // struct SectionAwareMemoryScheduler

struct DependencyAwareMemoryScheduler : public ExecutionTimeAwareMemoryScheduler {
private:
    struct TensorRelation {
        std::string current_operator = "";
        bool        schedule_changed = false;
        int         position         = 0;

        TensorRelation(const std::string op): current_operator(op), schedule_changed(false), position(0) {}
    };  // inner struct TensorRelation

private:
    std::unordered_map<std::string, TensorRelation> tensor_operator_relations;
    std::unordered_set<std::string>                 tensor_swapout_this_iter;

    bool   time_aware = true;
    size_t thershold  = 2;

public:
    DependencyAwareMemoryScheduler(const Context::View& _context, status::MemoryStatus& _status, events::Events& _events): ExecutionTimeAwareMemoryScheduler(_context, _status, _events) {
        time_aware = context.signal("dependency.timeaware");
        thershold  = std::stoul(context.at("dependency.thershold"));
    }

    virtual void analyzeBackwardEvents(const events::EventSet<events::MemoryEvent>& iter_1_backward_mem_res, const std::unordered_set<std::string>& tensors_swapped) override {
        auto iter_1_backward_access_res = iter_1_backward_mem_res.select().where([](const events::EventSet<events::MemoryEvent>::item& item) {
            if (item.second.type == events::MemoryEventType::allocate) return false;
            if (item.second.type == events::MemoryEventType::free) return false;
            return item.second.type != events::MemoryEventType::swapin && item.second.type != events::MemoryEventType::swapout;
        }).get();

        // Generate swap events.
        for (auto &x : tensors_swapped) {
            // Get the first access of this tensor in backword stage
            auto target_tensor_backward_res = iter_1_backward_access_res.select().where([&x](const events::EventSet<events::MemoryEvent>::item& item) {
                return item.second.tensor == x;
            }).get();

            if (target_tensor_backward_res.ref().empty()) continue;

            status::TensorPres pres = status.referenceTensor(x);
            std::string opb = (*target_tensor_backward_res.ref().begin())->second.op;
            size_t execution_time = 0;
            size_t transfer_time  = transferring_model.analyze(pres.getSize());
            for (int i = 0; i < thershold + 1; ++i) {
                if (!status.hasExecutionPrev(opb)) break;
                opb = status.getExecutionPrev(opb);
                assert(opb != "");
                if (!time_aware) continue;
                if (execution_time >= transfer_time) break;
                execution_time += execution_timespans[opb];
            }

            assert(opb != "");
            // Generate swapin event
            schedule_events.backward_schedule_events.execution[opb].emplace_back(pres.getOperatorName(), pres.getName(), pres.getSize(), events::ScheduleEventType::copyin, opb);
            tensor_operator_relations.emplace(pres.getName(), opb);
        }
    }
    virtual void onMemoryEvent(const events::MemoryEvent& event) override {
        FIFOMemoryScheduler::onMemoryEvent(event);

        if (!time_aware) return;

        if (event.stage == ApplicationStage::backward && (event.type == events::MemoryEventType::swapin || event.type == events::MemoryEventType::swapout)) {
            // Indicate swapping in too late or too early
            auto p = tensor_operator_relations.find(event.tensor);
            if (p == tensor_operator_relations.end()) return;

            // Indicate schedule already adjusted with no benefits
            if (p->second.position == 0 && p->second.schedule_changed) return;

            // Indicate this tensor is swapped in after swapped out in backward propagation
            if (tensor_swapout_this_iter.find(event.tensor) != tensor_swapout_this_iter.end()) return;

            auto& schedule_events_set = schedule_events.backward_schedule_events.execution;
            auto q = std::find_if(schedule_events_set[p->second.current_operator].begin(), schedule_events_set[p->second.current_operator].end(), [event](const events::ScheduleEvent& _event) {
                return _event.tensor_name == event.tensor;
            });
            assert(q != schedule_events_set[p->second.current_operator].end());

            std::string opb = "";
            if (event.type == events::MemoryEventType::swapin) {
                if (p->second.position == -4)   return;
                if (!status.hasExecutionPrev(p->second.current_operator)) return;
                opb = status.getExecutionPrev(p->second.current_operator);
                p->second.position -= 1;
            }
            else {
                if (p->second.position == 4)    return;
                if (!status.hasExecutionPost(p->second.current_operator)) return;
                opb = status.getExecutionPost(p->second.current_operator);
                p->second.position += 1;
                tensor_swapout_this_iter.insert(event.tensor);
            }
            // Cannot solve this schedule problem.
            assert(opb != "");
            
            events::ScheduleEvent new_event = *q;
            new_event.operator_name = opb;
            schedule_events_set[opb].push_back(new_event);
            schedule_events_set[p->second.current_operator].erase(q);

            p->second.current_operator = opb;
            p->second.schedule_changed = true;
        }
    }
    virtual void onNewIteration() override {
        tensor_swapout_this_iter.clear();
    }

    virtual ~DependencyAwareMemoryScheduler() = default;

};  // struct DependencyAwareMemoryScheduler

}   // namespace mori