#pragma once

#include <shared_mutex>
#include <cassert>

#include "frontend/memory_operation_executor.hpp"
#include "frontend/callbacks.hpp"
#include "includes/context.hpp"
#include "includes/memory_status.hpp"
#include "includes/memory_schedule_event.hpp"
#include "includes/logging.hpp"
#include "includes/exceptions.hpp"

namespace mori {

struct MemoryScheduleExecutor final{
protected:
    Context context;
    status::MemoryStatus& status;
    Logger* logger = nullptr;
    Callbacks callbacks;
    std::weak_ptr<BackendHandle> backend_handle;
    
    // Schedule information
    std::shared_mutex events_m;
    events::StageScheduleEvents forward_schedule_events;
    events::StageScheduleEvents backward_schedule_events;
    std::atomic<events::StageScheduleEvents*> current_eventset;
    std::shared_mutex events_mutex;

    std::mutex new_events_m;
    std::atomic<bool> events_updated = false;
    events::ScheduleEvents new_events;

    std::shared_mutex current_operator_m;
    std::string current_operator;

    // Executor thread
    std::thread executor_thread;
    std::recursive_mutex executor_mutex;

    std::deque<events::ScheduleEvent> activated_events;
    std::mutex queue_m;

    std::atomic<bool> half_iter_sync = false;
    std::atomic<bool> iter_sync = false;

    // The schedule events are ordered.
    // The operator-triggered events are ordered by the execution sequence of operators.
    // The time-triggered events are ordered by the triggering timepoint.
    std::chrono::steady_clock::time_point current_time_offset;
    std::vector<events::ScheduleEvent>::iterator current_timepoint_event_posi;
    std::vector<events::ScheduleEvent>::iterator current_execution_event_posi;

    MemoryOperationExecutor executor;

    std::atomic<bool> inited = false;

    // Time-triggered events require these methods to reset the schedule timepoint offset.
    inline int getExecutionTimepoint() { return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - current_time_offset).count(); }
    inline void resetExecution() {
        // Reset execution of execution-triggered events.
        current_execution_event_posi = current_eventset.load()->execution.begin();
        // Reset execution of timepoint-triggered events.
        current_time_offset = std::chrono::steady_clock::now();
        current_timepoint_event_posi = current_eventset.load()->timepoint.begin();

        activated_events.clear();
    }

    void activateEvents() {
        std::vector<events::ScheduleEvent>& eventset = current_eventset.load()->timepoint;
        std::shared_lock<std::shared_mutex>{events_mutex};
        
        // Activate timepoint triggered events.
        // Execution triggered events do not need to be activated here.
        int current_exec_timepoint = getExecutionTimepoint();
        auto current_end = std::find_if(current_timepoint_event_posi, eventset.end(), 
            [current_exec_timepoint](const events::ScheduleEvent& event) {return event.timepoint > current_exec_timepoint;});

        // Retrieve the schedule events that should be triggered.
        std::unique_lock<std::mutex> queue_lock{queue_m};
        while (current_timepoint_event_posi < current_end) {
            activated_events.push_back(*current_timepoint_event_posi);
            ++current_timepoint_event_posi;
        }
    }

    void executeEvents() {
        std::unique_lock<std::mutex> queue_lock{queue_m};
        while (!activated_events.empty()) {
            // Retrieve tensor information.
            events::ScheduleEvent event = activated_events.front();
            activated_events.pop_front();

            const std::string& operator_name = event.operator_name;
            const std::string& tensor_name = event.tensor_name;
            size_t size = event.size;

            std::shared_lock<std::shared_mutex> col{current_operator_m};
            if (current_operator == tensor_name) continue;

            status::TensorPres tensor = status.referenceTensor(tensor_name);
            try {
                switch (event.type) {
                    case events::ScheduleEventType::copyin:
                        executor.copyIn(tensor, size);
                        break;
                    case events::ScheduleEventType::copyout:
                        executor.copyOut(tensor, size);
                        break;
                    case events::ScheduleEventType::swapin:
                        executor.swapIn(tensor, size);
                        if (callbacks.count(CallbackStage::postSwapIn)) callbacks.at(CallbackStage::postSwapIn)(tensor_name, tensor.getSection(0).device_address);
                        (*logger)<<LogLevel::debug<<"Operator "<<operator_name<<": tensor "<<tensor_name<<" swapped in. (Prefetch)";
                        logger->flush();
                        // backend_handle.lock()->submitEvent(events::MemoryEvent(tensor_name, size, events::MemoryEventType::swapin));
                        break;
                    case events::ScheduleEventType::swapout:
                        executor.swapOut(tensor, size);
                        if (callbacks.count(CallbackStage::postSwapOut)) callbacks.at(CallbackStage::postSwapOut)(tensor_name, tensor.getSection(0).host_address);
                        (*logger)<<LogLevel::debug<<"Operator "<<operator_name<<": tensor "<<tensor_name<<" swapped out. (Instant)";
                        logger->flush();
                        // backend_handle.lock()->submitEvent(events::MemoryEvent(tensor_name, size, events::MemoryEventType::swapout));
                        break;
                    case events::ScheduleEventType::freehost:
                        executor.freeHost(tensor, size);
                        break;
                    case events::ScheduleEventType::freedev:
                        executor.freeDevice(tensor, size);
                    case events::ScheduleEventType::free:
                        executor.free(tensor, size);
                        break;
                    default:
                        break;
                }
            } catch(std::exception& e) {
                (*logger)<<LogLevel::debug<<"Exception in executing memory swapping events, reason: " << e.what();
                logger->flush();
            }
        }
    }

public:
    MemoryScheduleExecutor(Context _context, status::MemoryStatus& _status): context(_context), status(_status) {}

    MemoryScheduleExecutor(const MemoryScheduleExecutor&) = delete;
    MemoryScheduleExecutor(MemoryScheduleExecutor&& executor) = delete;

    void setBackendHandle(const std::weak_ptr<BackendHandle>& _backend_handle) {
        if (inited) throw inited_exception();
        backend_handle = _backend_handle;
    }

    void setMemoryManager(MemoryManager* _memory_manager) {
        if (inited) throw inited_exception();
        executor.setMemoryManager(_memory_manager);
    }

    void setLogger(Logger* _logger) {
        if (inited) throw inited_exception();
        logger = _logger;
    }
    
    void setCallback(CallbackStage stage, const std::function<int(const std::string&, void*)>& callback) {
        if (inited) throw inited_exception();
        callbacks.emplace(stage, callback);
    }

    void init() {
        if (inited) throw inited_exception();

        current_eventset.store(&forward_schedule_events);
        resetExecution();

        inited = true;

        executor_thread = std::thread([this]() {
            while (inited) {
                // Examine if synchronization required.
                if (half_iter_sync || iter_sync) {
                    // Inactivate all events, and prevent further events.
                    std::unique_lock<std::mutex> ql{queue_m};
                    activated_events.clear();

                    if (half_iter_sync) {
                        std::shared_lock<std::shared_mutex> em{events_m};
                        assert(current_eventset.load() == &this->forward_schedule_events);
                        current_eventset.store(&this->backward_schedule_events);
                        resetExecution();
                        half_iter_sync = false;
                    }
                    if (iter_sync) {
                        if (events_updated) {
                            std::unique_lock<std::shared_mutex> em_n{events_m};
                            std::unique_lock<std::mutex> nem{new_events_m};

                            this->forward_schedule_events  = std::move(this->new_events.forward_schedule_events);
                            this->backward_schedule_events = std::move(this->new_events.backward_schedule_events);
                            logger->submit(LogLevel::debug, "Memory schedule executor switches to new schedule event set.");
                            events_updated = false;
                        }

                        std::shared_lock<std::shared_mutex> em{events_m};
                        current_eventset.store(&this->forward_schedule_events);
                        resetExecution();

                        iter_sync = false;
                    }
                }

                // Execution of schedule events
                // Activate events should be triggered.
                activateEvents();
                // Executed activated events.
                executeEvents();
            }

        });
        // Examine if the thread starts properly
        while (!executor_thread.joinable());

        logger->submit(LogLevel::debug, "Memory schedule executor initialized.");
    }

    void updateSchedule(const events::ScheduleEvents& _new_events) {
        std::unique_lock<std::mutex>{new_events_m};
        this->new_events = _new_events;
        events_updated = true;
    }
    void updateSchedule(events::ScheduleEvents&& _new_events) {
        std::unique_lock<std::mutex>{new_events_m};
        this->new_events = std::move(_new_events);
        events_updated = true;
    }

    void setOperatorStarted(const std::string& op) {
        std::unique_lock<std::shared_mutex> col{current_operator_m};
        current_operator = op;
    }

    void setOperatorFinished(const std::string& op) {
        std::unique_lock<std::mutex> ql{queue_m};
        while (current_execution_event_posi != current_eventset.load()->execution.end()) {
            if (current_execution_event_posi->postop == op) {
                activated_events.push_back(*current_execution_event_posi++);
            } else break;
        }
        // logger->submit(LogLevel::debug, "Memory schedule executor moves to next operator.");
    }

    int getIteration() { return 0; }
    void setIteration(int _iteration) {}

    void newIteration() {
        if (!inited) throw uninited_exception();
        iter_sync = true;
        while (iter_sync);
        logger->submit(LogLevel::debug, "Memory schedule executor moves to next iteration.");
    }

    /**
     * @brief Set half of the iteration finished.
     * @note  The schedule events for forward propagation will be synchronized to be executed and the backward propagation schedule events will be prepared to triggered.
    */
    void halfIteration() {
        if (!inited) throw uninited_exception();
        half_iter_sync = true;
        while (half_iter_sync);
    }

    void terminate() {
        if (!inited) throw uninited_exception();

        inited = false;

        // Examine if the thread terminates properly
        if (executor_thread.joinable()) executor_thread.join();

    }

    ~MemoryScheduleExecutor() {
        if (inited) terminate();

        logger = nullptr;
    }

};  // struct MemoryScheduleExecutor

}   // namespace mori