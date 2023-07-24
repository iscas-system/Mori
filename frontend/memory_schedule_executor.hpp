#pragma once

#include <shared_mutex>
#include <cassert>

#include "frontend/memory_operation_executor.hpp"
#include "frontend/callbacks.hpp"
#include "includes/context.hpp"
#include "includes/memory_status.hpp"
#include "includes/memory_schedule_event.hpp"
#include "includes/logging.hpp"
#include "includes/exceptions/status_exceptions.hpp"

namespace mori {

struct MemoryScheduleExecutor final {
protected:
    Context context;
    status::MemoryStatus& status;
    layout::MemoryLayout& layout;
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

    // Executor thread
    std::thread executor_thread;
    std::recursive_mutex executor_mutex;

    std::deque<events::ScheduleEvent> activated_events;
    std::mutex queue_m;
    std::atomic<size_t> released_size;
    std::condition_variable released_cond;

    std::atomic<bool> half_iter_sync = false;
    std::atomic<bool> iter_sync = false;
    std::atomic<int>  iteration = 0;

    // The schedule events are ordered.
    // The operator-triggered events are ordered by the execution sequence of operators.
    // The time-triggered events are ordered by the triggering timepoint.
    std::chrono::steady_clock::time_point current_time_offset;
    std::vector<events::ScheduleEvent>::iterator current_timepoint_event_posi;

    MemoryOperationExecutor executor;

    std::atomic<bool> inited = false;

    // Time-triggered events require these methods to reset the schedule timepoint offset.
    inline long getExecutionTimepoint() { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - current_time_offset).count(); }
    inline void resetExecution() {
        std::unique_lock<std::mutex> queue_lock{queue_m};
        activated_events.clear();
        queue_lock.unlock();

        // Reset execution of timepoint-triggered events.
        current_time_offset = std::chrono::steady_clock::now();
        current_timepoint_event_posi = current_eventset.load()->timepoint.begin();
    }

    void activateEvents() {
        std::vector<events::ScheduleEvent>& eventset = current_eventset.load()->timepoint;
        std::shared_lock<std::shared_mutex> l{events_mutex};
        
        // Activate timepoint triggered events.
        // Execution triggered events do not need to be activated here.
        long current_exec_timepoint = getExecutionTimepoint();
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
        auto target_events = std::move(activated_events);
        queue_lock.unlock();
        
        while (!target_events.empty()) {
            if (half_iter_sync || iter_sync) break;
            // Retrieve tensor information.
            events::ScheduleEvent event = target_events.front();
            target_events.pop_front();

            const std::string& operator_name = event.operator_name;
            const std::string& tensor_name = event.tensor_name;
            size_t size = event.size;

            status::TensorPres tensor = status.referenceTensor(tensor_name);
            // (*logger) << LogLevel::debug << "Operator " << operator_name << ": tensor " << tensor_name << " start to be swapped out. (Instant)" << endl;
            if (!tensor.isMemoryLocated()) continue;
            try {
                switch (event.type) {
                    case events::ScheduleEventType::copyin:
                        if (tensor.isDeviceAllLocated()) break;     // No data to copy in.
                        executor.copyIn(tensor, size);
                        if (callbacks.count(CallbackStage::postSwapIn)) callbacks.at(CallbackStage::postSwapIn)(tensor_name, tensor.getSection(0).device_address);
                        (*logger) << LogLevel::debug << "Operator " << operator_name << ": tensor " << tensor_name << " copied in. (Prefetch)" << endl;
                        break;
                    case events::ScheduleEventType::copyout:
                        if (!tensor.isDeviceLocated()) break;       // No data to copy out.
                        executor.copyOut(tensor, size);
                        break;
                    case events::ScheduleEventType::swapin:
                        if (tensor.isDeviceAllLocated()) break;     // No data to swap in.
                        executor.swapIn(tensor, size);
                        if (callbacks.count(CallbackStage::postSwapIn)) callbacks.at(CallbackStage::postSwapIn)(tensor_name, tensor.getSection(0).device_address);
                        (*logger) << LogLevel::debug << "Operator " << operator_name << ": tensor " << tensor_name << " swapped in. (Prefetch)" << endl;
                        break;
                    case events::ScheduleEventType::swapout:
                        if (!tensor.isDeviceLocated()) break;       // No data to swap out.
                        executor.swapOut(tensor, size);
                        if (callbacks.count(CallbackStage::postSwapOut)) callbacks.at(CallbackStage::postSwapOut)(tensor_name, tensor.getSection(0).host_address);
                        (*logger) << LogLevel::debug << "Operator " << operator_name << ": tensor " << tensor_name << " swapped out. (Instant)" << endl;
                        // Finished swapout event.
                        released_size += size;
                        released_cond.notify_all();
                        break;
                    case events::ScheduleEventType::freehost:
                        if (!tensor.isHostLocated()) break;         // No data to free on host.
                        executor.freeHost(tensor, size);
                        break;
                    case events::ScheduleEventType::freedev:
                        if (!tensor.isDeviceLocated()) break;       // No data to free on device.
                        executor.freeDevice(tensor, size);
                        if (callbacks.count(CallbackStage::postSwapOut)) callbacks.at(CallbackStage::postSwapOut)(tensor_name, tensor.getSection(0).host_address);
                        (*logger) << LogLevel::debug << "Operator " << operator_name << ": tensor " << tensor_name << " freed on device. (Instant)" << endl;
                        // Finished freedev event.
                        released_size += size;
                        released_cond.notify_all();
                        break;
                    case events::ScheduleEventType::free:
                        executor.free(tensor, size);
                        break;
                    default:
                        break;
                }
            } catch(std::exception& e) {
                (*logger) << LogLevel::debug << "Exception in executing memory swapping events, reason: " << e.what() << endl;
            }
        }
        // Currently no more schedule events.
        released_cond.notify_all();
    }

public:
    MemoryScheduleExecutor(Context _context, status::MemoryStatus& _status, layout::MemoryLayout& _layout): context(_context), status(_status), layout(_layout), executor(_layout) {}

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
        std::unique_lock<std::mutex> l{new_events_m};
        this->new_events = _new_events;
        events_updated = true;
    }
    void updateSchedule(events::ScheduleEvents&& _new_events) {
        std::unique_lock<std::mutex> l{new_events_m};
        this->new_events = std::move(_new_events);
        events_updated = true;
    }

    void setOperatorStarted(const std::string& op) {}

    void setOperatorFinished(const std::string& op) {
        std::unique_lock<std::mutex> ql{queue_m};
        for (auto &x : current_eventset.load()->execution[op]) {
            activated_events.push_back(x);
        }
        // logger->submit(LogLevel::debug, "Memory schedule executor moves to next operator.");
    }

    int getIteration() { return iteration; }
    void setIteration(int _iteration) { iteration = _iteration; }

    void newIteration() {
        if (!inited) throw uninited_exception();
        iter_sync = true;
        while (iter_sync);
        logger->submit(LogLevel::debug, "Memory schedule executor moves to next iteration.");
        ++iteration;
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

    size_t waitMemory(size_t size) {
        // Assume swapout events only exist in forward propagation.
        if (current_eventset.load() == &this->backward_schedule_events) return 0;
        std::unique_lock<std::mutex> ql{queue_m};
        if (activated_events.empty()) return 0;

        released_size = 0;
        released_cond.wait(ql, [this, size]() {
            if (released_size >= size) return true;
            return activated_events.empty();
            });
        return released_size;
    }

    ~MemoryScheduleExecutor() {
        if (inited) terminate();

        logger = nullptr;
    }

};  // struct MemoryScheduleExecutor

}   // namespace mori