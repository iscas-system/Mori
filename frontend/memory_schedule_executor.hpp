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
#include "includes/presentation.hpp"

namespace mori {

struct BackendHandle;

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
    
    // Memory synchronization information
    std::mutex                            exec_sync_mutex;
    std::chrono::steady_clock::time_point exec_sync_time_offset;

    std::atomic<bool> half_iter_sync = false;
    std::atomic<bool> iter_sync      = false;
    std::atomic<bool> exec_sync      = false;
    std::atomic<bool> next_op_sync   = false;

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
        exec_sync_time_offset = current_time_offset;
        current_timepoint_event_posi = current_eventset.load()->timepoint.begin();

        next_op_sync = false;
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
            if (current_timepoint_event_posi->instant) executeEvent(*current_timepoint_event_posi);
            else activated_events.push_back(*current_timepoint_event_posi);
            ++current_timepoint_event_posi;
        }
    }

    bool executeEvent(const events::ScheduleEvent& event) {
        status::TensorView tensor_view = status.tryReferenceTensor(event.tensor_name);
        if (!tensor_view.isReferenced()) return false;
        status::TensorPres tensor_pres = tensor_view.reference();

        switch (event.type) {
            case events::ScheduleEventType::copyin:
                // No data to copy in.
                if (tensor_pres.isDeviceAllLocated()) break;
                    if (!tensor_pres.isHostLocated()) break;
                    executor.copyIn(tensor_pres, event.size);
                    if (callbacks.count(CallbackStage::postSwapIn)) callbacks.at(CallbackStage::postSwapIn)(event.tensor_name, tensor_pres.getSection(0).device_address);
                    (*logger) << LogLevel::debug << "Operator " << event.operator_name << ": tensor " << event.tensor_name << " copied in. (Prefetch)" << endl;
                    break;
                case events::ScheduleEventType::copyout:
                    // No data to copy out.
                    if (!tensor_pres.isDeviceLocated()) break;
                    if (tensor_pres.isHostAllLocated()) break;
                    executor.copyOut(tensor_pres, event.size);
                    break;
                case events::ScheduleEventType::swapin:
                    // No data to swap in.
                    if (tensor_pres.isDeviceAllLocated()) break;
                    if (!tensor_pres.isHostLocated()) break;
                    executor.swapIn(tensor_pres, event.size);
                    if (callbacks.count(CallbackStage::postSwapIn)) callbacks.at(CallbackStage::postSwapIn)(event.tensor_name, tensor_pres.getSection(0).device_address);
                    (*logger) << LogLevel::debug << "Operator " << event.operator_name << ": tensor " << event.tensor_name << " swapped in. (Prefetch)" << endl;
                    break;
                case events::ScheduleEventType::swapout:
                    // No data to swap out.
                    if (!tensor_pres.isDeviceLocated()) break;
                    if (tensor_pres.isHostAllLocated()) break;
                    executor.swapOut(tensor_pres, event.size);
                    if (callbacks.count(CallbackStage::postSwapOut)) callbacks.at(CallbackStage::postSwapOut)(event.tensor_name, tensor_pres.getSection(0).host_address);
                    (*logger) << LogLevel::debug << "Operator " << event.operator_name << ": tensor " << event.tensor_name << " swapped out. (Instant)" << endl;
                    break;
                case events::ScheduleEventType::freehost:
                    // No data to free on host.
                    if (!tensor_pres.isHostLocated()) break;
                    executor.freeHost(tensor_pres, event.size);
                    break;
                case events::ScheduleEventType::freedev:
                    // No data to free on device.
                    if (!tensor_pres.isDeviceLocated()) break;
                    executor.freeDevice(tensor_pres, event.size);
                    if (callbacks.count(CallbackStage::postSwapOut)) callbacks.at(CallbackStage::postSwapOut)(event.tensor_name, tensor_pres.getSection(0).host_address);
                    (*logger) << LogLevel::debug << "Operator " << event.operator_name << ": tensor " << event.tensor_name << " freed on device. (Instant)" << endl;
                    break;
                case events::ScheduleEventType::free:
                    // No data to free on host and device.
                    if (!tensor_pres.isMemoryLocated()) break;
                    executor.free(tensor_pres, event.size);
                    break;
                default:
                    break;
            }
        return true;
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

                if (exec_sync) continue;
                std::unique_lock<std::mutex> l{exec_sync_mutex, std::try_to_lock};
                if (!l.owns_lock()) continue;
                // Execution of schedule events
                // Activate events should be triggered.
                activateEvents();
                // Execute activated events.
                std::unique_lock<std::mutex> queue_lock{queue_m};
                size_t target_executed_events  = activated_events.size();
                size_t current_executed_events = 0;
                queue_lock.unlock();
                
                while (current_executed_events < target_executed_events) {
                    if (half_iter_sync || iter_sync || exec_sync) break;
                    if (next_op_sync) continue;

                    queue_lock.lock();
                    // Retrieve tensor information.
                    if (activated_events.empty()) return;
                    const events::ScheduleEvent& event = activated_events.front();
                    queue_lock.unlock();

                    try {                        
                        if (executeEvent(event)) {
                            // Success execution of event.
                            queue_lock.lock();
                            activated_events.pop_front();
                            queue_lock.unlock();
                            ++current_executed_events;
                        }
                    } catch(memory_insufficience& e) {
                        (*logger) << LogLevel::debug << "Exception in executing memory swapping events, reason: " << e.what() << ", " << e.demand() << " unmet." << endl;
                        next_op_sync = true;
                    } catch(std::exception& e) {
                        (*logger) << LogLevel::debug << "Exception in executing memory swapping events, reason: " << e.what() << endl;
                    }
                }
                // Currently no more schedule events.
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
        next_op_sync = false;
        std::unique_lock<std::mutex> ql{queue_m};
        for (auto &x : current_eventset.load()->execution[op]) {
            if (x.instant) executeEvent(x);
            else activated_events.push_back(x);
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

    /**
     * @brief Synchronize with memory schedule executor. Prevent further activation and exectuion of memory schedule events.
     * @note  Leverage with mori::utils::Presentation suggested.
     */
    void synchronize() {
        exec_sync = true;
        exec_sync_mutex.lock();
        // Memory insufficient, block the forward schedule events and perform passive memory swapping.
        // Since the memory swapping is performed on the specific copying stream, this synchroization does not lead to further overhead.
        exec_sync_time_offset = std::chrono::steady_clock::now();
    }
    /**
     * @brief Release memory scheduler executor. Proceed further activation and execution of memory schedule events.
     * @note  Leverage with mori::utils::Presentation suggested.
     */
    void release() {
        if (exec_sync_mutex.try_lock()) {
            // Indicating synchronization not locked.
            exec_sync_mutex.unlock();
            throw uninited_exception("Memory Schedule executor not in synchronization.");
        }
        std::chrono::steady_clock::duration synchronization_time_duration = std::chrono::steady_clock::now() - exec_sync_time_offset;
        current_time_offset += synchronization_time_duration;
        exec_sync_mutex.unlock();
        exec_sync = false;
    }

    ~MemoryScheduleExecutor() {
        if (inited) terminate();

        logger = nullptr;
    }

};  // struct MemoryScheduleExecutor

namespace utils {

template <>
struct PresentationFunction<MemoryScheduleExecutor> {
    inline static void require(MemoryScheduleExecutor& target) { target.synchronize(); }
    inline static void release(MemoryScheduleExecutor& target) { target.release();     }
};  // struct PresentationFunction<MemoryScheduleExecutor>

}   // namespace utils
}   // namespace mori