#pragma once

#include "frontend/memory_operation_executor.hpp"
#include "includes/context.hpp"
#include "includes/memory_status.hpp"
#include "includes/memory_schedule_event.hpp"
#include "includes/logging.hpp"
#include "includes/exceptions.hpp"

namespace mori {

struct MemoryScheduleExecutor {
protected:
    Context context;

    status::MemoryStatus& status;
    
    std::vector<events::ScheduleEvent> eventset;
    std::shared_mutex events_mutex;
    std::atomic<bool> events_flag;

    std::thread executor_thread;
    std::recursive_mutex executor_mutex;

    std::vector<events::ScheduleEvent>::iterator current_event_posi;
    std::atomic<bool> iter_flag;

    std::atomic<bool> inited = false;

    std::weak_ptr<BackendHandle> backend_handle;
    
    Logger* logger = nullptr;

    Callbacks callbacks;

    MemoryOperationExecutor executor;

    std::chrono::steady_clock::time_point current_time_offset;

    inline int getExecutionInterval() { return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - current_time_offset).count(); }
    inline void resetExecutionInterval() {
        current_time_offset = std::chrono::steady_clock::now();
        current_event_posi = eventset.begin();
    }

    void onNextIteration() {
        logger->submit(LogLevel::debug, "Memory schedule executor moves to next iteration.");
        resetExecutionInterval();
    }

    void onNextOperator() {
        logger->submit(LogLevel::debug, "Memory schedule executor moves to next operator.");
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

        resetExecutionInterval();

        inited = true;

        executor_thread = std::thread([this](){
            while (inited) {
                std::shared_lock<std::shared_mutex>{events_mutex};
                int current_exec_interval = getExecutionInterval();
                auto current_end = std::find_if(current_event_posi, eventset.end(), 
                    [current_exec_interval](const events::ScheduleEvent& event) {return event.interval > current_exec_interval;});

                // Retrieve the schedule events that should be triggered.
                while (current_event_posi < current_end) {
                    const std::string& operator_name = current_event_posi->operator_name;
                    const std::string& tensor_name = current_event_posi->tensor_name;
                    status::TensorPres tensor = status.referenceTensor(tensor_name);
                    switch (current_event_posi->type) {
                        case events::ScheduleEventType::allocate:
                            executor.allocate(tensor);
                            break;
                        case events::ScheduleEventType::copyin:
                            executor.copyIn(tensor);
                            break;
                        case events::ScheduleEventType::copyout:
                            executor.copyOut(tensor);
                            break;
                        case events::ScheduleEventType::swapin:
                            executor.swapIn(tensor);
                            if (callbacks.count(CallbackStage::postSwapIn)) callbacks.at(CallbackStage::postSwapIn)(tensor_name, tensor.getDevicePointer(0));
                            (*logger)<<LogLevel::debug<<"Operator "<<operator_name<<": tensor "<<tensor_name<<" swapped in. (Prefetch)";
                            logger->flush();
                            backend_handle.lock()->submitEvent(events::MemoryEvent(tensor_name, events::MemoryEventType::swapin));
                            break;
                        case events::ScheduleEventType::swapout:
                            executor.swapOut(tensor);
                            if (callbacks.count(CallbackStage::postSwapOut)) callbacks.at(CallbackStage::postSwapOut)(tensor_name, tensor.getHostPointer(0));
                            (*logger)<<LogLevel::debug<<"Operator "<<operator_name<<": tensor "<<tensor_name<<" swapped out. (Instant)";
                            logger->flush();
                            backend_handle.lock()->submitEvent(events::MemoryEvent(tensor_name, events::MemoryEventType::swapout));
                            break;
                        case events::ScheduleEventType::freehost:
                            executor.freeHost(tensor);
                            break;
                        case events::ScheduleEventType::freedev:
                            executor.freeDevice(tensor);
                        case events::ScheduleEventType::free:
                             executor.free(tensor);
                            break;
                        default:
                            break;
                    }

                    ++current_event_posi;
                }
            }

        });
        // Examine if the thread starts properly
        while (!executor_thread.joinable());

        logger->submit(LogLevel::debug, "Memory schedule executor initialized.");
    }

    template <typename T>
    void updateSchedule(const T& new_event_set) {
        std::unique_lock<std::shared_mutex>{events_mutex};
        eventset = std::vector<events::ScheduleEvent>(new_event_set.begin(), new_event_set.end());
        logger->submit(LogLevel::debug, "Memory schedule executor receives new schedule event set.");
    }
    // template <typename T>
    // void updateSchedule(T&& new_event_set) {
    //     std::unique_lock<std::shared_mutex>{events_mutex};
    //     eventset = std::vector<ScheduleEvent>(new_event_set.begin(), new_event_set.end());
    //     logger->submit(LogLevel::debug, "Memory schedule executor receives new schedule event set.");
    // }

    int getIteration() {
        return 0;
    }

    void setIteration(int _iteration) {

    }

    void increaseIteration() {
        if (!inited) throw uninited_exception();
        onNextIteration();
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