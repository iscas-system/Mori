#pragma once

#include "includes/stdlibs.hpp"

#include "frontend/memory_manager.hpp"
#include "includes/context.hpp"
#include "includes/memory_status.hpp"
#include "includes/memory_schedule_event.hpp"
#include "includes/logging.hpp"
#include "includes/exceptions.hpp"

namespace mori {

struct MemoryScheduleExecutor {
protected:
    Context context;

    std::vector<ScheduleEvent> eventset;
    std::shared_mutex events_mutex;
    std::atomic<bool> events_flag;

    std::thread executor_thread;
    std::recursive_mutex executor_mutex;

    std::vector<ScheduleEvent>::iterator current_event_posi;
    std::atomic<bool> iter_flag;

    std::atomic<bool> inited = false;
    
    MemoryManager* memory_manager = nullptr;
    MemoryStatuses& memory_status;
    
    Logger* logger = nullptr;

    void doAllocateMemory(const std::string& operator_name, const std::string& tensor_name) {
        TensorStatus& status = memory_status[operator_name][tensor_name];
        std::unique_lock<std::shared_mutex> lock(status.status_mutex);
        switch (status.data_status) {
            case MemoryDataStatusType::none:
                status.device_address = memory_manager->allocate(status.size);
                status.data_status = MemoryDataStatusType::empty;
            case MemoryDataStatusType::empty:
            case MemoryDataStatusType::device:
            case MemoryDataStatusType::host:
            case MemoryDataStatusType::coexist:
                break;
            case MemoryDataStatusType::swapin:
                assert(0);
            case MemoryDataStatusType::swapout:
                assert(0);
            default:
                break;
        }
    }

    void doCopyInMemory(const std::string& operator_name, const std::string& tensor_name) {
        TensorStatus& status = memory_status[operator_name][tensor_name];
        std::unique_lock<std::shared_mutex> lock(status.status_mutex);
        switch (status.data_status) {
            case MemoryDataStatusType::none:
            case MemoryDataStatusType::empty:
                throw status_error("Copying in non-exist or empty data.");
            case MemoryDataStatusType::device:
                break;
            case MemoryDataStatusType::host:
                status.device_address = memory_manager->copyIn(status.host_address, status.size);
                status.data_status = MemoryDataStatusType::coexist;
                break;
            case MemoryDataStatusType::coexist:
                break;
            case MemoryDataStatusType::swapin:
                assert(0);
            case MemoryDataStatusType::swapout:
                assert(0);
            default:
                break;
        }
    }

    void doCopyOutMemory(const std::string& operator_name, const std::string& tensor_name) {
        TensorStatus& status = memory_status[operator_name][tensor_name];
        std::unique_lock<std::shared_mutex> lock(status.status_mutex);
        switch (status.data_status) {
            case MemoryDataStatusType::none:
            case MemoryDataStatusType::empty:
                throw status_error("Copying out non-exist or empty data.");
            case MemoryDataStatusType::device:
                status.host_address = memory_manager->copyOut(status.device_address, status.size);
                status.data_status = MemoryDataStatusType::coexist;
                break;
            case MemoryDataStatusType::host:
            case MemoryDataStatusType::coexist:
                break;
            case MemoryDataStatusType::swapin:
                assert(0);
            case MemoryDataStatusType::swapout:
                assert(0);
            default:
                break;
        }
    }

    void doFreeDeviceMemory(const std::string& operator_name, const std::string& tensor_name) {
        TensorStatus& status = memory_status[operator_name][tensor_name];
        std::unique_lock<std::shared_mutex> lock(status.status_mutex);
        switch (status.data_status) {
            case MemoryDataStatusType::none:
                break;
            case MemoryDataStatusType::empty:
            case MemoryDataStatusType::device:
                memory_manager->freeDevice(status.device_address);
                status.data_status = MemoryDataStatusType::none;
                break;
            case MemoryDataStatusType::host:
                break;
            case MemoryDataStatusType::coexist:
                memory_manager->freeDevice(status.device_address);
                status.data_status = MemoryDataStatusType::host;
                break;
            case MemoryDataStatusType::swapin:
                assert(0);
            case MemoryDataStatusType::swapout:
                assert(0);
            default:
                break;
        }
    }

    void doFreeHostMemory(const std::string& operator_name, const std::string& tensor_name) {
        TensorStatus& status = memory_status[operator_name][tensor_name];
        std::unique_lock<std::shared_mutex> lock(status.status_mutex);
        switch (status.data_status) {
            case MemoryDataStatusType::none:
            case MemoryDataStatusType::empty:
            case MemoryDataStatusType::device:
                break;
            case MemoryDataStatusType::host:
                memory_manager->freeHost(status.host_address);
                status.data_status = MemoryDataStatusType::none;
                break;
            case MemoryDataStatusType::coexist:
                memory_manager->freeHost(status.host_address);
                status.data_status = MemoryDataStatusType::device;
                break;
            case MemoryDataStatusType::swapin:
                assert(0);
            case MemoryDataStatusType::swapout:
                assert(0);
            default:
                break;
        }
    }

    void doSwapInMemory(const std::string& operator_name, const std::string& tensor_name) {
        doCopyInMemory(operator_name, tensor_name);
        doFreeHostMemory(operator_name, tensor_name);
    }

    void doSwapOutMemory(const std::string& operator_name, const std::string& tensor_name) {

        doCopyOutMemory(operator_name, tensor_name);
        doFreeDeviceMemory(operator_name, tensor_name);
    }

    void doFreeMemory(const std::string& operator_name, const std::string& tensor_name) {
        TensorStatus& status = memory_status[operator_name][tensor_name];
        switch (status.data_status) {
            case MemoryDataStatusType::none:
                break;
            case MemoryDataStatusType::empty:
            case MemoryDataStatusType::device:
                memory_manager->freeDevice(status.device_address);
                status.data_status = MemoryDataStatusType::none;
                break;
            case MemoryDataStatusType::host:
                memory_manager->freeHost(status.host_address);
                status.data_status = MemoryDataStatusType::none;
                break;
            case MemoryDataStatusType::coexist:
                memory_manager->freeHost(status.host_address);
                memory_manager->freeDevice(status.device_address);
                status.data_status = MemoryDataStatusType::device;
                break;
            case MemoryDataStatusType::swapin:
                assert(0);
                break;
            case MemoryDataStatusType::swapout:
                assert(0);
                break;
            default:
                break;
        }
    }

    virtual int getExecutionInterval() =0;
    virtual void resetExecutionInterval() =0;

    virtual void onNextIteration() {
        logger->submit(LogLevel::debug, "Memory schedule executor moves to next iteration.");
        resetExecutionInterval();
    }

    virtual void onNextOperator() {
        logger->submit(LogLevel::debug, "Memory schedule executor moves to next operator.");
    }

    /**
     * onMemoryInsufficient
     * Process swapping out when memory is not sufficient.
     * Here a LRU algorithm is leveraged.
     */
    virtual void onMemoryInsufficient(size_t size) {
        (*logger) << LogLevel::debug << "Memory insufficient.";
        logger->flush();

        size_t cur_swapped_size = 0;
        for (auto &op : memory_status.exec_order) {
            for (auto &tensor_status : memory_status.at(op)) {
                switch (tensor_status.second.data_status) {
                case device:
                case coexist:
                    doSwapOutMemory(op, tensor_status.first);
                    (*logger)<<LogLevel::debug<<"Operator "<<op<<": tensor "<<tensor_status.first<<" swapped out. (On demand)";
                    logger->flush();
                    cur_swapped_size += tensor_status.second.size;
                default:
                    break;
                }

                if (cur_swapped_size >= size) return;
            }
            
        }
    }

public:
    MemoryScheduleExecutor(Context _context, MemoryStatuses& _memory_status): context(_context), memory_status(_memory_status) {}

    MemoryScheduleExecutor(const MemoryScheduleExecutor&) = delete;
    MemoryScheduleExecutor(MemoryScheduleExecutor&& executor) = delete;

    void setMemoryManager(MemoryManager* _memory_manager) {
        if (inited) throw inited_exception();
        memory_manager = _memory_manager;
    }

    void setLogger(Logger* _logger) {
        if (inited) throw inited_exception();
        logger = _logger;
    }

    virtual void init() {
        if (inited) throw inited_exception();

        if (memory_manager == nullptr) throw status_error("Memory manager not assigned.");

        resetExecutionInterval();

        inited = true;

        executor_thread = std::thread([this](MemoryScheduleExecutor* executor){
            while (inited) {
                std::shared_lock<std::shared_mutex>{events_mutex};
                int current_exec_interval = executor->getExecutionInterval();
                auto current_end = std::find_if(current_event_posi, eventset.end(), 
                    [current_exec_interval](const ScheduleEvent& event) {return event.interval > current_exec_interval;});

                // Retrieve the schedule events that should be triggered.
                while (current_event_posi < current_end) {
                    const std::string& operator_name = current_event_posi->operator_name;
                    const std::string& tensor_name = current_event_posi->tensor_name;
                    switch (current_event_posi->type) {
                        case ScheduleEventType::allocate:
                            doAllocateMemory(operator_name, tensor_name);
                            break;
                        case ScheduleEventType::copyin:
                            doCopyInMemory(operator_name, tensor_name);
                            break;
                        case ScheduleEventType::copyout:
                            doCopyOutMemory(operator_name, tensor_name);
                            break;
                        case ScheduleEventType::swapin:
                            doSwapInMemory(operator_name, tensor_name);
                            (*logger)<<LogLevel::debug<<"Operator "<<operator_name<<": tensor "<<tensor_name<<" swapped in. (Prefetch)";
                            logger->flush();
                            break;
                        case ScheduleEventType::swapout:
                            doSwapOutMemory(operator_name, tensor_name);
                            (*logger)<<LogLevel::debug<<"Operator "<<operator_name<<": tensor "<<tensor_name<<" swapped out. (Instant)";
                            logger->flush();
                            break;
                        case ScheduleEventType::freehost:
                            doFreeHostMemory(operator_name, tensor_name);
                            break;
                        case ScheduleEventType::freedev:
                            doFreeDeviceMemory(operator_name, tensor_name);
                        case ScheduleEventType::free:
                             doFreeMemory(operator_name, tensor_name);
                            break;
                        default:
                            break;
                    }

                    ++current_event_posi;
                }
            }

        }, this);
        // Examine if the thread starts properly
        while (!executor_thread.joinable());

        logger->submit(LogLevel::debug, "Memory schedule executor initialized.");
    }

    void updateSchedule(const std::vector<ScheduleEvent>& new_event_set) {
        std::unique_lock<std::shared_mutex>{events_mutex};
        eventset = new_event_set;
        logger->submit(LogLevel::debug, "Memory schedule executor receives new schedule event set.");
    }
    void updateSchedule(std::vector<ScheduleEvent>&& new_event_set) {
        std::unique_lock<std::shared_mutex>{events_mutex};
        eventset = new_event_set;
        logger->submit(LogLevel::debug, "Memory schedule executor receives new schedule event set.");
    }

    void nextIteration() {
        if (!inited) throw uninited_exception();
        onNextIteration();
    }

    void nextOperator() {
        if (!inited) throw uninited_exception();
        onNextOperator();
    }

    void waitMemory(size_t size) {
        if (!inited) throw uninited_exception();
        onMemoryInsufficient(size);
    }

    void terminate() {
        if (!inited) throw uninited_exception();

        inited = false;

        // Examine if the thread terminates properly
        if (executor_thread.joinable()) executor_thread.join();

    }

    virtual ~MemoryScheduleExecutor() {
        if (inited) terminate();

        memory_manager = nullptr;
        logger = nullptr;
    }

};  // struct MemoryScheduleExecutor

struct TimebasedMemoryScheduleExecutor : public MemoryScheduleExecutor {
protected:
    std::chrono::steady_clock::time_point current_time_offset;

public:
    TimebasedMemoryScheduleExecutor(const Context& _context, MemoryStatuses& _memory_status): MemoryScheduleExecutor(_context, _memory_status) {}
    TimebasedMemoryScheduleExecutor(const TimebasedMemoryScheduleExecutor&) = delete;

    virtual int getExecutionInterval() {
        return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - current_time_offset).count();
    }

    virtual void resetExecutionInterval() {
        current_time_offset = std::chrono::steady_clock::now();
        current_event_posi = eventset.begin();
    }

    virtual ~TimebasedMemoryScheduleExecutor() = default;

};  // struct TimebasedMemoryScheduleExecutor

struct DependencyMemoryScheduleExecutor : public MemoryScheduleExecutor {
protected:
    int current_offset;

public:
    DependencyMemoryScheduleExecutor(const Context& _context, MemoryStatuses& _memory_status): MemoryScheduleExecutor(_context, _memory_status) {}
    DependencyMemoryScheduleExecutor(const DependencyMemoryScheduleExecutor&) = delete;

    virtual int getExecutionInterval() {
        return current_offset;
    }

    virtual void onNextOperator() {
        ++current_offset;
    }

    virtual void resetExecutionInterval() {
        current_offset = 0;
        current_event_posi = eventset.begin();
    }

    virtual ~DependencyMemoryScheduleExecutor() = default;
};  // struct DependencyMemoryScheduleExecutor

static std::shared_ptr<MemoryScheduleExecutor> make_executor(const Context& _context, MemoryStatuses& _memory_status) {
    const std::string& type = _context.at("scheduler.trigger_event");
    if (type == "time") return std::shared_ptr<MemoryScheduleExecutor>(new TimebasedMemoryScheduleExecutor(_context, _memory_status));
    else if (type == "dependency") return std::shared_ptr<MemoryScheduleExecutor>(new DependencyMemoryScheduleExecutor(_context, _memory_status));
    else throw context_invalid("scheduler.trigger_event");
}

}   // namespace mori