#pragma once

#include "../includes/stdlibs.hpp"

#include "memory_manager.hpp"
#include "../includes/context.hpp"
#include "../includes/memory_status.hpp"
#include "../includes/memory_schedule_event.hpp"
#include "../includes/logging.hpp"

namespace mori {

struct MemoryScheduleExecutor {
protected:
    Context context;

    std::vector<ScheduleEvent> eventset;
    std::shared_mutex events_mutex;
    std::atomic<bool> events_flag;

    std::thread executor_thread;
    std::mutex executor_mutex;

    std::vector<ScheduleEvent>::iterator current_event_posi;
    std::atomic<bool> iter_flag;

    std::atomic<bool> inited = false;
    
    MemoryManager* memory_manager = nullptr;
    MemoryStatuses* memory_status = nullptr;
    
    Logger* logger = nullptr;

    void doAllocateMemory(const std::string& operator_name, const std::string& tensor_name) {
        TensorStatus& status = (*memory_status)[operator_name][tensor_name];
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
        TensorStatus& status = (*memory_status)[operator_name][tensor_name];
        std::unique_lock<std::shared_mutex> lock(status.status_mutex);
        switch (status.data_status) {
            case MemoryDataStatusType::none:
            case MemoryDataStatusType::empty:
                throw std::exception();
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
        TensorStatus& status = (*memory_status)[operator_name][tensor_name];
        std::unique_lock<std::shared_mutex> lock(status.status_mutex);
        switch (status.data_status) {
            case MemoryDataStatusType::none:
            case MemoryDataStatusType::empty:
                throw std::exception();
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
        TensorStatus& status = (*memory_status)[operator_name][tensor_name];
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
        TensorStatus& status = (*memory_status)[operator_name][tensor_name];
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
        TensorStatus& status = (*memory_status)[operator_name][tensor_name];
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
        logger->submit(LogLevel::debug, "Memory insufficient.");
        for (auto &op : memory_status->exec_order) {
            bool hosted = true;
            for (auto &tensor_status : memory_status->at(op)) {
                if (tensor_status.second.data_status == MemoryDataStatusType::host) continue;
                hosted = false;
            }

            if (hosted) continue;

            // Swap out this operator
            for (auto &tensor_status : memory_status->at(op)) {
                doSwapOutMemory(op, tensor_status.first);
                (*logger)<<LogLevel::debug<<"Operator "<<op<<": tensor "<<tensor_status.first<<" swapped out. (On demand)";
                logger->flush();
            }
            
            return;
        }
    }

public:
    MemoryScheduleExecutor(Context _context) {
        context = _context;
    }

    MemoryScheduleExecutor(const MemoryScheduleExecutor&) = delete;
    MemoryScheduleExecutor(MemoryScheduleExecutor&& executor) = delete;

    void setMemoryManager(MemoryManager* _memory_manager) {
        if (inited) throw std::exception();
        memory_manager = _memory_manager;
    }

    void setMemoryStatuses(MemoryStatuses* _memory_status) {
        if (inited) throw std::exception();
        memory_status = _memory_status;
    }

    void setLogger(Logger* _logger) {
        if (inited) throw std::exception();
        logger = _logger;
    }

    virtual void init() {
        if (inited) throw std::exception();

        if (memory_manager == nullptr) throw std::exception();
        if (memory_status == nullptr) throw std::exception();

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
        if (!inited) throw std::exception();
        onNextIteration();
    }

    void nextOperator() {
        if (!inited) throw std::exception();
        onNextOperator();
    }

    void waitMemory(size_t size) {
        if (!inited) throw std::exception();
        onMemoryInsufficient(size);
    }

    void terminate() {
        if (!inited) throw std::exception();

        inited = false;

        // Examine if the thread terminates properly
        if (executor_thread.joinable()) executor_thread.join();
        memory_manager = nullptr;
        logger = nullptr;
    }

    virtual ~MemoryScheduleExecutor() {
        if (inited) terminate();
    }

};  // struct MemoryScheduleExecutor

struct TimebasedMemoryScheduleExecutor : public MemoryScheduleExecutor {
protected:
    std::chrono::steady_clock::time_point current_time_offset;

public:
    TimebasedMemoryScheduleExecutor(const Context& _context): MemoryScheduleExecutor(_context) {}
    TimebasedMemoryScheduleExecutor(const TimebasedMemoryScheduleExecutor&) = delete;

    virtual int getExecutionInterval() {
        return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - current_time_offset).count();
    }

    virtual void resetExecutionInterval() {
        current_time_offset = std::chrono::steady_clock::now();
        current_event_posi = eventset.begin();
    }

    virtual ~TimebasedMemoryScheduleExecutor() {}

};  // struct TimebasedMemoryScheduleExecutor

struct DependencyMemoryScheduleExecutor : public MemoryScheduleExecutor {
protected:
    int current_offset;

public:
    DependencyMemoryScheduleExecutor(const Context& _context): MemoryScheduleExecutor(_context) {}
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

    virtual ~DependencyMemoryScheduleExecutor() {}
};  // struct DependencyMemoryScheduleExecutor

static std::shared_ptr<MemoryScheduleExecutor> make_executor(const Context& _context) {
    const std::string& type = _context["scheduler.trigger_event"];
    if (type == "time") return std::shared_ptr<MemoryScheduleExecutor>(new TimebasedMemoryScheduleExecutor(_context));
    else if (type == "dependency") return std::shared_ptr<MemoryScheduleExecutor>(new DependencyMemoryScheduleExecutor(_context));
    else throw std::exception();
}

}   // namespace mori