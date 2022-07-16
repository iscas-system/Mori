#pragma once

#include "includes/stdlibs.hpp"

#include "frontend/backend_handle.hpp"
#include "frontend/memory_schedule_executor.hpp"
#include "includes/memory_status.hpp"
#include "includes/memory_event.hpp"

namespace mori {

/**
 * MemorySession
 * Management of a memory session, which is a complete memory lifecycle of a training iteration.
 */
struct MemorySession {
protected:
    Context context;

    std::atomic<int> step;

    std::weak_ptr<BackendHandle> backend_handle;

    std::weak_ptr<MemoryScheduleExecutor> executor;

    MemoryStatuses* memory_status;
    MemoryManager* memory_manager;

    bool inited = false;

    Logger* logger;

public:
    MemorySession(const Context& _context): context(_context) {}

    void setBackendHandle(const std::weak_ptr<BackendHandle>& _backend_handle) {
        if (inited) throw inited_exception();
        backend_handle = _backend_handle;
    }

    void setMemoryStatusStorage(MemoryStatuses* _memory_status) {
        if (inited) throw inited_exception();
        memory_status = _memory_status;
    }

    void setExecutor(const std::weak_ptr<MemoryScheduleExecutor>& _executor) {
        if (inited) throw inited_exception();
        executor = _executor;
    }

    void setMemoryManager(MemoryManager* _memory_manager) {
        if (inited) throw inited_exception();
        memory_manager = _memory_manager;
    }

    void setLogger(Logger* _logger) {
        if (inited) throw inited_exception();
        logger = _logger;
    }

    void init() {
        if (inited) throw inited_exception();

        if (backend_handle.lock() == nullptr) throw status_error("Backend not inited.");
        if (memory_status == nullptr) throw status_error("Memory status storage not assigned.");
        if (executor.lock() == nullptr) throw status_error("Memory schedule executor not assigned.");

        executor.lock()->init();

        inited = true;
    }

    bool isInited() {return inited;}

    /**
     * allocateMemory
     * Allocate memory on device for tensor.
     * @param op operator name
     * @param tensor tensor name
     */
    void allocateMemory(const std::string& op, const std::string& tensor) {
        if (!inited) throw uninited_exception();

        if (!memory_status->isTensorRegistered(op, tensor)) throw status_error("Operator or tensor not registered.");

        void* dev_addr = nullptr;
        int retry_count = 2;
        {
            TensorStatus& tensor_status = (*memory_status)[op][tensor];
            std::unique_lock<std::shared_mutex>{tensor_status.status_mutex};
            if (tensor_status.data_status != MemoryDataStatusType::none) throw status_error("Allocating for existing data.");

            while (dev_addr == nullptr && retry_count-- > 0) {
                dev_addr = memory_manager->allocate(tensor_status.size);
            }

            if (retry_count <= 0) {
                // TODO: deal with concurrency
                waitMemory(tensor_status.size);
                dev_addr = memory_manager->allocate(tensor_status.size);
            }

            // update tensor memory status
            tensor_status.device_address = dev_addr;
            tensor_status.data_status = MemoryDataStatusType::empty;
        }

        // emit memory event
        backend_handle.lock()->submitEvent(MemoryEvent(op, tensor, MemoryEventType::allocate));

    }
    // void allocateMemory(const std::string& op, const std::string& tensor) {
    //     allocateMemory(op, tensor, [](size_t size){return memory_manager->allocate(tensor_status.size);});
    // }

    void setMemoryDataAllocated(const std::string& op, const std::string& tensor, void* address) {
        if (!inited) throw uninited_exception();

        if (!memory_status->isTensorRegistered(op, tensor)) throw status_error("Operator or tensor not registered.");

        {
            TensorStatus& tensor_status = (*memory_status)[op][tensor];
            std::unique_lock<std::shared_mutex>{tensor_status.status_mutex};
            if (tensor_status.data_status != MemoryDataStatusType::none) throw status_error("Allocating for existing data.");

            // update tensor memory status
            tensor_status.device_address = address;
            tensor_status.data_status = MemoryDataStatusType::empty;
        }

        // emit memory event
        backend_handle.lock()->submitEvent(MemoryEvent(op, tensor, MemoryEventType::allocate));
    }


    void setMemoryDataAssigned(const std::string& op, const std::string& tensor) {
        if (!inited) throw uninited_exception();

        if (!memory_status->isTensorRegistered(op, tensor)) throw status_error("Operator or tensor not registered.");
            
        TensorStatus& tensor_status = (*memory_status)[op][tensor];
        std::unique_lock<std::shared_mutex>{tensor_status.status_mutex};
        
        switch (tensor_status.data_status) {
            case none:
            case host:
                (*logger)<<LogLevel::error<<"Assigning of operator "<<op<<": tensor "<<tensor_status.name<<" that not exists or exists on host.";
                logger->flush();
                throw status_error("Setting non-exist or host data assigned.");
                break;
            case swapin:
            case swapout:
                (*logger)<<LogLevel::warning<<"Assigning of operator "<<op<<": tensor "<<tensor_status.name<<" that is swapping.";
                logger->flush();
            case empty:
            case device:
            case coexist:
            default:
                break;
        }

        tensor_status.data_status = MemoryDataStatusType::device;

        // emit memory event
        backend_handle.lock()->submitEvent(MemoryEvent(op, tensor, MemoryEventType::write));
    }

    /**
     * setMemoryDataAcquired
     * Set the memory data is acquired, or read.
     * @param op operator name
     * @param tensor tensor name
     */
    void setMemoryDataAcquired(const std::string& op, const std::string& tensor) {
        if (!inited) throw uninited_exception();

        if (!memory_status->isTensorRegistered(op, tensor)) throw status_error("Operator or tensor not registered.");

        TensorStatus& tensor_status = (*memory_status)[op][tensor];
        std::unique_lock<std::shared_mutex>{tensor_status.status_mutex};

        switch (tensor_status.data_status) {
            case none:
            case host:
                (*logger)<<LogLevel::error<<"Acquiring of operator "<<op<<": tensor "<<tensor_status.name<<" that not exists or exists on host.";
                logger->flush();
                throw status_error("Setting non-exist or host data acquired.");
                break;
            case swapin:
            case swapout:
                (*logger)<<LogLevel::warning<<"Acquiring of operator "<<op<<": tensor "<<tensor_status.name<<" that is swapping.";
                logger->flush();
            case empty:
            case device:
            case coexist:
            default:
                break;
        }

        // emit memory event
        backend_handle.lock()->submitEvent(MemoryEvent(op, tensor, MemoryEventType::read));
    }

    void setMemoryDataAccessed(const std::string& op, const std::string& tensor) {
        if (!inited) throw uninited_exception();

        if (!memory_status->isTensorRegistered(op, tensor)) throw status_error("Operator or tensor not registered.");

        TensorStatus& tensor_status = (*memory_status)[op][tensor];
        std::unique_lock<std::shared_mutex>{tensor_status.status_mutex};

        switch (tensor_status.data_status) {
            case none:
            case host:
                (*logger)<<LogLevel::error<<"Accessing of operator "<<op<<": tensor "<<tensor_status.name<<" that not exists or exists on host.";
                logger->flush();
                throw status_error("Setting non-exist or host data accessed.");
                break;
            case swapin:
            case swapout:
                (*logger)<<LogLevel::warning<<"Accessing of operator "<<op<<": tensor "<<tensor_status.name<<" that is swapping.";
                logger->flush();
            case empty:
            case device:
            case coexist:
            default:
                break;
        }

        tensor_status.data_status = MemoryDataStatusType::device;

        // emit memory event
        backend_handle.lock()->submitEvent(MemoryEvent(op, tensor, MemoryEventType::access));
    }

    int getIteration() {
        if (!inited) throw uninited_exception();
        return 0;
    }

    void increaseIteration() {
        if (!inited) throw uninited_exception();
    }

    /**
     * isMemoryReady
     * Make sure the data of a operator is moved to the device memory before the operator is launched.
     * @return If the memory data for a operator is on the device
     */
    bool isMemoryReady(const std::string& op) {
        if (!inited) throw uninited_exception();

        for (const auto& x : memory_status->at(op)) {
            if (x.second.data_status != MemoryDataStatusType::device) return false;
        }
        return true;
    }

    /**
     * waitData
     * Make sure the data of a operator is moved to the device memory before the operator is launched.
     */
    void waitData(const std::string& op) {
        assert(0);
    }

    void releaseData() {
        assert(0);
    }

    /**
     * withData
     * Process a operator with its memory data pinned in device memory
     * @param op name of the operator
     * @param func the function that process the operator
     */
    void withData(const std::string& op, std::function<void(void)> func) {
        if (!inited) throw uninited_exception();

        if (!memory_status->isOperatorRegistered(op)) throw status_error("Operator not registered.");

        // Uniquely lock the operator to prevent further scheduling.
        std::unordered_set<std::string> tensors_not_ready;
        auto& op_status = (*memory_status)[op];
        std::unique_lock(op_status.status_mutex);

        for (auto& x : op_status) tensors_not_ready.insert(x.first);

        bool tensors_ready = false;
        while (!tensors_ready) {
            tensors_ready = true;

            auto p = tensors_not_ready.begin();
            while (p != tensors_not_ready.end()) {
                auto& tensor_status = op_status[*p];
                switch (tensor_status.data_status) {
                    case MemoryDataStatusType::empty:
                    case MemoryDataStatusType::device:
                    case MemoryDataStatusType::coexist:
                        p = tensors_not_ready.erase(p);
                        break;
                    case MemoryDataStatusType::none:
                        throw status_error("Accessing non-exist data.");
                    case MemoryDataStatusType::host:
                    case MemoryDataStatusType::swapin:
                    case MemoryDataStatusType::swapout:
                    default:
                        tensors_ready = false;

                        if (tensor_status.data_status != MemoryDataStatusType::host) throw status_error("Requesting swapping-in for data not on host.");

                        tensor_status.data_status = MemoryDataStatusType::swapin;
                        auto device_address = memory_manager->swapIn(tensor_status.host_address, tensor_status.size);
                        tensor_status.device_address = device_address;
                        tensor_status.data_status = MemoryDataStatusType::device;

                        (*logger)<<LogLevel::debug<<"Operator "<<op<<": tensor "<<tensor_status.name<<" swapped in. (Memory access)";
                        logger->flush();

                        ++p;
                        break;
                }
            }
        }

        func();
    }

    /**
     * waitMemory
     * Wait for available memory.
     * Memory insufficent is an emergency event, hence an independent method is provided.
     */
    void waitMemory(size_t size) {
        if (!inited) throw uninited_exception();
        executor.lock()->waitMemory(size);
    }

    void freeMemory(const std::string& op, const std::string& tensor) {
        if (!inited) throw uninited_exception();

        if (!memory_status->isTensorRegistered(op, tensor)) throw status_error("Operator or tensor not registered.");
        
        {
            TensorStatus& tensor_status = (*memory_status)[op][tensor];
            std::unique_lock<std::shared_mutex>{tensor_status.status_mutex};

            switch (tensor_status.data_status) {
                case MemoryDataStatusType::none:
                    throw status_error("Freeing non-exist data.");
                    break;
                case MemoryDataStatusType::empty:
                case MemoryDataStatusType::device:
                    memory_manager->freeDevice(tensor_status.device_address);
                    tensor_status.device_address = nullptr;
                    tensor_status.data_status = MemoryDataStatusType::none;
                    break;
                case MemoryDataStatusType::host:
                    memory_manager->freeHost(tensor_status.host_address);
                    tensor_status.host_address = nullptr;
                    tensor_status.data_status = MemoryDataStatusType::none;
                    break;
                case MemoryDataStatusType::coexist:
                    memory_manager->freeDevice(tensor_status.device_address);
                    memory_manager->freeHost(tensor_status.host_address);
                    tensor_status.device_address = nullptr;
                    tensor_status.host_address = nullptr;
                    tensor_status.data_status = MemoryDataStatusType::none;
                    break;
                case MemoryDataStatusType::swapin:
                    assert(0);
                case MemoryDataStatusType::swapout:
                    assert(0);
                default:
                    break;
            }
        }

        // emit memory event
        backend_handle.lock()->submitEvent(MemoryEvent(op, tensor, MemoryEventType::free));
    }

    void setMemoryDataFreed(const std::string& op, const std::string& tensor) {
        if (!inited) throw uninited_exception();

        if (!memory_status->isTensorRegistered(op, tensor)) throw status_error("Operator or tensor not registered.");
        
        {
            TensorStatus& tensor_status = (*memory_status)[op][tensor];
            std::unique_lock<std::shared_mutex>{tensor_status.status_mutex};
            if (tensor_status.data_status == MemoryDataStatusType::none) throw status_error("Freeing non-exist data.");

            tensor_status.device_address = nullptr;
            tensor_status.host_address = nullptr;
            tensor_status.data_status = MemoryDataStatusType::none;

        }

        // emit memory event
        backend_handle.lock()->submitEvent(MemoryEvent(op, tensor, MemoryEventType::free));
    }

    void terminate() {
        if (!inited) throw uninited_exception();
        logger = nullptr;
        
        inited = false;
        executor.lock()->terminate();
    }

    ~MemorySession() {
        if (inited) terminate();
    }
};  // struct MemorySession

}   // namespace mori