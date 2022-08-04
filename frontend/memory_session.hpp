#pragma once

#include "includes/stdlibs.hpp"

#include "frontend/backend_handle.hpp"
#include "frontend/memory_schedule_executor.hpp"
#include "includes/memory_status.hpp"
#include "includes/memory_event.hpp"

namespace mori {

struct MemoryRequest final {
protected:
    Context context;

    std::string op_name;

    MemoryStatuses& memory_status;
    std::weak_ptr<BackendHandle> backend_handle;
    MemoryManager* memory_manager;
    Logger* logger;

    std::unordered_multimap<std::string, std::string> requested_tensors;
    bool waiting = false;

    /**
     * isTensorWaited
     * Check if tensor has been selected and waited.
     * @param op operator name
     * @param tensor tensor name
     * @return if the tensor has been selected and waited.
     */
    bool isTensorWaited(const std::string& op, const std::string& tensor) {
        // Actually we do not want a operator - tensor to be submitted multiple times.
        auto range = requested_tensors.equal_range(op);
        if (range.first == range.second) {
            // first == second == end(), operator not registered.
            return false;
        }
        
        for (auto p = range.first; p != range.second; ++p) {
            assert(p->first == op);
            if (p->second == tensor) {
                return true;
            }
        }

        return false;
    }

    /**
     * waitTensors
     * Wait the tensors to be located in device memory.
     * Internal interface
     * @param tensors tensors to be waited
     */
    void waitTensors(std::vector<std::pair<std::string, std::string>>& tensors) {
        bool tensors_ready = false;
        while (!tensors_ready) {
            tensors_ready = true;

            auto p = tensors.begin();
            while (p != tensors.end()) {
                auto& tensor_status = memory_status[p->first][p->second];
                switch (tensor_status.data_status) {
                    case MemoryDataStatusType::empty:
                    case MemoryDataStatusType::device:
                    case MemoryDataStatusType::coexist:
                        p = tensors.erase(p);
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

                        (*logger)<<LogLevel::debug<<"Operator "<<p->first<<": tensor "<<tensor_status.name<<" swapped in. (Memory access)";
                        logger->flush();

                        ++p;
                        break;
                }
            }
        }
    }

public:
    MemoryRequest(const Context& _context, const std::string& _op_name, MemoryStatuses& _memory_status, std::weak_ptr<BackendHandle>& _backend_handle, MemoryManager* _memory_manager, Logger* _logger): context(_context), op_name(_op_name), memory_status(_memory_status), backend_handle(_backend_handle), memory_manager(_memory_manager), logger(_logger) {}

    MemoryRequest(const MemoryRequest&) = delete;
    MemoryRequest(MemoryRequest&& _request): memory_status(_request.memory_status), memory_manager(_request.memory_manager), logger(_request.logger) {
        context = std::move(_request.context);
        op_name = std::move(_request.op_name);
        requested_tensors = std::move(_request.requested_tensors);
        backend_handle = std::move(_request.backend_handle);
    }

    MemoryRequest& operator=(const MemoryRequest&) = delete;
    MemoryRequest& operator=(MemoryRequest&& _request) = delete;

    void waitData() {
        if (waiting) throw status_error("Memory request already has its tensors waited.");

        // Step 1: set waiting data of this operator.
        auto& op_status = memory_status[op_name];
        op_status.status_mutex.lock();

        std::vector<std::pair<std::string, std::string>> tensors_not_ready;
        for (auto &x : op_status) {
            x.second.status_mutex.lock();
            tensors_not_ready.push_back(std::make_pair(op_name, x.second.name));
            requested_tensors.insert(std::make_pair(op_status.name, x.first));
        }

        // Step 2: set waiting data of dependent tensors
        // Dependent tensors belong to other operators, but should be fetched for this operator.
        for (auto &x : op_status.prev_deps) {
            auto& tensor_status = memory_status[x.first][x.second];
            tensor_status.status_mutex.lock();
            tensors_not_ready.push_back(std::make_pair(x.first, x.second));
            requested_tensors.insert(std::make_pair(x.first, x.second));
        }
        for (auto &x : op_status.post_deps) {
            auto& tensor_status = memory_status[x.first][x.second];
            tensor_status.status_mutex.lock();
            tensors_not_ready.push_back(std::make_pair(x.first, x.second));
            requested_tensors.insert(std::make_pair(x.first, x.second));
        }

        // Step 3: waiting data.
        waitTensors(tensors_not_ready);

        // for (auto &s : op_status.getPrevs()) {
        //     if (context.signal("operators.external_inputs"))
        //         waitTensors(memory_status[s], MemoryType::inout);
        //     if (context.signal("operators.external_weights"))
        //         waitTensors(memory_status[s], MemoryType::weight);
        //     if (context.signal("operators.external_constants"))
        //         waitTensors(memory_status[s], MemoryType::constant);
        // }

        waiting = true;
    }

    /**
     * setMemoryDataAssigned
     * Set the memory data is assigned, or written.
     * The operator name should be provided since the outputs of the prev operators may be accessed.
     * @param op operator name
     * @param tensor tensor name
     */
    void setMemoryDataAssigned(const std::string& op, const std::string& tensor) {
        if (!waiting) throw uninited_exception();
        if (!isTensorWaited(op, tensor)) throw status_error("Operator or tensor not waited.");
        
        // Do not acquire locks here since the tensor is awaited.
        TensorStatus& tensor_status = memory_status[op][tensor];
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
     * setMemoryDataAssigned
     * Set the memory data is assigned, or written.
     * The operator name comes from the memory request.
     * @param tensor tensor name
     */
    void setMemoryDataAssigned(const std::string& tensor) { setMemoryDataAccessed(op_name, tensor); }

    /**
     * setMemoryDataAcquired
     * Set the memory data is acquired, or read.
     * The operator name should be provided since the outputs of the prev operators may be accessed.
     * @param op operator name
     * @param tensor tensor name
     */
    void setMemoryDataAcquired(const std::string& op, const std::string& tensor) {
        if (!waiting) throw uninited_exception();
        if (!isTensorWaited(op, tensor)) throw status_error("Operator or tensor not waited.");

        TensorStatus& tensor_status = memory_status[op][tensor];
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

    /**
     * setMemoryDataAcquired
     * Set the memory data is acquired, or read.
     * The operator name comes from the memory request.
     * @param tensor tensor name
     */
    void setMemoryDataAcquired(const std::string& tensor) { setMemoryDataAcquired(op_name, tensor); }

    /**
     * setMemoryDataAccessed
     * Set the memory data is accessed.
     * The operator name should be provided since the outputs of the prev operators may be accessed.
     * @param op operator name
     * @param tensor tensor name
     */
    void setMemoryDataAccessed(const std::string& op, const std::string& tensor) {
        if (!waiting) throw uninited_exception();
        if (!isTensorWaited(op, tensor)) throw status_error("Operator or tensor not waited.");

        TensorStatus& tensor_status = memory_status[op][tensor];
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


    /**
     * setMemoryDataAccessed
     * Set the memory data is accessed.
     * The operator name comes from the memory request.
     * @param tensor tensor name
     */
    void setMemoryDataAccessed(const std::string& tensor) { setMemoryDataAccessed(op_name, tensor); }

    void releaseData() {
        if (!waiting) throw uninited_exception();

        for (auto &x : requested_tensors) {
            auto& tensor_status = memory_status[x.first][x.second];
            if (tensor_status.data_status != MemoryDataStatusType::device && tensor_status.data_status != MemoryDataStatusType::coexist) {
                (*logger) << LogLevel::error << "Operator: " << x.first << " status error, tensor: " << tensor_status.name << " not on device.";
                logger->flush();
                throw status_error("Releasing tensor not on device.");
            }
            tensor_status.status_mutex.unlock();
        }

        memory_status[op_name].status_mutex.unlock();

        waiting = false;
    }

    ~MemoryRequest() {
        if (waiting) releaseData();
    }

};  // struct MemoryRequest

/**
 * MemorySession
 * Management of a memory session, which is a complete memory lifecycle of a training iteration.
 */
struct MemorySession final {
protected:
    Context context;

    std::atomic<int> step;

    std::weak_ptr<BackendHandle> backend_handle;

    std::weak_ptr<MemoryScheduleExecutor> executor;

    MemoryStatuses& memory_status;
    MemoryManager* memory_manager;

    bool inited = false;

    Logger* logger;

public:
    MemorySession(const Context& _context, MemoryStatuses& _memory_status): context(_context), memory_status(_memory_status) {}

    void setBackendHandle(const std::weak_ptr<BackendHandle>& _backend_handle) {
        if (inited) throw inited_exception();
        backend_handle = _backend_handle;
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
        if (executor.lock() == nullptr) throw status_error("Memory schedule executor not assigned.");

        executor.lock()->init();

        inited = true;

        logger->submit(LogLevel::info, "Mori session inited.");
    }

    inline bool isInited() const {return inited;}

    /**
     * allocateMemory
     * Allocate memory on device for tensor.
     * @param op operator name
     * @param tensor tensor name
     */
    void* allocateMemory(const std::string& op, const std::string& tensor) {
        if (!inited) throw uninited_exception();

        if (!memory_status.isTensorRegistered(op, tensor)) throw status_error("Operator or tensor not registered.");

        void* dev_addr = nullptr;
        int retry_count = 2;
        {
            TensorStatus& tensor_status = memory_status[op][tensor];
            std::unique_lock<std::shared_mutex>{tensor_status.status_mutex};
            if (tensor_status.data_status != MemoryDataStatusType::none) throw status_error("Allocating for existing data.");

            dev_addr = memory_manager->allocate(tensor_status.size);

            while (dev_addr == nullptr && retry_count > 0) {
                (*logger) << LogLevel::debug << "Memory allocation failed.";
                logger->flush();

                waitMemory(tensor_status.size);
                dev_addr = memory_manager->allocate(tensor_status.size);
                --retry_count;
            }

            // update tensor memory status
            tensor_status.device_address = dev_addr;
            tensor_status.data_status = MemoryDataStatusType::empty;
        }

        // emit memory event
        backend_handle.lock()->submitEvent(MemoryEvent(op, tensor, MemoryEventType::allocate));

        return dev_addr;

    }

    /**
     * setMemoryDataAllocated
     * Set the memory data is allocated.
     * @param op operator name
     * @param tensor tensor name
     * @param address tensor address
     */
    void setMemoryDataAllocated(const std::string& op, const std::string& tensor, void* address) {
        if (!inited) throw uninited_exception();

        if (!memory_status.isTensorRegistered(op, tensor)) throw status_error("Operator or tensor not registered.");

        {
            TensorStatus& tensor_status = memory_status[op][tensor];
            std::unique_lock<std::shared_mutex>{tensor_status.status_mutex};
            if (tensor_status.data_status != MemoryDataStatusType::none) throw status_error("Allocating for existing data.");

            // update tensor memory status
            tensor_status.device_address = address;
            tensor_status.data_status = MemoryDataStatusType::empty;
        }

        // emit memory event
        backend_handle.lock()->submitEvent(MemoryEvent(op, tensor, MemoryEventType::allocate));
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

        for (const auto& x : memory_status.at(op)) {
            if (x.second.data_status != MemoryDataStatusType::device 
                && x.second.data_status != MemoryDataStatusType::coexist
                && x.second.data_status != MemoryDataStatusType::empty) return false;
        }
        return true;
    }

    /**
     * waitData
     * Make sure the data of a operator is moved to the device memory before the operator is launched.
     * @param op operator name
     */
    MemoryRequest waitData(const std::string& op) {
        if (!inited) throw uninited_exception();

        if (!memory_status.isOperatorRegistered(op)) throw status_error("Operator not registered.");

        MemoryRequest re(context, op, memory_status, backend_handle, memory_manager,logger);
        re.waitData();

        return re;
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

        if (!memory_status.isTensorRegistered(op, tensor)) throw status_error("Operator or tensor not registered.");
        
        {
            TensorStatus& tensor_status = memory_status[op][tensor];
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

        if (!memory_status.isTensorRegistered(op, tensor)) throw status_error("Operator or tensor not registered.");
        
        {
            TensorStatus& tensor_status = memory_status[op][tensor];
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
        
        inited = false;
        executor.lock()->terminate();
    }

    ~MemorySession() {
        if (inited) terminate();

        logger = nullptr;
    }
};  // struct MemorySession

}   // namespace mori