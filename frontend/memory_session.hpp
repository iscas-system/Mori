#pragma once

#include "frontend/backend_handle.hpp"
#include "frontend/memory_operation_executor.hpp"
#include "frontend/callbacks.hpp"
#include "includes/memory_status.hpp"
#include "includes/memory_event.hpp"

namespace mori {

struct MemoryRequest final {
protected:
    status::MemoryStatus& status;
    MemoryOperationExecutor& executor;
    Callbacks& callbacks;
    std::weak_ptr<BackendHandle> backend_handle;
    Logger* logger;

    std::unordered_map<std::string, status::TensorPres> requested_tensors;
    std::atomic<bool> waiting = true;

    /**
     * isTensorWaited
     * Check if tensor has been selected and waited.
     * @param tensor tensor name
     * @return if the tensor has been selected and waited.
     */
    bool isTensorWaited(const std::string& tensor) {
        return requested_tensors.find(tensor) != requested_tensors.end();
    }

public:
    MemoryRequest(status::MemoryStatus& _status, MemoryOperationExecutor& _executor, Callbacks& _callbacks, std::weak_ptr<BackendHandle>& _backend_handle, Logger* _logger): status(_status), executor(_executor), callbacks(_callbacks), backend_handle(_backend_handle), logger(_logger) {}

    MemoryRequest(const MemoryRequest&) = delete;
    MemoryRequest(MemoryRequest&& _request): status(_request.status), executor(_request.executor), callbacks(_request.callbacks), logger(_request.logger) {
        requested_tensors = std::move(_request.requested_tensors);
        backend_handle = std::move(_request.backend_handle);
    }

    MemoryRequest& operator=(const MemoryRequest&) = delete;
    MemoryRequest& operator=(MemoryRequest&& _request) = delete;

    void waitTensor(const std::string& tensor) {
        // If the tensor waited, it would have been locked on device memory.
        if (isTensorWaited(tensor)) return;

        auto p = requested_tensors.emplace(tensor, status.referenceTensor(tensor));
        status::TensorPres& pres = p.first->second;
        switch (pres.getStatus()) {
            case status::MemoryStatusType::none:
                throw status_exception("Accessing non-exist data.");
            case status::MemoryStatusType::host:
                executor.swapIn(pres);
                if (callbacks.count(CallbackStage::postSwapIn)) callbacks.at(CallbackStage::postSwapIn)(tensor, pres.getDevicePointer(0));
                // Swap in = copy in + host freed.
                (*logger) << LogLevel::debug << "Tensor: " << tensor << " swapped in. (Memory access)";
                logger->flush();
                backend_handle.lock()->submitEvent(events::MemoryEvent(tensor, events::MemoryEventType::swapin));
                break;
            default:
                break;
        }
    }

    // /**
    //  * waitOperator
    //  * Wait all the tensors of an operator to be located in device memory.
    //  * @param op operator to be waited
    //  */
    // void waitTensors(const std::string& op) {
    //     for (auto &s : status.)
    //         waitTensor(x.first, x.second);
    // }

    /**
     * setMemoryDataAssigned
     * Set the memory data is assigned, or written.
     * The operator name should be provided since the outputs of the prev operators may be accessed.
     * @param op operator name
     * @param tensor tensor name
     */
    void setMemoryDataAssigned(const std::string& tensor) {
        if (!waiting) throw uninited_exception();
        if (!isTensorWaited(tensor)) throw status_exception("Tensor not waited.");
        
        // Do not acquire locks here since the tensor is awaited.
        // Tensor exists since isTensorWaited(tensor) is true.
        status::TensorPres& pres = requested_tensors.at(tensor);
        pres.setAssigned();

        // emit memory event
        backend_handle.lock()->submitEvent(events::MemoryEvent(tensor, events::MemoryEventType::write));
    }

    /**
     * setMemoryDataAcquired
     * Set the memory data is acquired, or read.
     * The operator name should be provided since the outputs of the prev operators may be accessed.
     * @param op operator name
     * @param tensor tensor name
     */
    void setMemoryDataAcquired(const std::string& tensor) {
        if (!waiting) throw uninited_exception();
        if (!isTensorWaited(tensor)) throw status_exception("Operator or tensor not waited.");

        status::TensorPres& pres = requested_tensors.at(tensor);
        pres.setAcquired();

        // emit memory event
        backend_handle.lock()->submitEvent(events::MemoryEvent(tensor, events::MemoryEventType::read));
    }

    /**
     * setMemoryDataAccessed
     * Set the memory data is accessed.
     * The operator name should be provided since the outputs of the prev operators may be accessed.
     * @param tensor tensor name
     */
    void setMemoryDataAccessed(const std::string& tensor) {
        if (!waiting) throw uninited_exception();
        if (!isTensorWaited(tensor)) throw status_exception("Tensor not waited.");
        
        status::TensorPres& pres = requested_tensors.at(tensor);
        pres.setAccessed();

        // emit memory event
        backend_handle.lock()->submitEvent(events::MemoryEvent(tensor, events::MemoryEventType::access));
    }

    void release() {
        for (auto &x : requested_tensors)
            x.second.release();
        waiting = false;
    }

    ~MemoryRequest() {
        if (waiting) release();
    }

};  // struct MemoryRequest

struct Frontend;

/**
 * MemorySession
 * Management of a memory session, which is a complete memory lifecycle of a training iteration.
 */
struct MemorySession final {
private:
    friend struct Frontend;

private:
    Context context;

    std::atomic<int> step;

    std::weak_ptr<BackendHandle> backend_handle;

    status::MemoryStatus& status;
    MemoryOperationExecutor executor;

    Callbacks callbacks;

    Logger* logger;

    void setBackendHandle(const std::weak_ptr<BackendHandle>& _backend_handle) {
        backend_handle = _backend_handle;
    }
    void setMemoryManager(MemoryManager* _memory_manager) {
        executor.setMemoryManager(_memory_manager);
    }
    void setLogger(Logger* _logger) {
        logger = _logger;
    }
    void setCallback(CallbackStage stage, const std::function<int(const std::string&, void*)>& callback) {
        callbacks.emplace(stage, callback);
    }

public:
    MemorySession(const Context& _context, status::MemoryStatus& _status): context(_context), status(_status) {}

    /**
     * @brief Set the memory data is allocated.
     * @param tensor tensor name
     * @param address tensor address
     */
    void setMemoryDataAllocated(const std::string& tensor, void* address) {
        if (address == nullptr) throw memory_device_insufficience();
        if (!status.isTensorRegistered(tensor)) throw status_exception("Tensor not registered.");

        status::TensorPres pres = status.referenceTensor(tensor);
        pres.setAllocated(address);

        // emit memory event
        backend_handle.lock()->submitEvent(events::MemoryEvent(tensor, events::MemoryEventType::allocate));
    }

    inline int getIteration() const { return 0; }
    inline void setIteration(int _iteration) {}
    inline void increaseIteration() {}

    /**
     * @brief  Assure the data of a operator is moved to the device memory before the operator is launched.
     * @return MemoryRequest object
     */
    MemoryRequest createRequest() {
        // if (!status.isOperatorRegistered(op)) throw status_exception("Operator not registered.");
        MemoryRequest re(status, executor, callbacks, backend_handle, logger);
        return re;
    }

    /**
     * @brief Wait for available memory. Memory insufficent is an emergency event, hence an independent method is provided.
     * @param size Memory size that should be released.
     * @note Currently this method adopts a FIFO strategy that the firstly forward-propagating operator will be firstly released. 
     */
    void waitMemory(size_t size) {
        size_t released_size = 0;
        std::deque<std::string> ops;
        for (auto &op_name : status.getExecutionOrder()) {
            status::OperatorPres op_pres = status.referenceOperator(op_name);
            // Forward propagation and backward propagation share the same set of operators.
            if (op_pres.isBackwardPropagation()) continue;
            (*logger) << LogLevel::debug << "Considering " << op_name;
            logger->flush();
            for (auto &tensor_name : op_pres.getTensors()) { 
                status::TensorPres tensor_pres = status.referenceTensor(tensor_name);
                if (tensor_pres.isPersistant()) continue;
                switch (tensor_pres.getStatus()) {
                    case status::MemoryStatusType::coexist:
                        executor.freeDevice(tensor_pres);
                        break;
                    case status::MemoryStatusType::device:
                        executor.swapOut(tensor_pres);
                        break;
                    case status::MemoryStatusType::host:
                    case status::MemoryStatusType::empty:
                    default:
                        continue;
                }
                if (callbacks.count(CallbackStage::postSwapOut)) callbacks.at(CallbackStage::postSwapOut)(tensor_name, tensor_pres.getHostPointer(0));
                (*logger) << LogLevel::debug << "Releasing " << tensor_pres.getName();
                logger->flush();

                backend_handle.lock()->submitEvent(events::MemoryEvent(tensor_name, events::MemoryEventType::swapout));

                released_size += tensor_pres.getSize();
                if (released_size >= size) break;
            }
        }

        if (released_size >= size) {
            // (*logger) << LogLevel::info << "Memory insufficient, mori releases " << released_size << " of memory.";
            // logger->flush();
        } else {
            // Mori wait memory failed.
            (*logger) << LogLevel::info << "Mori memory releasing failed, " << " unmet.";
            logger->flush();
        }
    }

    /**
     * @brief Set the memory data is freed.
     * @param tensor tensor name
     */
    void setMemoryDataFreed(const std::string& tensor) {
        if (!status.isTensorRegistered(tensor)) throw status_exception("Operator or tensor not registered.");

        status::TensorPres pres = status.referenceTensor(tensor);
        pres.setFreed();

        // emit memory event
        backend_handle.lock()->submitEvent(events::MemoryEvent(tensor, events::MemoryEventType::free));
    }

    ~MemorySession() = default;
};  // struct MemorySession

}   // namespace mori