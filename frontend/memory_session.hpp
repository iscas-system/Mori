#pragma once

#include "frontend/backend_handle.hpp"
#include "frontend/memory_schedule_executor.hpp"
#include "frontend/memory_operation_executor.hpp"
#include "frontend/callbacks.hpp"
#include "includes/memory_status.hpp"
#include "includes/memory_event.hpp"
#include "includes/application_stage.hpp"

namespace mori {

struct MemoryRequest final {
private:
    friend class MemorySession;

private:
    std::string op;

    status::MemoryStatus& status;
    MemoryScheduleExecutor& sch_executor;
    MemoryOperationExecutor& op_executor;
    Callbacks& callbacks;
    std::weak_ptr<BackendHandle> backend_handle;
    Logger* logger;

    ApplicationStage stage;

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

    MemoryRequest(const std::string& _op, status::MemoryStatus& _status, MemoryScheduleExecutor& _sch_executor, MemoryOperationExecutor& _op_executor, Callbacks& _callbacks, std::weak_ptr<BackendHandle>& _backend_handle, Logger* _logger, ApplicationStage _stage): op(_op), status(_status), sch_executor(_sch_executor), op_executor(_op_executor), callbacks(_callbacks), backend_handle(_backend_handle), logger(_logger), stage(_stage) {}

public:
    MemoryRequest(const MemoryRequest&) = delete;
    MemoryRequest(MemoryRequest&& _request): status(_request.status), sch_executor(_request.sch_executor), op_executor(_request.op_executor), callbacks(_request.callbacks), logger(_request.logger) {
        requested_tensors = std::move(_request.requested_tensors);
        backend_handle = std::move(_request.backend_handle);
    }

    void waitTensor(const std::string& tensor) {
        // If the tensor waited, it would have been locked on device memory.
        if (isTensorWaited(tensor)) return;

        auto p = requested_tensors.emplace(tensor, status.referenceTensor(tensor));
        status::TensorPres& pres = p.first->second;
        // Do not swap in tensor that already on device.
        if (pres.getSize() == pres.getRemainingSize()) return;
        
        op_executor.swapIn(pres, pres.getSize() - pres.getRemainingSize());
        
        if (callbacks.count(CallbackStage::postSwapIn)) callbacks.at(CallbackStage::postSwapIn)(tensor, pres.getSection(0).device_address);
        
        (*logger) << LogLevel::debug << "Tensor: " << tensor << " swapped in. (Memory access)";
        logger->flush();
        backend_handle.lock()->submitEvent(events::MemoryEvent(op, tensor, pres.getSize() - pres.getRemainingSize(), events::MemoryEventType::swapin, stage));
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
        backend_handle.lock()->submitEvent(events::MemoryEvent(op, tensor, pres.getSize() ,events::MemoryEventType::write, stage));
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
        backend_handle.lock()->submitEvent(events::MemoryEvent(op, tensor, pres.getSize(), events::MemoryEventType::read, stage));
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
        backend_handle.lock()->submitEvent(events::MemoryEvent(op, tensor, pres.getSize(), events::MemoryEventType::access, stage));
    }

    void release() {
        for (auto &x : requested_tensors)
            x.second.release();

        sch_executor.setOperatorFinished(op);
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

    std::weak_ptr<BackendHandle> backend_handle;

    status::MemoryStatus& status;
    MemoryScheduleExecutor& sch_executor;
    MemoryOperationExecutor op_executor;

    Callbacks callbacks;

    Logger* logger;

    ApplicationStage stage = ApplicationStage::forward;

    void setBackendHandle(const std::weak_ptr<BackendHandle>& _backend_handle) {
        backend_handle = _backend_handle;
    }
    void setMemoryManager(MemoryManager* _memory_manager) {
        op_executor.setMemoryManager(_memory_manager);
    }
    void setLogger(Logger* _logger) {
        logger = _logger;
    }
    void setCallback(CallbackStage stage, const std::function<int(const std::string&, void*)>& callback) {
        callbacks.emplace(stage, callback);
    }

public:
    MemorySession(const Context& _context, MemoryScheduleExecutor& _executor, status::MemoryStatus& _status): context(_context), sch_executor(_executor), status(_status) {}

    int getIteration() const { return 0; }
    
    void setIteration(int iteration) {
        sch_executor.setIteration(iteration);
        backend_handle.lock()->setIteration(iteration);
    }
    
    /**
     * @brief Set the new iteration is ready.
     * @note  This method will synchronize with the schedule executor, to assure all the swappings are finished.
    */
    void newIteration() {
        // Reset stage
        stage = ApplicationStage::forward;

        sch_executor.newIteration();
        backend_handle.lock()->newIteration();
    }

    /**
     * @brief Set the forward progagation, or half of the iteration is executed.
     * @note  This method will synchronize with the schedule executor, to assure all the swap-outs are finished.
    */
    void halfIteration() {
        // Reverse stage
        if (stage == ApplicationStage::forward) stage = ApplicationStage::backward;
        else stage = ApplicationStage::forward;

        sch_executor.halfIteration();
        // backend_handle.lock()->
    }

    /**
     * @brief Set the memory data is allocated.
     * @param op operator name
     * @param tensor tensor name
     * @param address tensor address
     */
    void setMemoryDataAllocated(const std::string& op, const std::string& tensor, void* address) {
        if (address == nullptr) throw memory_device_insufficience();
        if (!status.isTensorRegistered(tensor)) throw status_exception("Tensor not registered.");

        status::TensorPres pres = status.referenceTensor(tensor);
        pres.setAllocated(address);

        // emit memory event
        backend_handle.lock()->submitEvent(events::MemoryEvent(op, tensor, pres.getSize(), events::MemoryEventType::allocate, stage));
    }

    void setMemoryDataAllocated(const std::string& tensor, void* address) { setMemoryDataAllocated("", tensor, address); }

    /**
     * @brief  Assure the data of a operator is moved to the device memory before the operator is launched.
     * @return MemoryRequest object
     */
    MemoryRequest createRequest(const std::string& op = "") {
        // if (!status.isOperatorRegistered(op)) throw status_exception("Operator not registered.");
        MemoryRequest re(op, status, sch_executor, op_executor, callbacks, backend_handle, logger, stage);
        return re;
    }

    /**
     * @brief Wait for available memory. Memory insufficent is an emergency event, hence an independent method is provided.
     * @param size Memory size that should be released.
     * @note Currently this method adopts a FIFO strategy that the firstly forward-propagating operator will be firstly released. 
     */
    void waitMemory(size_t size) {
        size_t released_size = 0;

        for (auto &op_name : status.getExecutionOrder()) {
            status::OperatorPres op_pres = status.referenceOperator(op_name);
            // Forward propagation and backward propagation share the same set of operators.
            if (op_pres.isBackwardPropagation()) continue;
            (*logger) << LogLevel::debug << "Considering " << op_name;
            logger->flush();

            for (auto &tensor_name : op_pres.getTensors()) { 
                status::TensorPres tensor_pres = status.referenceTensor(tensor_name);
                // Do not swap out persistant tensors.
                if (tensor_pres.isPersistant()) continue;
                // Do not swap out tensors that already host-only.
                if (tensor_pres.getRemainingSize() == 0) continue;
                int releasing_size = 0;
                if (tensor_pres.getRemainingSize() + released_size <= size) releasing_size = tensor_pres.getRemainingSize();
                else releasing_size = size - released_size;
                op_executor.swapOut(tensor_pres, releasing_size);

                if (callbacks.count(CallbackStage::postSwapOut)) callbacks.at(CallbackStage::postSwapOut)(tensor_name, tensor_pres.getSection(0).host_address);
                (*logger) << LogLevel::debug << "Releasing " << tensor_pres.getName();
                logger->flush();

                backend_handle.lock()->submitEvent(events::MemoryEvent(op_name, tensor_name, releasing_size, events::MemoryEventType::swapout, stage));

                released_size += releasing_size;
                if (released_size >= size) break;
            }
            if (released_size >= size) break;
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
     * @param op operator name
     * @param tensor tensor name
     */
    void setMemoryDataFreed(const std::string& op, const std::string& tensor) {
        if (!status.isTensorRegistered(tensor)) throw status_exception("Operator or tensor not registered.");

        status::TensorPres pres = status.referenceTensor(tensor);
        pres.setFreed();

        // emit memory event
        backend_handle.lock()->submitEvent(events::MemoryEvent(op, tensor, pres.getSize(), events::MemoryEventType::free, stage));
    }

    void setMemoryDataFreed(const std::string& tensor) { setMemoryDataFreed("", tensor); }

    ~MemorySession() = default;
};  // struct MemorySession

}   // namespace mori