#pragma once

#include "frontend/backend_handle.hpp"
#include "frontend/memory_schedule_executor.hpp"
#include "frontend/memory_defragmentation_executor.hpp"
#include "frontend/memory_operation_executor.hpp"
#include "frontend/callbacks.hpp"
#include "includes/memory_status.hpp"
#include "includes/memory_event.hpp"
#include "includes/symbols.hpp"
#include "includes/presentation.hpp"

namespace mori {

/**
 * MemorySession
 * Management of a memory session, which is a complete memory lifecycle of a training iteration.
 */
struct MemorySession final {
private:
    friend struct Frontend;

public:
    using MemoryFunction = std::function<bool()>;

public:
    struct Request final {
    private:
        friend class MemorySession;

    private:
        MemorySession& session;

    private:
        std::string op;

        ApplicationStage stage;

        std::unordered_map<std::string, status::TensorPres> requested_tensors;
        std::atomic<bool> waiting = true;
        std::atomic<bool> executing = false;

        /**
         * isTensorWaited
         * Check if tensor has been selected and waited.
         * @param tensor tensor name
         * @return if the tensor has been selected and waited.
         */
        bool isTensorWaited(const std::string& tensor) {
            return requested_tensors.find(tensor) != requested_tensors.end();
        }

        Request(MemorySession& _session, const std::string& _op, ApplicationStage _stage): session(_session), op(_op), stage(_stage) {}

    public:
        Request(const Request&) = delete;
        Request(Request&& _request): session(_request.session) {
            op = std::move(_request.op);
            stage = _request.stage;
        }

        void waitTensor(const std::string& tensor) {
            if (!waiting) throw uninited_exception();
            if (executing) throw inited_exception();

            // If the tensor waited, it would have been locked on device memory.
            if (isTensorWaited(tensor)) return;

            auto p = requested_tensors.emplace(tensor, session.status.referenceTensor(tensor));
            status::TensorPres& pres = p.first->second;
            // Do not swap in tensor that already on device.
            if (pres.isDeviceAllLocated()) return;
            
            size_t acquiring_size = pres.getSize() - pres.getDeviceSize();
            try {
                session.op_executor.copyIn(pres, pres.getSize() - pres.getDeviceSize());
            } catch(memory_device_insufficience& e) {
                // Memory on device not insufficience.
                size_t released_size = session.waitMemory(pres.getSize(), [this, &pres]() {
                    try {
                        session.op_executor.copyIn(pres, pres.getSize() - pres.getDeviceSize());
                    } catch(memory_device_insufficience& e) {
                        return false;
                    }
                    return true;
                });
                if (!pres.isDeviceAllLocated()) throw e;
            }
            assert(pres.isDeviceAllLocated());

            if (session.callbacks.count(CallbackStage::postSwapIn)) session.callbacks.at(CallbackStage::postSwapIn)(tensor, pres.getSection(0).device_address);
            (*session.logger) << LogLevel::debug << "Operator: " << op << ", tensor: " << tensor << " swapped in. (Memory access)" << endl;
            session.backend_handle.lock()->submitEvent(events::MemoryEvent(op, tensor, acquiring_size, events::MemoryEventType::swapin, stage));
        }

        // /**
        //  * waitOperator
        //  * Wait all the tensors of an operator to be located in device memory.
        //  * @param op operator to be waited
        //  */
        // void waitOperator(const std::string& op) {
        //     for (auto &s : status.)
        //         waitTensor(x.first, x.second);
        // }

        void setOperationStarted() {
            if (!waiting) throw uninited_exception();
            if (executing) throw inited_exception();

            session.sch_executor.setOperatorStarted(op);
            session.backend_handle.lock()->submitEvent(events::ExecutionEvent(op, events::ExecutionEventType::request, stage));

            executing = true;
        }

        /**
         * setMemoryDataAssigned
         * Set the memory data is assigned, or written.
         * The operator name should be provided since the outputs of the prev operators may be accessed.
         * @param op operator name
         * @param tensor tensor name
         */
        void setMemoryDataAssigned(const std::string& tensor) {
            if (!waiting) throw uninited_exception();
            if (executing) throw inited_exception();
            if (!isTensorWaited(tensor)) throw status_exception("Tensor not waited.");
            
            // Do not acquire locks here since the tensor is awaited.
            // Tensor exists since isTensorWaited(tensor) is true.
            status::TensorPres& pres = requested_tensors.at(tensor);
            if (pres.isHostLocated()) session.op_executor.freeHost(pres, pres.getHostSize());
            assert(!pres.isHostLocated());
            pres.setAssigned();

            // emit memory event
            session.backend_handle.lock()->submitEvent(events::MemoryEvent(op, tensor, pres.getSize() ,events::MemoryEventType::write, stage));
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
            if (executing) throw inited_exception();
            if (!isTensorWaited(tensor)) throw status_exception("Operator or tensor not waited.");

            status::TensorPres& pres = requested_tensors.at(tensor);
            pres.setAcquired();

            // emit memory event
            session.backend_handle.lock()->submitEvent(events::MemoryEvent(op, tensor, pres.getSize(), events::MemoryEventType::read, stage));
        }

        /**
         * setMemoryDataAccessed
         * Set the memory data is accessed.
         * The operator name should be provided since the outputs of the prev operators may be accessed.
         * @param tensor tensor name
         */
        void setMemoryDataAccessed(const std::string& tensor) {
            if (!waiting) throw uninited_exception();
            if (executing) throw inited_exception();
            if (!isTensorWaited(tensor)) throw status_exception("Tensor not waited.");
            
            status::TensorPres& pres = requested_tensors.at(tensor);
            if (pres.isHostLocated()) session.op_executor.freeHost(pres, pres.getHostSize());
            assert(!pres.isHostLocated());
            pres.setAccessed();

            // emit memory event
            session.backend_handle.lock()->submitEvent(events::MemoryEvent(op, tensor, pres.getSize(), events::MemoryEventType::access, stage));
        }

        void setOperationFinished() {
            if (!waiting) throw uninited_exception();
            if (!executing) throw uninited_exception();
            session.sch_executor.setOperatorFinished(op);
            session.backend_handle.lock()->submitEvent(events::ExecutionEvent(op, events::ExecutionEventType::release, stage));

            executing = false;
        }

        void release() {
            if (executing) setOperationFinished();
            if (!waiting) return;

            for (auto &x : requested_tensors)
                x.second.release();

            waiting = false;
        }

        ~Request() {
            if (waiting) release();
        }

    };  // inner struct Request

private:
    Context context;

    std::weak_ptr<BackendHandle> backend_handle;

    status::MemoryStatus& status;
    layout::MemoryLayout& layout;

    MemoryScheduleExecutor& sch_executor;
    MemoryOperationExecutor op_executor;

    MemoryInfo memory_info;
    
    layout::MemoryDefragmentationExecutor defrag_executor;

    Callbacks callbacks;

    Logger* logger;

    ApplicationStage stage = ApplicationStage::forward;

    void setBackendHandle(const std::weak_ptr<BackendHandle>& _backend_handle) {
        backend_handle = _backend_handle;
    }
    void setMemoryManager(MemoryManager* _memory_manager) {
        op_executor.setMemoryManager(_memory_manager);
        defrag_executor.setMemoryManager(_memory_manager);

        memory_info = _memory_manager->getMemoryInfo();
    }
    void setLogger(Logger* _logger) {
        logger = _logger;
    }
    void setCallback(CallbackStage stage, const std::function<int(const std::string&, void*)>& callback) {
        callbacks.emplace(stage, callback);
    }

    size_t waitTensorMemory(size_t size, const std::string& initial_tensor) {
        std::string tensor_name = initial_tensor;
        while (true) {
            // Since the memory schedule executor is synchronized, a tensor that cannot be referenced must be waited by memory session.
            status::TensorView tensor_view = status.tryReferenceTensor(tensor_name);
            if (!tensor_view.isReferenced()) return 0;
            status::TensorPres tensor_pres = tensor_view.reference();
            // Do not swap out tensors that already host-only.
            if (!tensor_pres.isDeviceLocated()) return 0;
            uint8_t *device_address_e = (uint8_t *)(tensor_pres.getLastSection().device_address);
            // // Do not swap out persistent or transient tensors.
            if (!layout.isCommon(device_address_e)) return 0;

            // Prepare to swap out this tensor.
            // Step 1: Locate the first section on device.
            const status::MemorySection *section = &(tensor_pres.getFirstSection());
            while (section != nullptr) {
                if (section->status == status::MemoryStatusType::empty || section->status == status::MemoryStatusType::device || section->status == status::MemoryStatusType::coexist) break;
                section = section->next();
            }
            // Since data located on device, a section will be selected finally.
            assert(section != nullptr);

            // Step 2: Locate prev memory region.
            size_t avail_size = 0;
            if (layout.isRegionExist(section->device_address, Direction::prev)) {
                layout::MemoryRegion memory_region_prev = layout.getMemoryRegion(section->device_address, Direction::prev);
                if (!memory_region_prev.allocated) avail_size = memory_region_prev.size;
            } else avail_size = 0;
            if (avail_size >= size) return avail_size;

            // Step 3: Calculate swapping amount
            device_address_e += tensor_pres.getLastSection().size;
            if (tensor_pres.getFragment().status == status::MemoryStatusType::empty) device_address_e += tensor_pres.getFragment().size;
            size_t releasing_b = tensor_pres.getDeviceSize();
            if (tensor_pres.getFragment().status == status::MemoryStatusType::empty) releasing_b += tensor_pres.getFragment().size;
            size_t releasing_size = releasing_b;
            if (releasing_size + avail_size > size) releasing_size = size - avail_size;
            // If partically swap tensor, reserve aligned size
            assert(utils::get_memory_aligned_size(releasing_size, memory_info.device.align_size) >= releasing_size);
            size_t releasing_alignment_size = utils::get_memory_aligned_size(releasing_size, memory_info.device.align_size) - releasing_size;
            if (releasing_size + releasing_alignment_size <= tensor_pres.getDeviceSize()) releasing_size += releasing_alignment_size;
            else releasing_alignment_size = 0;
            op_executor.swapOut(tensor_pres, releasing_size);
            size_t releasing_e = tensor_pres.getDeviceSize();
            if (tensor_pres.getFragment().status == status::MemoryStatusType::empty) releasing_e += tensor_pres.getFragment().size;

            std::string op_name = tensor_pres.getOperatorName();
            if (callbacks.count(CallbackStage::postSwapOut)) callbacks.at(CallbackStage::postSwapOut)(tensor_name, tensor_pres.getSection(0).host_address);
            (*logger) << LogLevel::debug << "Operator " << op_name << ": tensor " << tensor_name << " swapped out. (Memory insufficience)" << endl;

            backend_handle.lock()->submitEvent(events::MemoryEvent(op_name, tensor_name, releasing_b - releasing_e, events::MemoryEventType::swapout, stage));

            assert(releasing_b >= (releasing_e + releasing_alignment_size));
            avail_size = avail_size + releasing_b - releasing_e - releasing_alignment_size;
            if (avail_size >= size) return avail_size;

            assert(releasing_e == 0);

            if (!layout.isRegionExist(device_address_e)) return avail_size;
            layout::MemoryRegion region = layout.getMemoryRegion(device_address_e);
            if (!region.allocated) {
                avail_size += region.size;
                if (avail_size >= size) return avail_size;
                // Memory demand unmet.
                device_address_e += region.size;
                if (!layout.isRegionExist(device_address_e)) return avail_size;
                region = layout.getMemoryRegion(device_address_e);
            }
            tensor_name = region.name;
        }
        return 0;
    }

public:
    MemorySession(const Context& _context, MemoryScheduleExecutor& _executor, status::MemoryStatus& _status, layout::MemoryLayout& _layout): context(_context), status(_status), layout(_layout), sch_executor(_executor),  op_executor(_layout), defrag_executor(_status, _layout) {}

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

        (*logger) << LogLevel::info << "Iteration: " << sch_executor.getIteration() << endl;
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
        (*logger) << LogLevel::debug << "Half iteration: " << sch_executor.getIteration() << endl;
        // backend_handle.lock()->
    }

    inline ApplicationStage getStage() { return stage; }

    /**
     * @brief Set the memory data has dynamic shape and size changed.
     * @param op operator name
     * @param tensor tensor name
     * @param size dynamic shape size
     */
    void setMemoryDataReshaped(const std::string& op, const std::string& tensor, size_t size) {
        if (!status.isTensorRegistered(tensor)) throw status_exception("Tensor not registered.");

        status::TensorPres pres = status.referenceTensor(tensor);
        pres.setReshaped(size);

        // emit memory event
        backend_handle.lock()->submitEvent(events::MemoryEvent(op, tensor, pres.getSize(), events::MemoryEventType::reshape, stage));
    }

    /**
     * @brief Set the memory data is allocated.
     * @param op operator name
     * @param tensor tensor name
     * @param address tensor address
     */
    void setMemoryDataAllocated(const std::string& op, const std::string& tensor, void* address) {
        if (!status.isTensorRegistered(tensor)) throw status_exception("Tensor not registered.");

        status::TensorPres pres = status.referenceTensor(tensor);
        pres.setAllocated(address);
        if (pres.hasFragment()) op_executor.fragment(pres);

        layout.recordMemoryAllocateEvent(address, pres.getSize(), tensor);
        // if (layout.isTransient(address)) defrag_executor.recordMemoryAllocateEvent(address);

        // emit memory event
        backend_handle.lock()->submitEvent(events::MemoryEvent(op, tensor, pres.getSize(), events::MemoryEventType::allocate, stage));
    }

    void setMemoryDataAllocated(const std::string& tensor, void* address) { setMemoryDataAllocated("", tensor, address); }

    /**
     * @brief  Assure the data of a operator is moved to the device memory before the operator is launched.
     * @return MemoryRequest object
     */
    Request createRequest(const std::string& op = "") {
        // if (!status.isOperatorRegistered(op)) throw status_exception("Operator not registered.");
        Request re(*this, op, stage);
        return re;
    }

    /**
     * @brief Wait for available memory. Memory insufficent is an emergency event, hence an independent method is provided.
     * @param size Memory size that should be released.
     * @return Memory size released.
     * @note Currently this method adopts a FIFO strategy that the firstly forward-propagating operator will be firstly released. 
     */
    size_t waitMemory(size_t size, const MemoryFunction& func = []() { return false; }) {
        utils::Presentation<MemoryScheduleExecutor> presentation(sch_executor);
        presentation.require();
        if (func()) return size;

        size_t avail_size = 0;
        for (auto &op_name : status.getExecutionOrder()) {
            status::OperatorView op_view = status.tryReferenceOperator(op_name);
            if (!op_view.isReferenced()) continue;
            status::OperatorPres op_pres = op_view.reference();
            // Forward propagation and backward propagation share the same set of tensors.
            // if (op_pres.isBackwardPropagation()) continue;

            for (auto &s : op_pres.getTensors()) { 
                // Try to release memory from tensors.
                avail_size = waitTensorMemory(size, s);
                if (avail_size >= size) break;
            }
            if (avail_size >= size) break;
        }

        // Memory demand unmet, try defragmentation.
        // auto pair = defrag_executor.getTransientBlockAllocatableSize(size);
        // if (pair.second >= size) defrag_executor.performDefragmentation(size);

        if (avail_size >= size) (*logger) << LogLevel::info << "Memory insufficient, mori releases " << avail_size << " of memory." << endl;
        else (*logger) << LogLevel::info << "Mori memory releasing failed, " << size - avail_size << " unmet." << endl;

        func();
        presentation.release();
        return avail_size;
    }

    /**
     * @brief Set the memory data is freed.
     * @param op operator name
     * @param tensor tensor name
     */
    void setMemoryDataFreed(const std::string& op, const std::string& tensor) {
        if (!status.isTensorRegistered(tensor)) throw status_exception("Operator or tensor not registered.");
        
        status::TensorPres pres = status.referenceTensor(tensor);
        op_executor.freeHost(pres, pres.getHostSize());
        const status::MemorySection* section = (&pres.getFirstSection());
        do {
            switch (section->status) {
                case status::MemoryStatusType::empty:
                case status::MemoryStatusType::device: {
                    void* device_address = section->device_address;
                    pres.setDeviceFreed(section->offset);
                    // if (layout.isTransient(device_address)) defrag_executor.recordMemoryFreeEvent(device_address);
                    layout.recordMemoryFreeEvent(device_address);
                }
                case status::MemoryStatusType::none:
                    break;
                default:
                    // coexist, host
                    assert(0);
                    break;
            }

            if (pres.isMergeable(section->offset)) pres.merge(section->offset);
            const status::MemorySection* section_prev = section->prev();
            if (section_prev != nullptr && pres.isMergeable(section_prev->offset)) section = &(pres.merge(section->offset));

            section = section->next();
        } while (section != nullptr);

        assert(pres.getSectionCount() == 1);
        assert(!pres.isDeviceLocated());

        // emit memory event
        backend_handle.lock()->submitEvent(events::MemoryEvent(op, tensor, pres.getSize(), events::MemoryEventType::free, stage));
    }

    void setMemoryDataFreed(const std::string& tensor) { setMemoryDataFreed("", tensor); }

    ~MemorySession() = default;
};  // struct MemorySession

using MemoryRequest = MemorySession::Request;

}   // namespace mori