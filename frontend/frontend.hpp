#pragma once

#include "frontend/memory_session.hpp"
#include "frontend/memory_schedule_executor.hpp"
#include "frontend/memory_manager.hpp"
#include "frontend/backend_handle.hpp"
#include "frontend/callbacks.hpp"
#include "includes/context.hpp"
#include "includes/memory_status.hpp"
#include "includes/logging.hpp"
#include "includes/exceptions.hpp"

namespace mori {

/**
 * Frontend of Mori, provided to the DL system.
 * In this version only single-thread graph execution is considered.
 */
struct Frontend {
protected:
    struct Impl {
        Frontend& frontend;
        Impl(Frontend& _frontend): frontend(_frontend) {}

        virtual void setMemoryManager(MemoryManager* _mem_manager) = 0;
        virtual void setLogger(Logger* _logger) = 0;
    
        virtual void init() = 0;
        virtual bool isInited() const noexcept = 0;

        virtual void registerTensor(const status::Tensor& tensor) = 0;
        virtual void registerOperator(const status::Operator& operator_status) = 0;
        // virtual void updateOperator(const std::string& op, const status::Tensor& tensor_status) = 0;

        virtual void setEntry(const std::string& _op) = 0;

        virtual void setCallback(CallbackStage stage, const std::function<int(const std::string& tensor, void* ptr)>& callback) = 0;

        virtual void start() = 0;
        virtual bool isStarted() const noexcept = 0;

        virtual MemorySession& getSession() = 0;

        virtual int getIteration() const = 0;
        virtual void setIteration(int iteration) = 0;
        virtual void increaseIteration() = 0;

        virtual void updateSchedule() = 0;

        virtual void unregisterTensor(const std::string& tensor) = 0;
        virtual void unregisterOperator(const std::string& op) = 0;

        virtual void stop() = 0;

        virtual void terminate() = 0;
    };  // struct Impl

    struct UninitedImpl final : public Impl {
        UninitedImpl(Frontend& _frontend): Impl(_frontend) {}

        virtual void setMemoryManager(MemoryManager* _mem_manager) override {
            frontend.mem_manager = _mem_manager;
            frontend.executor.setMemoryManager(_mem_manager);
            frontend.session.setMemoryManager(_mem_manager);
        }
        virtual void setLogger(Logger* _logger) override {
            if (_logger == nullptr) frontend.logger = &frontend.empty_logger;
            frontend.logger = _logger;

            frontend.backend_handle->setLogger(frontend.logger);
            frontend.session.setLogger(frontend.logger);
            frontend.executor.setLogger(frontend.logger);
        }
    
        virtual void init() override {
            // Forward callpath initialization failed.
            if (frontend.backend_handle == nullptr) throw status_exception("Backend not inited.");
            // Backward callpath initialization failed.
            if (frontend.mem_manager == nullptr) throw status_exception("Memory manager not assigned.");
            
            frontend.backend_handle->init();

            frontend.impl = &frontend.inited_impl;
            frontend.logger->submit(LogLevel::info, "Mori frontend inited.");
        }
        virtual bool isInited() const noexcept override { return false; }

        virtual void registerTensor(const status::Tensor& tensor) override {
            (*frontend.logger)<<LogLevel::error<<"Registering tensor "<<tensor.getName()<<" while frontend not initialized.";
            frontend.logger->flush();
            throw uninited_exception();
        }
        virtual void registerOperator(const status::Operator& operator_status) override {
            (*frontend.logger)<<LogLevel::error<<"Registering operator "<<operator_status.getName()<<" while frontend not initialized.";
            frontend.logger->flush();
            throw uninited_exception();
        }
        // virtual void updateOperator(const std::string& op, const status::Tensor& tensor_status) override {
        //     (*logger)<<LogLevel::error<<"Updating operator "<<op<<" while frontend not initialized.";
        //     logger->flush();
        //     throw uninited_exception();
        // }
        virtual void setEntry(const std::string& _op) override {
            (*frontend.logger)<<LogLevel::error<<"Setting entry operator "<<_op<<" while frontend not initialized.";
            frontend.logger->flush();
            throw uninited_exception();
        }

        virtual void setCallback(CallbackStage stage, const std::function<int(const std::string& tensor, void* ptr)>& callback) override {
            (*frontend.logger)<<LogLevel::error<<"Setting callbacks while frontend not initialized.";
            frontend.logger->flush();
            throw uninited_exception();
        }

        virtual void start() override {
            (*frontend.logger)<<LogLevel::error<<"Starting uninitialized frontend.";
            frontend.logger->flush();
            throw uninited_exception();
        }
        virtual bool isStarted() const noexcept override { return false; }

        virtual MemorySession& getSession() override {
            (*frontend.logger)<<LogLevel::error<<"Referencing to session from uninitialized frontend.";
            frontend.logger->flush();
            throw uninited_exception();
        }

        virtual int getIteration() const override {
            (*frontend.logger)<<LogLevel::error<<"Frontend uninitialized.";
            frontend.logger->flush();
            throw uninited_exception();
        }
        virtual void setIteration(int iteration) override {        
            (*frontend.logger)<<LogLevel::error<<"Frontend uninitialized.";
            frontend.logger->flush();
            throw uninited_exception();
        }
        virtual void increaseIteration() override {        
            (*frontend.logger)<<LogLevel::error<<"Frontend uninitialized.";
            frontend.logger->flush();
            throw uninited_exception();
        }

        virtual void updateSchedule() override {
            (*frontend.logger)<<LogLevel::error<<"Updating schedule while frontend not initialized.";
            frontend.logger->flush();
            throw uninited_exception();
        }

        virtual void unregisterTensor(const std::string& tensor) override {
            (*frontend.logger)<<LogLevel::error<<"Unregistering tensor "<<tensor<<" while frontend not initialized.";
            frontend.logger->flush();
            throw uninited_exception();
        }
        virtual void unregisterOperator(const std::string& op) override {
            (*frontend.logger)<<LogLevel::error<<"Unregistering operator "<<op<<" while frontend not initialized.";
            frontend.logger->flush();
            throw uninited_exception();
        }

        virtual void stop() override {
            (*frontend.logger)<<LogLevel::error<<"Stopping uninitialized frontend.";
            frontend.logger->flush();
            throw uninited_exception();
        }

        virtual void terminate() override {
            (*frontend.logger)<<LogLevel::error<<"Terminating uninitialized frontend.";
            frontend.logger->flush();
            throw uninited_exception();
        }
    };  // struct UninitedImpl

    struct InitedImpl final : public Impl {
        InitedImpl(Frontend& _frontend): Impl(_frontend) {}

        virtual void setMemoryManager(MemoryManager* _mem_manager) override {
            (*frontend.logger)<<LogLevel::error<<"Setting memory manager for initialized frontend.";
            frontend.logger->flush();
            throw inited_exception();
        }

        virtual void setLogger(Logger* _logger) override {
            (*frontend.logger)<<LogLevel::error<<"Setting logger for initialized frontend.";
            frontend.logger->flush();
            throw inited_exception();
        }

        virtual void init() override {
            (*frontend.logger)<<LogLevel::error<<"Initializing frontend that already inited.";
            frontend.logger->flush();
            throw inited_exception();
        }
        virtual bool isInited() const noexcept override { return true; }

        virtual void registerTensor(const status::Tensor& tensor) override {
            frontend.memory_status.registerTensor(tensor);
            frontend.backend_handle->registerTensor(tensor);

            (*frontend.logger)<<LogLevel::debug<<"Tensor "<<tensor.getName()<<" registered.";
            frontend.logger->flush();
        }

        virtual void registerOperator(const status::Operator& operator_status) override {
            frontend.memory_status.registerOperator(operator_status);
            frontend.backend_handle->registerOperator(operator_status);

            (*frontend.logger)<<LogLevel::debug<<"Operator "<<operator_status.getName()<<" registered.";
            frontend.logger->flush();
        }

        // virtual void updateOperator(const std::string& op, const status::Tensor& tensor_status) override {
        //     memory_status.updateOperator(op, tensor_status);
        //     // backend_handle->updateOperator(op, tensor_status);

        //     (*logger)<<LogLevel::debug<<"Operator "<<op<<" updated.";
        //     logger->flush();
        // }

        virtual void setEntry(const std::string& _op) override {
            frontend.memory_status.setEntry(_op);
            frontend.backend_handle->setEntry(_op);
        }

        virtual void setCallback(CallbackStage stage, const std::function<int(const std::string& tensor, void* ptr)>& callback) override {
            frontend.executor.setCallback(stage, callback);
            frontend.session.setCallback(stage, callback);
        }

        virtual void start() override {
            frontend.executor.init();
            frontend.backend_handle->start();
            frontend.impl = &frontend.started_impl;
            (*frontend.logger)<<LogLevel::debug<<"Mori started.";
            frontend.logger->flush();
        }

        virtual bool isStarted() const noexcept override { return false; }

        virtual MemorySession& getSession() override {
            (*frontend.logger)<<LogLevel::error<<"Referencing to session from not-started frontend.";
            frontend.logger->flush();
            throw uninited_exception();
        }

        virtual int getIteration() const override {
            (*frontend.logger)<<LogLevel::error<<"Frontend not started.";
            frontend.logger->flush();
            throw uninited_exception();
        }
        virtual void setIteration(int iteration) override {        
            (*frontend.logger)<<LogLevel::error<<"Frontend not started.";
            frontend.logger->flush();
            throw uninited_exception();
        }
        virtual void increaseIteration() override {        
            (*frontend.logger)<<LogLevel::error<<"Frontend not started.";
            frontend.logger->flush();
            throw uninited_exception();
        }

        virtual void updateSchedule() override {
            (*frontend.logger)<<LogLevel::error<<"Updating schedule for not-started frontend.";
            frontend.logger->flush();
            throw uninited_exception();
        }

        virtual void unregisterTensor(const std::string& tensor) override {
            frontend.memory_status.unregisterTensor(tensor);
            frontend.backend_handle->unregisterTensor(tensor);

            (*frontend.logger)<<LogLevel::debug<<"Tensor "<<tensor<<" unregistered.";
            frontend.logger->flush();
        }

        virtual void unregisterOperator(const std::string& op) override {
            frontend.memory_status.unregisterOperator(op);
            frontend.backend_handle->unregisterOperator(op);

            (*frontend.logger)<<LogLevel::debug<<"Operator "<<op<<" unregistered.";
            frontend.logger->flush();
        }

        virtual void stop() override {
            (*frontend.logger)<<LogLevel::error<<"Stopping non-started frontend.";
            frontend.logger->flush();
            throw uninited_exception();
        }

        virtual void terminate() override {
            frontend.backend_handle -> terminate();
            frontend.memory_status.clear();

            frontend.impl = &frontend.uninited_impl;

            frontend.logger->submit(LogLevel::info, "Mori frontend terminated.");
        }
    };  // struct InitedImpl

    struct StartedImpl final : public Impl {
        StartedImpl(Frontend& _frontend): Impl(_frontend) {}

        virtual void setMemoryManager(MemoryManager* _mem_manager) override {
            (*frontend.logger)<<LogLevel::error<<"Setting memory manager for started frontend.";
            frontend.logger->flush();
            throw inited_exception();
        }
        virtual void setLogger(Logger* _logger) override {
            (*frontend.logger)<<LogLevel::error<<"Setting logger for started frontend.";
            frontend.logger->flush();
            throw inited_exception();
        }

        virtual void init() override {
            (*frontend.logger)<<LogLevel::error<<"Initializing frontend that already started.";
            frontend.logger->flush();
            throw inited_exception();
        }
        virtual bool isInited() const noexcept override {return true;}

        virtual void registerTensor(const status::Tensor& tensor) override {
            (*frontend.logger)<<LogLevel::error<<"Registering tensor "<<tensor.getName()<<" while frontend started.";
            frontend.logger->flush();
            throw inited_exception();
        }
        virtual void registerOperator(const status::Operator& operator_status) override {
            (*frontend.logger)<<LogLevel::error<<"Registering operator "<<operator_status.getName()<<" while frontend started.";
            frontend.logger->flush();
            throw inited_exception();
        }

        // virtual void updateOperator(const std::string& op, const status::Tensor& tensor_status) {
        //     if (!inited) {
        //         (*logger)<<LogLevel::error<<"Updating operator "<<op<<" while frontend not initialized.";
        //         logger->flush();
        //         throw uninited_exception();
        //     }
        // }

        virtual void setEntry(const std::string& _op) override {
            (*frontend.logger)<<LogLevel::error<<"Setting entry operator "<<_op<<" while frontend started.";
            frontend.logger->flush();
            throw inited_exception();
        }

        virtual void setCallback(CallbackStage stage, const std::function<int(const std::string& tensor, void* ptr)>& callback) override {
            (*frontend.logger)<<LogLevel::error<<"Setting callbacks while frontend started.";
            frontend.logger->flush();
            throw inited_exception();
        }

        virtual void start() override {
            (*frontend.logger)<<LogLevel::error<<"Frontend already started.";
            frontend.logger->flush();
            throw inited_exception();
        }

        virtual bool isStarted() const noexcept override { return true; }

        virtual MemorySession& getSession() override { return frontend.session; }

        virtual int getIteration() const override { return frontend.executor.getIteration(); }
        virtual void setIteration(int iteration) override { return frontend.executor.setIteration(iteration); }
        virtual void increaseIteration() override { return frontend.executor.increaseIteration(); }
        
        virtual void updateSchedule() override {
            auto&& event_set = frontend.backend_handle->getScheduleEvents();
            frontend.executor.updateSchedule(event_set);

            (*frontend.logger)<<LogLevel::debug<<"Schedule updated.";
            frontend.logger->flush();
        }

        virtual void unregisterTensor(const std::string& tensor) override {
            (*frontend.logger)<<LogLevel::error<<"Unregistering tensor "<<tensor<<" while frontend not initialized.";
            frontend.logger->flush();
            throw uninited_exception();
        }

        virtual void unregisterOperator(const std::string& op) override {
            (*frontend.logger)<<LogLevel::error<<"Unregistering operator "<<op<<" while frontend not initialized.";
            frontend.logger->flush();
            throw uninited_exception();
        }

        virtual void stop() override {
            frontend.executor.terminate();
            frontend.backend_handle->stop();
            frontend.impl = &frontend.inited_impl;
        }

        virtual void terminate() override {
            stop();
            frontend.impl = &frontend.inited_impl;
            // Terminate should switch the frontend to uninited state.
            frontend.impl->terminate();
        }
    };  // struct InitedStartedImpl

protected:
    UninitedImpl uninited_impl;
    InitedImpl   inited_impl;
    StartedImpl  started_impl;

    Impl* impl = &uninited_impl;

protected:
    Context context;

    std::shared_ptr<BackendHandle> backend_handle = nullptr;

    MemoryManager* mem_manager = nullptr;
    status::MemoryStatus memory_status;

    // Each frontend holds one memory session.
    MemorySession session;
    MemoryScheduleExecutor executor;
    
    Logger empty_logger;
    Logger* logger = nullptr;

public:
    Frontend(const Context& _context): 
        uninited_impl(*this),
        inited_impl(*this),
        started_impl(*this),
        context(_context), 
        session(_context, memory_status),
        executor(_context, memory_status) {
        // Set backend
        backend_handle = make_backend_handle(_context);

        session.setBackendHandle(backend_handle);
        executor.setBackendHandle(backend_handle);

        logger = &empty_logger;
    }

    /**
     * @brief Set memory manager for memory swapping.
     * @param _mem_manager Pointer to memory manager used in DL framework.
     */
    inline void setMemoryManager(MemoryManager* _mem_manager) { impl->setMemoryManager(_mem_manager); }
    /**
     * @brief Set logger.
     * @param _logger Pointer to logger used in DL framework.
     */
    inline void setLogger(Logger* _logger) {impl->setLogger(_logger);}

    /**
     * @brief Init mori frontend.
     * @note Mutiple call to this method would lead to mori::inited_exception.
     */
    inline void init() { impl->init(); }
    /**
     * @brief If mori frontend inited.
     * @return If mori frontend inited.
     */
    inline bool isInited() const { return impl->isInited(); }

    /**
     * @brief Register a tensor in DL procedure.
     * @param tensor Information of the tensor to be registered.
     */ 
    inline void registerTensor(const status::Tensor& tensor) { impl->registerTensor(tensor); }

    /**
     * @brief Register an operator in this graph.
     * @param operator_status Information of the operator to be registered.
     * @note The register order will be regarded as the execution order.
     */
    inline void registerOperator(const status::Operator& operator_status) { impl->registerOperator(operator_status); }

    /**
     * @brief Update operator.
     * @param operator_status Information of the operator to be updated.
     */
    // inline void updateOperator(const std::string& op, const status::Tensor& tensor_status) { impl->updateOperator(op, tensor_status); }

    /**
     * @brief Set entry operator of the DL computation graph.
     * @param op Entry operator name.
     */
    inline void setEntry(const std::string& op) { impl->setEntry(op); }

    /**
     * @brief Set callback functions for memory swapping. postSwapOut and postSwapIn supported.
     * @param stage Stage of the callback function.
     * @param callback Callback function.
     * @see enum struct mori::CallbackStage
     */
    inline void setCallback(CallbackStage stage, const std::function<int(const std::string& tensor, void* ptr)>& callback) { impl->setCallback(stage, callback); }

    /**
     * @brief Start mori frontend. Mori session and background executor will be started.
     * @note Mutiple call to this method would lead to mori::inited_exception.
     */
    inline void start() { impl->start(); }
    /**
     * @brief If mori frontend started.
     * @return If mori frontend started.
     */
    inline bool isStarted() const noexcept { return impl->isStarted(); }

    /**
     * @brief Reference to Mori memory swapping session.
     * @return Reference to Mori memory swapping session.
     */
    inline MemorySession& getSession() { return impl->getSession(); }

    /**
     * @brief Get current iteration count.
     * @return Current iteration count.
     */
    inline int getIteration() { return impl->getIteration(); }
    /**
     * @brief Set current iteration count.
     */
    inline void setIteration(int iteration) { impl->setIteration(iteration); }
    /**
     * @brief Increase iteration count.
     */
    inline void increaseIteration() { impl->increaseIteration(); }

    /**
     * @brief Update current memory swapping schedule
     */
    inline void updateSchedule() { impl->updateSchedule(); }

    /**
     * @brief Unregister a tensor in DL procedure.
     * @param tensor Tensor to be unregistered.
     */ 
    inline void unregisterTensor(const std::string& tensor) { impl->unregisterTensor(tensor); }

    /**
     * @brief Unregister an operator in this graph.
     * @param op Operator to be unregistered.
     */
    inline void unregisterOperator(const std::string& op) { impl->unregisterOperator(op); }

    /**
     * @brief Stop mori frontend. Mori session and background executor will be stopped.
     * @note Mutiple call to this method would lead to mori::uninited_exception.
     */
    inline void stop() { impl->stop(); }

    /**
     * @brief Terminate mori frontend.
     * @note Mutiple call to this method would lead to mori::uninited_exception.
     */
    inline void terminate() { impl->terminate(); }

    ~Frontend() {
        if (impl != &uninited_impl) impl->terminate();
        
        backend_handle.reset();
        mem_manager = nullptr;
        logger = nullptr;
    }

};  // struct Frontend

using MemorySwappingManager = Frontend;

}   // namespace mori