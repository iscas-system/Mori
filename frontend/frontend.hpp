#pragma once

#include "../includes/stdlibs.hpp"

#include "memory_session.hpp"
#include "memory_schedule_executor.hpp"
#include "memory_manager.hpp"
#include "backend_handle.hpp"
#include "../includes/context.hpp"
#include "../includes/memory_status.hpp"
#include "../includes/logging.hpp"

namespace mori {

/**
 * Frontend
 * Frontend of Mori, provided to the DL system.
 * In this version only single-thread graph execution is considered.
 */
struct Frontend {
protected:
    Context context;

    std::shared_ptr<BackendHandle> backend_handle = nullptr;

    MemoryManager* mem_manager = nullptr;
    MemoryStatuses memory_status;

    // Each frontend holds one memory session.
    MemorySession session;
    std::shared_ptr<MemoryScheduleExecutor> executor;
    
    Logger empty_logger;
    Logger* logger = nullptr;

    bool inited = false;

public:
    Frontend(const Context& _context): 
        context(_context), 
        session(_context) {
        // Set backend
        backend_handle = make_backend_handle(_context);

        // Set executor
        executor = make_executor(context);

        executor->setMemoryStatuses(&memory_status);

        session.setBackendHandle(std::weak_ptr<BackendHandle>(backend_handle));
        session.setMemoryStatusStorage(&memory_status);
        session.setExecutor(executor);

        logger = &empty_logger;
    }

    /**
     * setMemoryManager
     * Set up callback path for memory swapping
     * 
     */
    void setMemoryManager(MemoryManager* _mem_manager) {
        if (inited) throw std::exception();
        mem_manager = _mem_manager;
        executor->setMemoryManager(_mem_manager);
        session.setMemoryManager(_mem_manager);
    }

    void setLogger(Logger* _logger) {
        if (inited) throw std::exception();
        if (_logger == nullptr) logger = &empty_logger;
        logger = _logger;

        backend_handle->setLogger(logger);
        session.setLogger(logger);
        executor->setLogger(logger);
    }

    void init() {
        if (inited) throw std::exception();

        // Forward callpath initialization failed.
        if (backend_handle == nullptr) throw std::exception();
        // Backward callpath initialization failed.
        if (mem_manager == nullptr) throw std::exception();
        
        backend_handle->init();

        inited = true;

        logger->submit(LogLevel::debug, "Mori frontend inited.");
    }

    bool isInited() {return inited;}

    /**
     * registerOperator
     * Register an operator in this graph.
     * @param operator_status information of the operator to be registered.
     */
    void registerOperator(const OperatorStatus& operator_status) {
        if (!inited) {
            (*logger)<<LogLevel::error<<"Registering operator "<<operator_status.name<<" while frontend not initialized.";
            logger->flush();
            throw std::exception();
        }

        memory_status.registerOperator(operator_status);
        backend_handle->registerOperator(operator_status);

        (*logger)<<LogLevel::info<<"Operator "<<operator_status.name<<" registered.";
        logger->flush();
        
    }

    MemorySession& getSession() {
        if (!inited) {
            (*logger)<<LogLevel::error<<"Referencing to session from uninitialized frontend.";
            logger->flush();
            throw std::exception();
        }
        return session;
    }

    void updateSchedule() {
        if (!inited) throw std::exception();
        auto&& event_set = backend_handle->getScheduleEvents();
        executor->updateSchedule(event_set);
    }

    /**
     * unregisterOperator
     * Unregister an operator from Mori memory swapping management module.
     * @param op name of the operator to be unregistered.
     */
    void unregisterOperator(const std::string& op) {
        if (!inited) throw std::exception();

        memory_status.unregisterOperator(op);
        backend_handle->unregisterOperator(op);
    }

    void terminate() {
        if (!inited) throw std::exception();

        backend_handle -> terminate();
        
        backend_handle.reset();
        mem_manager = nullptr;
        logger = nullptr;

        inited = false;
    }

};

using MemorySwappingManager = Frontend;

static MemorySwappingManager make_swapping_mamager(const Context& _context) {
    return MemorySwappingManager(_context);
}

// static MemorySwappingManager make_swapping_mamager(const std::string& _context_path) {
//     return MemorySwappingManager(_context);
// }

}   // namespace mori