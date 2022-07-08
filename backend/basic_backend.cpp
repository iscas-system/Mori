#include <functional>
#include <memory>
#include <chrono>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <unordered_set>
#include <unordered_map>

#include <dlfcn.h>

#include "memory_scheduler.hpp"
#include "events.hpp"
#include "memory_scheduler.hpp"
#include "../includes/memory_status.hpp"
#include "../includes/backend.hpp"
#include "../includes/context.hpp"
#include "../includes/memory_event.hpp"

namespace mori {

extern "C" __attribute__((visibility("default"))) int backend_entry(std::unique_ptr<Backend>& ptr, const Context& _context);

/**
 * BasicBackend
 * The backend of Mori.
 */
struct BasicBackend final : public Backend {
    // Backend information
    Context context;

    MemoryStatuses memory_status;
    Events events;

    std::unique_ptr<MemoryScheduler> scheduler;
    void* hInst = nullptr;
    
    std::atomic<bool> inited = false;

    // Scheduling information
    std::thread scheduler_thread;
    std::mutex scheduler_mutex;
    int sleep_interval = 5;     // millisecond

    // DL training information
    std::atomic<int> iteration = 0;

    BasicBackend(Context _context) {
        context = _context;

        std::string scheduler_name = context["scheduler"];
        if (scheduler_name == "fifo") scheduler = std::unique_ptr<MemoryScheduler>(new FIFOMemoryScheduler());
        else if (scheduler_name == "dependency") scheduler = std::unique_ptr<MemoryScheduler>(new DependencyAwareMemoryScheduler());
        else if (scheduler_name == "maxsize") scheduler = std::unique_ptr<MemoryScheduler>(new MaximumSizePriorityMemoryScheduler());
        else if (scheduler_name == "rwaware") scheduler = std::unique_ptr<MemoryScheduler>(new RWAwareMemoryScheduler());
        else {
            std::string scheduler_path = context["scheduler_path"];

            typedef int(*SchedulerEntryType)(std::unique_ptr<MemoryScheduler>&);

            hInst = dlopen(scheduler_path.c_str(), RTLD_LAZY);
            if (!hInst) throw std::exception();
            SchedulerEntryType scheduler_entry = (SchedulerEntryType)dlsym(hInst, "scheduler_entry");

            int ret;
            if (scheduler_entry) ret = scheduler_entry(scheduler);
            else throw std::exception();

            if (ret != 0) throw std::exception();
        }
       
    }

    virtual void init() {
        // Step 1: Check if the backend is ready to init

        // Step 2: Init scheduler
        scheduler->init();

        inited = true;

        // Step 3: Init scheduler thread for active scheduler
        if (scheduler->isActiveScheduler()) {
            scheduler_thread = std::thread([this]() {
                std::unique_lock<std::mutex>{scheduler_mutex};
                while (inited) {
                    scheduler->schedule();
                    std::this_thread::sleep_for(std::chrono::milliseconds{sleep_interval});
                }
            });
            // Examine if the thread starts properly
            while (!scheduler_thread.joinable());
        }
    }

    virtual void registerOperator(const OperatorStatus& operator_status) {
        if (!inited) throw std::exception();

        if (memory_status.isOperatorRegistered(operator_status.name)) throw std::exception();

        memory_status.registerOperator(operator_status);
    }

    virtual void submitEvent(const MemoryEvent& event) {
        if (!inited) throw std::exception();

        events.submitEvent(event);
        scheduler->submitEvent(event);
    }

    virtual std::vector<ScheduleEvent> getScheduleEvents() {
        return scheduler->getScheduleEvents();
    }

    /**
     * getIteration
     * The current iteration
     * @return current iteration
     */
    virtual int getIteration() {
        return iteration;
    }

    /**
     * increaseIteration
     * Increase the iteration counting of DL training
     * Since Mori should schedule the swapping in current iteration, this method may block the training until the scheduler is synchorized with backend and DL training.
     * @return iteration
     */
    virtual int increaseIteration() {
        ++iteration;
        // Block to synchorize with scheduler.
        scheduler->increaseIteration();
        return iteration;
    }

    virtual void unregisterOperator(const std::string& op) {
        if (!inited) throw std::exception();

        if (!memory_status.isOperatorRegistered(op)) throw std::exception();

        memory_status.unregisterOperator(op);
    }

    virtual void terminate() {
        if (!inited) return;

        // Terminate scheduler thread
        inited = false;

        // Examine if the thread terminates properly
        if (scheduler_thread.joinable()) scheduler_thread.join();

        scheduler->terminate();
    }

    virtual ~BasicBackend() {
        if (hInst) dlclose(hInst);
    }
};  // struct BasicBackend

int backend_entry(std::unique_ptr<Backend>& ptr, const Context& _context) {
    // Backend should be explictly destroyed before the dylib released.
    // Therefore, a heap object and a shared pointer.
    ptr.reset(new mori::BasicBackend(_context));
    return 0;
} 

}   // namespace mori