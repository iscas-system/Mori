#pragma once

#include <functional>
#include <memory>
#include <chrono>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <unordered_set>
#include <unordered_map>

#include "backend/memory_scheduler.hpp"
#include "backend/events.hpp"
#include "backend/dylibs_util.hpp"
#include "includes/memory_status.hpp"
#include "includes/backend.hpp"
#include "includes/context.hpp"
#include "includes/memory_event.hpp"
#include "includes/exceptions.hpp"

namespace mori {

/**
 * BasicBackend
 * The backend of Mori.
 */
struct BasicBackend final : public Backend {
protected:
    // Backend information
    Context context;

    status::MemoryStatus status;
    std::unique_ptr<exporter::TensorsExporter> tensors_exporter;
    void* tensors_exporter_hinst = nullptr;

    events::Events events;
    std::unique_ptr<exporter::EventsExporter> events_exporter;
    void* events_exporter_hinst = nullptr;

    std::unique_ptr<MemoryScheduler> scheduler;
    void* scheduler_hinst = nullptr;
    
    std::atomic<bool> inited  = false;
    std::atomic<bool> started = false;

    // Scheduling information
    std::thread scheduler_thread;
    std::recursive_mutex scheduler_mutex;
    int sleep_interval = 5;     // millisecond

public:
    BasicBackend(Context _context) {
        context = _context;

        // Set up scheduler
        std::string scheduler_name = context.at("scheduler");
        if (scheduler_name == "fifo") scheduler = std::unique_ptr<MemoryScheduler>(new FIFOMemoryScheduler(context, status, events));
        else if (scheduler_name == "dependency") scheduler = std::unique_ptr<MemoryScheduler>(new DependencyAwareMemoryScheduler(context, status, events));
        else if (scheduler_name == "maxsize") scheduler = std::unique_ptr<MemoryScheduler>(new MaximumSizePriorityMemoryScheduler(context, status, events));
        else scheduler_hinst = utils::load_dylib("Scheduler", context.at("scheduler.path"), "scheduler_entry", scheduler);

        // Set up events exporter
        std::string events_exporter_name = context.at("exporters.events");
        if (events_exporter_name == "empty") events_exporter = std::unique_ptr<exporter::EventsExporter>(new exporter::EventsExporter(context));
        else events_exporter_hinst = utils::load_dylib("Events Exporter", context.at("exporters.events.path"), "events_exporter_entry", events_exporter, context.view());

        // Set up tensors exporter
        std::string tensors_exporter_name = context.at("exporters.tensors");
        if (tensors_exporter_name == "empty") tensors_exporter = std::unique_ptr<exporter::TensorsExporter>(new exporter::TensorsExporter(context));
        else tensors_exporter_hinst = utils::load_dylib("Tensors Exporter", context.at("exporters.tensors.path"), "tensors_exporter_entry", tensors_exporter, context.view());
    }

    virtual void init() override {
        if (inited) throw inited_exception();
        if (started) throw inited_exception();

        inited = true;
    }

    virtual void registerTensor(const status::Tensor& tensor) override {
        if (!inited) throw uninited_exception();
        if (started) throw inited_exception();
        status.registerTensor(tensor);
        tensors_exporter->onTensor(tensor);
    }
    virtual void registerOperator(const status::Operator& operator_status) override {
        if (!inited) throw uninited_exception();
        if (started) throw inited_exception();
        status.registerOperator(operator_status);
        tensors_exporter->onOperator(operator_status);
    }

    virtual void setEntry(const std::string& op) override {
        if (!inited) throw uninited_exception();
        if (started) throw inited_exception();
        status.setEntry(op);
        tensors_exporter->onEntry(op);
    }

    virtual void start() override {
        if (!inited) throw uninited_exception();
        if (started) throw inited_exception();

        started = true;

        // Init scheduler
        // scheduler->init();
        // if (scheduler->isActiveScheduler()) {
        //     scheduler_thread = std::thread([this]() {
        //         std::unique_lock<std::recursive_mutex>{scheduler_mutex};
        //         while (started) {
        //             scheduler->schedule();
        //             std::this_thread::sleep_for(std::chrono::milliseconds{sleep_interval});
        //         }
        //     });
        //     // Examine if the thread starts properly
        //     while (!scheduler_thread.joinable());
        // }
    }

    virtual void submitEvent(const events::MemoryEvent& event) override {
        if (!inited) throw uninited_exception();
        if (!started) throw uninited_exception();

        events.submitEvent(event);
        events_exporter->onEvent(event);
        scheduler->submitEvent(event);
    }

    virtual events::ScheduleEvents getScheduleEvents() override {
        if (!inited) throw uninited_exception();
        if (!started) throw uninited_exception();
        return scheduler->getScheduleEvents();
    }

    /**
     * getIteration
     * The current iteration
     * @return current iteration
     */
    virtual int getIteration() {
        if (!inited) throw uninited_exception();
        if (!started) throw uninited_exception();
        return events.getIteration();
    }

    virtual void setIteration(int _iteration) override { events.setIteration(_iteration); }

    /**
     * newIteration
     * Increase the iteration counting of application
     * Since Mori should schedule the swapping in current iteration, this method may block the training until the scheduler is synchorized with backend and application.
     * @return iteration
     */
    virtual void newIteration() override {
        if (!inited) throw uninited_exception();
        if (!started) throw uninited_exception();
        // Block to synchorize with scheduler.
        events.newIteration();
        scheduler->newIteration();
    }

    virtual void halfIteration() override {
        if (!inited) throw uninited_exception();
        if (!started) throw uninited_exception();
    }

    virtual void stop() override {
        if (!inited) throw uninited_exception();
        if (!started) throw uninited_exception();

        started = false;
        // Examine if the thread terminates properly
        if (scheduler_thread.joinable()) scheduler_thread.join();
    }

    virtual void unregisterTensor(const std::string& tensor) override {
        if (!inited) throw uninited_exception();
        if (started) throw inited_exception();
        status.unregisterTensor(tensor);
    }
    virtual void unregisterOperator(const std::string& op) override {
        if (!inited) throw uninited_exception();
        if (started) throw inited_exception();
        status.unregisterOperator(op);
    }

    virtual void terminate() override {
        if (!inited) throw uninited_exception();
        if (started) throw inited_exception();

        inited = false;
    }

    virtual ~BasicBackend() {
        scheduler.release();
        events_exporter.release();
        tensors_exporter.release();
        
        if (scheduler_hinst) dlclose(scheduler_hinst);
        if (events_exporter_hinst) dlclose(events_exporter_hinst);
        if (tensors_exporter_hinst) dlclose(tensors_exporter_hinst);
    }
};  // struct BasicBackend

}   // namespace mori