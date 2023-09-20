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

#include "backend/events.hpp"
#include "backend/dylibs_util.hpp"
#include "backend/schedulers/memory_scheduler.hpp"
#include "includes/memory_status.hpp"
#include "includes/backend.hpp"
#include "includes/context.hpp"
#include "includes/execution_event.hpp"
#include "includes/memory_event.hpp"
#include "includes/memory_info.hpp"
#include "includes/exceptions/status_exceptions.hpp"

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
    std::mutex events_m;

    std::unique_ptr<MemoryScheduler> scheduler;
    void* scheduler_hinst = nullptr;
    std::unique_ptr<exporter::ScheduleExporter> schedule_exporter;
    void* schedule_exporter_hinst = nullptr;
    
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
        Context::View scheduler_context = context.view("scheduler");
        if (scheduler_name == "section") scheduler = std::unique_ptr<MemoryScheduler>(new SectionAwareMemoryScheduler(scheduler_context, status, events));
        else if (scheduler_name == "dependency") scheduler = std::unique_ptr<MemoryScheduler>(new DependencyAwareMemoryScheduler(scheduler_context, status, events));
        else scheduler_hinst = utils::load_dylib("Scheduler", context.at("scheduler.path"), "scheduler_entry", scheduler, scheduler_context);

        // Set up events exporter
        std::string events_exporter_name = context.at("exporters.events");
        if (events_exporter_name == "empty") events_exporter = std::unique_ptr<exporter::EventsExporter>(new exporter::EventsExporter(context.view("exporters.events")));
        else events_exporter_hinst = utils::load_dylib("Events Exporter", context.at("exporters.events.path"), "events_exporter_entry", events_exporter, context.view("exporters.events"));

        // Set up tensors exporter
        std::string tensors_exporter_name = context.at("exporters.tensors");
        if (tensors_exporter_name == "empty") tensors_exporter = std::unique_ptr<exporter::TensorsExporter>(new exporter::TensorsExporter(context.view("exporters.tensors")));
        else tensors_exporter_hinst = utils::load_dylib("Tensors Exporter", context.at("exporters.tensors.path"), "tensors_exporter_entry", tensors_exporter, context.view("exporters.tensors"));

        // Set up schedule exporter
        std::string schedule_exporter_name = context.at("exporters.schedule");
        if (schedule_exporter_name == "empty") schedule_exporter = std::unique_ptr<exporter::ScheduleExporter>(new exporter::ScheduleExporter(context.view("exporters.schedule")));
        else schedule_exporter_hinst = utils::load_dylib("Schedule Exporter", context.at("exporters.schedule.path"), "schedule_exporter_entry", schedule_exporter, context.view("exporters.schedule"));
    }

    virtual void init() override {
        if (inited) throw inited_exception();
        if (started) throw inited_exception();

        inited = true;
    }

    virtual void submitMemoryStatus(const status::MemoryStatus& _status) override {
        status = _status;
        tensors_exporter->onTensors(status);
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

        std::unique_lock<std::mutex> l{events_m};
        events.submitEvent(event);
        events_exporter->onMemoryEvent(event);
        l.unlock();
        scheduler->submitEvent(event);
    }

    virtual void submitEvent(const events::ExecutionEvent& event) override {
        if (!inited) throw uninited_exception();
        if (!started) throw uninited_exception();

        std::unique_lock<std::mutex> l{events_m};
        events.submitEvent(event);
        events_exporter->onExecutionEvent(event);
        l.unlock();
        scheduler->submitEvent(event);
    }

    virtual events::ScheduleEvents getScheduleEvents() override {
        if (!inited) throw uninited_exception();
        if (!started) throw uninited_exception();

        events::ScheduleEvents&& re = scheduler->getScheduleEvents();
        for (auto &x : re.memory_map.getFragmentInfo()) {
            status::TensorPres pres = status.referenceTensor(x.first);
            pres.setFragment(x.second);
        }
        schedule_exporter->onScheduleEvents(re);
        return re;
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