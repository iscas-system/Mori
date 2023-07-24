#pragma once

#include "includes/backend.hpp"
#include "includes/context.hpp"
#include "includes/logging.hpp"
#include "includes/memory_status.hpp"
#include "includes/memory_event.hpp"
#include "includes/execution_event.hpp"
#include "includes/exceptions/status_exceptions.hpp"

#ifndef ENABLE_EXTERNAL_BACKEND
#include "includes/basic_backend.hpp"
#else
extern "C" int backend_entry(std::unique_ptr<Backend>& ptr, const Context& _context);
#endif

namespace mori {

struct BackendHandle {
    bool inited = false;

    Logger* logger;

    BackendHandle() = default;
    BackendHandle(BackendHandle&& backend_handle) = default;

    void setLogger(Logger* _logger) {
        if (inited) throw inited_exception();
        logger = _logger;
    }

    virtual void init() = 0;

    virtual void submitMemoryStatus(const status::MemoryStatus& status) = 0;

    virtual void start() {}
    
    virtual void submitEvent(const events::MemoryEvent& event) = 0;
    virtual void submitEvent(const events::ExecutionEvent& event) = 0;
    virtual events::ScheduleEvents getScheduleEvents() = 0;

    virtual void setIteration(int _iteration) = 0;
    virtual void newIteration() = 0;
    virtual void halfIteration() = 0;

    virtual void stop() {}

    virtual void terminate() {
        if (!inited) throw uninited_exception();

        inited = false;
    }

    virtual ~BackendHandle() {
        logger = nullptr;
    }
};  // struct BackendHandle  

struct LocalBackendHandle : public BackendHandle {
    std::unique_ptr<Backend> backend;

    LocalBackendHandle() {}
    LocalBackendHandle(LocalBackendHandle&& backend_handle) {
        backend = std::move(backend_handle.backend);
    }

    virtual void init() override {
        if (inited) return;
        backend->init();
        inited = true;
    }

    virtual void submitMemoryStatus(const status::MemoryStatus& _status) override {
        backend->submitMemoryStatus(_status);
    }

    virtual void start() override { backend->start(); }

    virtual void submitEvent(const events::MemoryEvent& event) override {
        (*logger) << LogLevel::debug << "Submiting of event " << event << endl;
        backend->submitEvent(event);
    }
    virtual void submitEvent(const events::ExecutionEvent& event) override {
        (*logger) << LogLevel::debug << "Submiting of event " << event << endl;
        backend->submitEvent(event);
    }

    virtual void setIteration(int _iteration) override { backend->setIteration(_iteration); }
    virtual void newIteration() override { backend->newIteration(); }
    virtual void halfIteration() override { backend->halfIteration(); }

    virtual events::ScheduleEvents getScheduleEvents() override {
        return backend->getScheduleEvents();
    }

    virtual void stop() override { backend->stop(); }

    virtual void terminate() override {
        if (!inited) throw uninited_exception();

        backend->terminate();

        BackendHandle::terminate();
    }

    virtual ~LocalBackendHandle() {
        backend.reset();
    }
};  // struct LocalBackendHandle

#ifndef ENABLE_EXTERNAL_BACKEND
/**
 * Handle for integrated library backend.
 */
struct IntegratedBackendHandle : public LocalBackendHandle {
    IntegratedBackendHandle(const Context& _context) {
        backend.reset(new mori::BasicBackend(_context));
    }
};  // struct IntegratedBackendHandle
#else
/**
 * Handle for dynamic library backend.
 */
struct DylibBackendHandle : public LocalBackendHandle {
    void* hInst;

    DylibBackendHandle(const Context& _context): LocalBackendHandle(_context) {
        typedef int(*BackendEntryType)(std::unique_ptr<Backend>&, const Context&);

        const std::string& path = _context.at("path");
        std::string obj_path = std::string(path.begin() + 8, path.end()).c_str();

		hInst = dlopen(obj_path.c_str(), RTLD_LAZY);
		if (!hInst) throw dynamic_library_exception("Failed to open backend dynamic library.");
		BackendEntryType backend_entry = (BackendEntryType)dlsym(hInst, "backend_entry");

		int ret;
		if (backend_entry) ret = backend_entry(backend, _context);
		else throw dynamic_library_exception("Failed to access backend entry.");

		if (ret != 0) throw dynamic_library_exception("Failed to enter backend.");
    }

    virtual ~DylibBackendHandle() {
        backend.reset();
        dlclose(hInst);
    }
};  // struct DylibBackendHandle

#ifdef ENABLE_REMOTE_BACKEND

/**
 * RemoteBackendHandle
 * Handle for remote library backend.
 * Currently, HTTP Mori Server.
 */

// struct SharedMemoryBackendHandle : public BackendHandle {
//     SharedMemoryBackendHandle(const std::string& path) {}
//     virtual ~SharedMemoryBackendHandle() {}
// };  // struct SharedMemoryBackendHandle

// struct UnixSocketBackendHandle : public BackendHandle {
//     UnixSocketBackendHandle(const std::string& path) {}
//     ~UnixSocketBackendHandle() {}
// };  // struct UnixSocketBackendHandle

// struct HTTPBackendHandle : public BackendHandle {
//     HTTPBackendHandle(const std::string& path) {}
//     ~HTTPBackendHandle() {}
// };  // struct HTTPBackendHandle
#endif
#endif

static std::unique_ptr<BackendHandle> make_backend_handle(const Context& context) {
    // if (!context.isParamExists("path")) throw context_missing();

    const std::string& path = context.at("path");
#ifndef ENABLE_EXTERNAL_BACKEND
    if (path.find("int://") == 0) return std::unique_ptr<BackendHandle>(new IntegratedBackendHandle(context));
#else
    if (path.find("dylib://") == 0) return std::unique_ptr<BackendHandle>(new DylibBackendHandle(context));
#ifdef ENABLE_REMOTE_BACKEND
    if (path.find("http://") == path.begin()) return std::unique_ptr<BackendHandle>(new RemoteBackendHandle(_context));
    if (path.find("https://") == path.begin()) return std::unique_ptr<BackendHandle>(new RemoteBackendHandle(_context));
#endif
#endif
    else throw context_invalid("path");
}

}   // namespace mori