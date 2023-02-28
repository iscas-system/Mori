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

#include "backend/events.hpp"
#include "backend/basic_backend.hpp"
#include "backend/schedulers/memory_scheduler.hpp"
#include "includes/memory_status.hpp"
#include "includes/backend.hpp"
#include "includes/context.hpp"
#include "includes/memory_event.hpp"
#include "includes/exceptions.hpp"

extern "C" __attribute__((visibility("default"))) int backend_entry(std::unique_ptr<mori::Backend>& ptr, const mori::Context& _context);

int backend_entry(std::unique_ptr<mori::Backend>& ptr, const mori::Context& _context) {
    // Backend should be explictly destroyed before the dylib released.
    // Therefore, a heap object and a shared pointer.
    ptr.reset(new mori::BasicBackend(_context));
    return 0;
} 
