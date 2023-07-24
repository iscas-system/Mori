#include <memory>

#include <dlfcn.h>

// Include mori here.
#include "libmori.hpp"

extern "C" __attribute__((visibility("default"))) int schedule_exporter_entry(std::unique_ptr<mori::exporter::ScheduleExporter>& ptr, const mori::Context::View& _context);

int schedule_exporter_entry(std::unique_ptr<mori::exporter::EventsExporter>& ptr, const mori::Context::View& _context) {
    // Backend should be explictly destroyed before the dylib released.
    // Therefore, a heap object and a shared pointer.
    // Set up events exporter here.
    ptr.reset(new mori::exporter::ScheduleExporter(_context));
    return 0;
} 
