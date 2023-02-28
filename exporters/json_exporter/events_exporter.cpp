#include <memory>

#include <dlfcn.h>

#include "libmori.hpp"

#include "deps/json.h"

extern "C" __attribute__((visibility("default"))) int events_exporter_entry(std::unique_ptr<mori::exporter::EventsExporter>& ptr, const mori::Context::View& _context);

namespace mori {
namespace exporter {

using json = nlohmann::json;

struct JSONEventsExporter : public EventsExporter {
    JSONEventsExporter(const Context::View& _context): EventsExporter(_context) {}

    virtual void onEvent(const events::MemoryEvent& event) override {
        json obj;
        obj["tensor"] = event.tensor;
        obj["type"] = events::util::get_event_type_str(event.type);
        obj["timestamp"] = events::util::get_timestamp_val(event.timestamp);

        export_method->exportMessage(obj.dump());
    }

};  // struct JSONEventsExporter

}   // namespace exporter
}   // namespace mori

int events_exporter_entry(std::unique_ptr<mori::exporter::EventsExporter>& ptr, const mori::Context::View& _context) {
    // Backend should be explictly destroyed before the dylib released.
    // Therefore, a heap object and a shared pointer.
    // Set up events exporter here.
    ptr.reset(new mori::exporter::JSONEventsExporter(_context));
    return 0;
} 
