#include <memory>

#include <dlfcn.h>

#include "libmori.hpp"

#include "deps/json.h"

extern "C" __attribute__((visibility("default"))) int events_exporter_entry(std::unique_ptr<mori::exporter::EventsExporter>& ptr, const mori::Context::View& _context);

namespace mori {
namespace events {

static void to_json(nlohmann::json& obj, const MemoryEvent& event) {
    obj["type"] = "memory";
    obj["event"]["operator"] = event.op;
    obj["event"]["tensor"] = event.tensor;
    obj["event"]["size"]   = event.size;
    obj["event"]["type"] = events::utils::get_event_type_str(event.type);
    obj["event"]["stage"] = mori::utils::get_application_stage_str(event.stage);
    obj["event"]["timestamp"] = mori::utils::get_timestamp_val(event.timestamp);
}

static void to_json(nlohmann::json& obj, const ExecutionEvent& event) {
    obj["type"] = "execution";
    obj["event"]["operator"] = event.op;
    obj["event"]["type"] = events::utils::get_event_type_str(event.type);
    obj["event"]["stage"] = mori::utils::get_application_stage_str(event.stage);
    obj["event"]["timestamp"] = mori::utils::get_timestamp_val(event.timestamp);
}

}   // namespace events

namespace exporter {

using json = nlohmann::json;

struct JSONEventsExporter : public EventsExporter {
    JSONEventsExporter(const Context::View& _context): EventsExporter(_context) {}

    virtual void onMemoryEvent(const events::MemoryEvent& event) const override {
        json obj = event;
        export_method->exportMessage(obj.dump(2));
    }
    virtual void onExecutionEvent(const events::ExecutionEvent& event) const override {
        json obj = event;
        export_method->exportMessage(obj.dump(2));
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
