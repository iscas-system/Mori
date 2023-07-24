#include <memory>

#include <dlfcn.h>

#include "libmori.hpp"

#include "deps/json.h"

extern "C" __attribute__((visibility("default"))) int schedule_exporter_entry(std::unique_ptr<mori::exporter::ScheduleExporter>& ptr, const mori::Context::View& _context);

namespace mori {
namespace layout {

static void to_json(nlohmann::json& obj, const Region& region) {
    obj["name"]          = region.name;
    obj["size"]          = region.size;
    obj["sections"]      = region.sections;
    obj["fragment_size"] = region.fragment_size;
}

static void to_json(nlohmann::json& obj, const Layer& layer) {
    obj["regions"]        = layer.regions;
    obj["size"]           = layer.size;
    obj["requested_size"] = layer.requested_size;
}

}   // namespace layout

namespace events {

static void to_json(nlohmann::json& obj, const ScheduleEvent& event) {
    obj["operator"]      = event.operator_name;
    obj["tensor"]        = event.tensor_name;
    obj["size"]          = event.size;
    obj["type"]          = event.type;
    obj["post_operator"] = event.postop;
    obj["timepoint"]     = event.timepoint;
}

}   // namespace events

namespace exporter {

using json = nlohmann::json;

struct JSONScheduleExporter : public ScheduleExporter {
    JSONScheduleExporter(const Context::View& _context): ScheduleExporter(_context) {}

    virtual void onScheduleEvents(const events::ScheduleEvents& events) const {
        json obj;
        obj["memory_map"] = json();

        obj["memory_map"]["regions"] = events.memory_map.getRegions();
        obj["memory_map"]["layers"]  = events.memory_map.getLayers();

        obj["forward_schedule_events"] = json();
        obj["backward_schedule_events"] = json();

        obj["forward_schedule_events"]["execution"] = events.forward_schedule_events.execution;
        obj["forward_schedule_events"]["timepoint"] = events.forward_schedule_events.timepoint;
        obj["backward_schedule_events"]["execution"] = events.backward_schedule_events.execution;
        obj["backward_schedule_events"]["timepoint"] = events.backward_schedule_events.timepoint;

        export_method->exportMessage(obj.dump(2));
    }

};  // struct JSONEventsExporter

}   // namespace exporter
}   // namespace mori

int schedule_exporter_entry(std::unique_ptr<mori::exporter::ScheduleExporter>& ptr, const mori::Context::View& _context) {
    // Backend should be explictly destroyed before the dylib released.
    // Therefore, a heap object and a shared pointer.
    // Set up events exporter here.
    ptr.reset(new mori::exporter::JSONScheduleExporter(_context));
    return 0;
} 
