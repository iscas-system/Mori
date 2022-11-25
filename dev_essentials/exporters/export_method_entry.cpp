#include <memory>

#include <dlfcn.h>

// Include mori here.
#include "libmori.hpp"

extern "C" __attribute__((visibility("default"))) int export_method_entry(std::unique_ptr<mori::exporter::exportimpl::ExportMethod>& ptr, const mori::Context::View&);

int export_method_entry(std::unique_ptr<mori::exporter::EventsExporter>& ptr, const mori::Context::View& _view) {
    // Backend should be explictly destroyed before the dylib released.
    // Therefore, a heap object and a shared pointer.
    // Set up events exporter here.
    ptr.reset(new mori::exporter::exportimpl::ExportMethod(_view));
    return 0;
} 
