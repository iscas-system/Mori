#include <memory>

#include <dlfcn.h>

// Include mori here.
#include "libmori.hpp"

extern "C" __attribute__((visibility("default"))) int tensors_exporter_entry(std::unique_ptr<mori::exporter::TensorsExporter>& ptr, const mori::Context::View& _context);

int tensors_exporter_entry(std::unique_ptr<mori::exporter::TensorsExporter>& ptr, const mori::Context::View& _context) {
    // Backend should be explictly destroyed before the dylib released.
    // Therefore, a heap object and a shared pointer.
    // Set up tensors exporter here.
    ptr.reset(new mori::exporter::TensorsExporter(_context));
    return 0;
} 
