#include <memory>

#include <dlfcn.h>

// Include mori here.
#include "libmori.hpp"

#include "deps/json.h"

extern "C" __attribute__((visibility("default"))) int tensors_exporter_entry(std::unique_ptr<mori::exporter::TensorsExporter>& ptr, const mori::Context::View& _context);

namespace mori {
namespace status {

static void to_json(nlohmann::json& obj, const TensorPres& pres) {
    obj["name"] = pres.getName();
    obj["size"] = pres.getSize();
    obj["type"] = status::utils::get_tensor_type_str(pres.getType());
    obj["persistent"] = pres.isPersistent();
    obj["transient"]  = pres.isTransient();
}

static void to_json(nlohmann::json& obj, const OperatorPres& pres) {
    obj["name"] = pres.getName();
    obj["backprop"] = pres.isBackwardPropagation();
    obj["tensors"] = pres.getTensors();
    obj["prevs"] = pres.getPrevs();
    obj["posts"] = pres.getPosts();
}

}   // namespace status

namespace exporter {

using json = nlohmann::json;

struct JSONTensorsExporter : public TensorsExporter {
    JSONTensorsExporter(const Context::View& context): TensorsExporter(context) {}
    virtual void onTensors(status::MemoryStatus& status) const override {
        json obj;

        obj["tensors"] = json();
        obj["operators"] = json();

        for (auto &s : status.getTensors()) {
            status::TensorPres pres = status.referenceTensor(s);
            obj["tensors"][s] = pres;
        }
        for (auto &s : status.getOperators()) {
            status::OperatorPres pres = status.referenceOperator(s);
            obj["operators"][s] = pres;
        }

        obj["entry"] = status.getEntry();
        obj["execution_order"] = status.getExecutionOrder();

        export_method->exportMessage(obj.dump(2));
    }
};  // struct JSONTensorExporter

}   // namespace exporter
}   // namespace mori

int tensors_exporter_entry(std::unique_ptr<mori::exporter::TensorsExporter>& ptr, const mori::Context::View& _context) {
    // Backend should be explictly destroyed before the dylib released.
    // Therefore, a heap object and a shared pointer.
    // Set up tensors exporter here.
    ptr.reset(new mori::exporter::JSONTensorsExporter(_context));
    return 0;
} 
