#include <memory>

#include <dlfcn.h>

// Include mori here.
#include "libmori.hpp"

#include "deps/json.h"

extern "C" __attribute__((visibility("default"))) int tensors_exporter_entry(std::unique_ptr<mori::exporter::TensorsExporter>& ptr, const mori::Context& _context);

namespace mori {
namespace exporter {

using json = nlohmann::json;

struct JSONTensorsExporter : public TensorsExporter {
    JSONTensorsExporter(const Context& context): TensorsExporter(context) {}
    virtual void onTensor(const status::Tensor& tensor) override {
        json obj;
        obj["catagory"] = "tensor";
        obj["name"] = tensor.getName();
        obj["size"] = tensor.getSize();
        obj["type"] = status::util::get_tensor_type_str(tensor.getType());
        obj["persistent"] = tensor.isPersistant();

        export_method->exportMessage(obj.dump());
    }
    virtual void onOperator(const status::Operator& operator_status) override {
        json obj;
        obj["catagory"] = "operator";
        obj["name"] = operator_status.getName();
        obj["backprop"] = operator_status.isBackwardPropagation();
        obj["tensors"] = operator_status.getTensors();

        export_method->exportMessage(obj.dump());
    }
    virtual void onEntry(const std::string& op) override {
        json obj;
        obj["catagory"] = "entry";
        obj["operator"] = op;

        export_method->exportMessage(obj.dump());
    }
};  // struct JSONTensorExporter

}   // namespace exporter
}   // namespace mori

int tensors_exporter_entry(std::unique_ptr<mori::exporter::TensorsExporter>& ptr, const mori::Context& _context) {
    // Backend should be explictly destroyed before the dylib released.
    // Therefore, a heap object and a shared pointer.
    // Set up tensors exporter here.
    ptr.reset(new mori::exporter::JSONTensorsExporter(_context));
    return 0;
} 
