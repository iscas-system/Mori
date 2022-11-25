#pragma once

#include <fstream>
#include <iostream>
#include <memory>

#include "backend/dylibs_util.hpp"

#include "includes/context.hpp"
#include "includes/memory_status.hpp"

namespace mori {
namespace exporter {

namespace exportimpl {
/**
 * Implementation of export methods.
 */
struct ExportMethod {
    ExportMethod(const Context::View&) {}
    virtual void exportMessage(const std::string&) {}
    virtual ~ExportMethod() = default;
};  // struct ExportMethod

struct FileExportMethod : public ExportMethod {
    std::ofstream fout;
    FileExportMethod(const Context::View& context_view): ExportMethod(context_view) {
        std::string export_file = context_view.at("filename");
        fout.open(export_file);
    }

    virtual void exportMessage(const std::string& message) override {
        fout << message << std::endl;
    }

    virtual ~FileExportMethod() {
        fout.close();
    }
};  // struct FileExportMethod

}   // namespace exportimpl

using ExportMethod = exportimpl::ExportMethod;

/**
 * Export DL memory events.
 */
struct EventsExporter {
    std::unique_ptr<exportimpl::ExportMethod> export_method;
    void* hInst = nullptr;

    EventsExporter(const Context& context) {
        std::string export_method_name = context.at("exporters.events.method");
        Context::View context_view = context.view("exporters.events.method");
        if (export_method_name == "empty") export_method.reset(new exportimpl::ExportMethod(context_view));
        else if (export_method_name == "file") export_method.reset(new exportimpl::FileExportMethod(context_view));
        else {
            typedef int(*ExportMethodType)(std::unique_ptr<exportimpl::ExportMethod>&, const Context::View&);
            hInst = utils::load_dylib<ExportMethodType>("Events Export Method", context.at("exporters.events.method.path"), "export_method_entry", export_method, context_view);
        }
    }

    virtual void onEvent(const events::MemoryEvent& event) {}

    virtual ~EventsExporter() {
        if (hInst) dlclose(hInst);
    }
};  // struct EventsExporter

struct TensorsExporter {
    std::unique_ptr<exportimpl::ExportMethod> export_method;
    void* hInst = nullptr;

    TensorsExporter(const Context& context) {
        std::string export_method_name = context.at("exporters.tensors.method");
        Context::View context_view = context.view("exporters.tensors.method");
        if (export_method_name == "empty") export_method.reset(new exportimpl::ExportMethod(context_view));
        else if (export_method_name == "file") export_method.reset(new exportimpl::FileExportMethod(context_view));
        else {
            typedef int(*ExportMethodType)(std::unique_ptr<exporter::ExportMethod>&, const Context::View&);
            hInst = utils::load_dylib<ExportMethodType>("Events Export Method", context.at("exporters.tensors.method.path"), "export_method_entry", export_method, context_view);
        }
    }
    virtual void onTensor(const status::Tensor& tensor) {}
    virtual void onOperator(const status::Operator& operator_status) {}
    virtual void onEntry(const std::string& op) {}

    virtual ~TensorsExporter() {
        if (hInst) dlclose(hInst);
    }
};  // struct TensorExporter

}   // namespace exporter
}   // namespace mori