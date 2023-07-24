#pragma once

#include <fstream>
#include <iostream>
#include <memory>

#include "backend/dylibs_util.hpp"

#include "includes/context.hpp"
#include "includes/memory_status.hpp"
#include "includes/memory_layout.hpp"
#include "includes/memory_event.hpp"
#include "includes/execution_event.hpp"
#include "includes/memory_schedule_event.hpp"

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

    EventsExporter(const Context::View& context) {
        std::string export_method_name = context.at("method");
        Context::View context_view = context.view("method");
        if (export_method_name == "empty") export_method.reset(new exportimpl::ExportMethod(context_view));
        else if (export_method_name == "file") export_method.reset(new exportimpl::FileExportMethod(context_view));
        else hInst = utils::load_dylib("Events Export Method", context.at("method.path"), "export_method_entry", export_method, context_view);
    }

    virtual void onMemoryEvent(const events::MemoryEvent& event) const {}
    virtual void onExecutionEvent(const events::ExecutionEvent& event) const {}

    virtual ~EventsExporter() {
        if (hInst) dlclose(hInst);
    }
};  // struct EventsExporter

struct TensorsExporter {
    std::unique_ptr<exportimpl::ExportMethod> export_method;
    void* hInst = nullptr;

    TensorsExporter(const Context::View& context) {
        std::string export_method_name = context.at("method");
        Context::View context_view = context.view("method");
        if (export_method_name == "empty") export_method.reset(new exportimpl::ExportMethod(context_view));
        else if (export_method_name == "file") export_method.reset(new exportimpl::FileExportMethod(context_view));
        else hInst = utils::load_dylib("Events Export Method", context.at("method.path"), "export_method_entry", export_method, context_view);
    }
    virtual void onTensors(status::MemoryStatus& status) const {}

    virtual ~TensorsExporter() {
        if (hInst) dlclose(hInst);
    }
};  // struct TensorExporter

struct ScheduleExporter {
    std::unique_ptr<exportimpl::ExportMethod> export_method;
    void* hInst = nullptr;

    ScheduleExporter(const Context::View& context) {
        std::string export_method_name = context.at("method");
        Context::View context_view = context.view("method");
        if (export_method_name == "empty") export_method.reset(new exportimpl::ExportMethod(context_view));
        else if (export_method_name == "file") export_method.reset(new exportimpl::FileExportMethod(context_view));
        else hInst = utils::load_dylib("Schedule Events Export Method", context.at("method.path"), "export_method_entry", export_method, context_view);
    }
    virtual void onScheduleEvents(const events::ScheduleEvents& events) const {}

    virtual ~ScheduleExporter() {
        if (hInst) dlclose(hInst);
    }
};  // struct TensorExporter

}   // namespace exporter
}   // namespace mori