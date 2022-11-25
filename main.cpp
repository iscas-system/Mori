#include <iostream>
#include <map>
#include <utility>
#include <memory>
#include <cassert>
#include <thread>

#include "frontend/libmori.hpp"
#include "demo_memory_manager.hpp"

void* allocate(DemoMemoryManager& mem_manager, mori::MemorySession& session, size_t size) {
    session.waitMemory(size);
    return mem_manager.allocate(size);
}

void free(std::unordered_map<std::string, void*>& device_addresses, DemoMemoryManager& mem_manager, const std::string& tensor) {
    auto p = device_addresses.find(tensor);
    assert(p != device_addresses.end());
    mem_manager.free(p->second);
    device_addresses.erase(p);
}

int main(int argc, char** argv) {
    mori::Context context;
    DemoMemoryManager mem_manager;
    mori::StdIOLogger logger;

    //context["path"] = "dylib://libmori.so";
    context["exporters.events"]                  = "json";
    context["exporters.events.path"]             = "build/libmori_exporter_events_json.so";
    context["exporters.events.method"]           = "file";
    context["exporters.events.method.filename"]  = "events_export.log";
    context["exporters.tensors"]                 = "json";
    context["exporters.tensors.path"]            = "build/libmori_exporter_tensors_json.so";
    context["exporters.tensors.method"]          = "file";
    context["exporters.tensors.method.filename"] = "tensor_export.log";

    mori::Frontend frontend(context);
    frontend.setMemoryManager(&mem_manager);
    frontend.setLogger(&logger);

    // Simulate memory pool.
    std::unordered_map<std::string, void*> device_addresses;

    mori::status::Operator o1("o1"), o2("o2"), o3("o3");
    o1.setPost("o2");
    o2.setPrev("o1");
    o2.setPost("o3");
    o3.setPrev("o1");
    o1.setTensor("t1");
    o2.setTensor("t2");
    o3.setTensor("t3");

    frontend.init();

    frontend.registerTensor(mori::status::Tensor("t1", 1024, mori::status::MemoryDataType::inout));
    frontend.registerTensor(mori::status::Tensor("t2", 1024, mori::status::MemoryDataType::inout));
    frontend.registerTensor(mori::status::Tensor("t3", 1024, mori::status::MemoryDataType::inout));

    frontend.registerOperator(o1);
    frontend.registerOperator(o2);
    frontend.registerOperator(o3);
    frontend.setEntry("o1");

    frontend.setCallback(mori::CallbackStage::postSwapOut, [&device_addresses](const std::string& tensor, void* address) {
        auto p = device_addresses.find(tensor);
        if (p == device_addresses.end()) return 1;
        p->second = nullptr;
        return 0;
    });

    frontend.setCallback(mori::CallbackStage::postSwapIn, [&device_addresses](const std::string& tensor, void* address) {
        auto p = device_addresses.find(tensor);
        if (p == device_addresses.end()) return 1;
        p->second = address;
        return 0;
    });
    
    frontend.start();
    mori::MemorySession& session = frontend.getSession();

    // Forward propagation op1
    void* t1_ptr = allocate(mem_manager, session, 1024);
    device_addresses.emplace("t1", t1_ptr);
    session.setMemoryDataAllocated("t1", t1_ptr);
    auto r1 = session.createRequest();
    r1.waitTensor("t1");
    r1.setMemoryDataAssigned("t1");
    r1.release();

    // Forward propagation op2
    void* t2_ptr = allocate(mem_manager, session, 1024);
    device_addresses.emplace("t2", t2_ptr);
    session.setMemoryDataAllocated("t2", t2_ptr);
    auto r2 = session.createRequest();
    r2.waitTensor("t2");
    r2.setMemoryDataAssigned("t2");
    r2.release();

    // Forward propagation op3
    void* t3_ptr = allocate(mem_manager, session, 1024);
    device_addresses.emplace("t3", t3_ptr);
    session.setMemoryDataAllocated("t3", t3_ptr);
    auto r3 = session.createRequest();
    r3.waitTensor("t3");
    r3.setMemoryDataAssigned("t3");
    r3.release();

    // Backward propagation op3
    auto r3b = session.createRequest();
    r3b.waitTensor("t3");
    r3b.setMemoryDataAcquired("t3");
    r3b.release();
    free(device_addresses, mem_manager, "t3");
    session.setMemoryDataFreed("t3");

    // Backward propagation op2
    auto r2b = session.createRequest();
    r2b.waitTensor("t2");
    r2b.setMemoryDataAcquired("t2");
    r2b.release();
    free(device_addresses, mem_manager, "t2");
    session.setMemoryDataFreed("t2");

    // Backward propagation op1
    auto r1b = session.createRequest();
    r1b.waitTensor("t1");
    r1b.setMemoryDataAcquired("t1");
    r1b.release();
    free(device_addresses, mem_manager, "t1");
    session.setMemoryDataFreed("t1");

    frontend.stop();
    frontend.terminate();
    
    std::cout<<"Hello world!\n";
    return 0;
}