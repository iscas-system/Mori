#include <iostream>
#include <map>
#include <utility>
#include <memory>
#include <cassert>
#include <thread>

#include "frontend/libmori.hpp"
#include "demo_memory_manager.hpp"

void* allocate(DemoMemoryManager& mem_manager, mori::MemorySession& session, size_t size) {
    void* re = mem_manager.allocate(size);
    if (re != nullptr) return re;
    
    session.waitMemory(size);
    re = mem_manager.allocate(size);
    assert(re != nullptr);
    return re;
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

    // context["path"] = "dylib://libmori.so";
    // context["exporters.events"]                  = "json";
    // context["exporters.events.path"]             = "build/libmori_exporter_events_json.so";
    // context["exporters.events.method"]           = "file";
    // context["exporters.events.method.filename"]  = "events_export.log";
    // context["exporters.tensors"]                 = "json";
    // context["exporters.tensors.path"]            = "build/libmori_exporter_tensors_json.so";
    // context["exporters.tensors.method"]          = "file";
    // context["exporters.tensors.method.filename"] = "tensor_export.log";

    mori::Frontend frontend(context);
    frontend.setMemoryManager(&mem_manager);
    frontend.setLogger(&logger);

    // Simulate memory pool.
    std::unordered_map<std::string, void*> device_addresses;

    mori::status::Operator o1("o1"), o2("o2"), o3("o3"), o4("o4"), o5("o5");
    o1.setPost("o2");
    o2.setPrev("o1");
    o2.setPost("o3");
    o2.setPost("o4");
    o3.setPrev("o2");
    o3.setPost("o5");
    o4.setPrev("o2");
    o4.setPost("o5");
    o5.setPrev("o3");
    o5.setPrev("o4");
    o1.setTensor("t1");
    o2.setTensor("t2");
    o3.setTensor("t3");
    o4.setTensor("t4");
    o5.setTensor("t5");

    frontend.init();

    frontend.registerTensor(mori::status::Tensor("t1", 1024, mori::status::MemoryDataType::inout));
    frontend.registerTensor(mori::status::Tensor("t2", 512, mori::status::MemoryDataType::inout));
    frontend.registerTensor(mori::status::Tensor("t3", 256, mori::status::MemoryDataType::inout));
    frontend.registerTensor(mori::status::Tensor("t4", 256, mori::status::MemoryDataType::inout));
    frontend.registerTensor(mori::status::Tensor("t5", 128, mori::status::MemoryDataType::inout));

    frontend.registerOperator(o1);
    frontend.registerOperator(o2);
    frontend.registerOperator(o3);
    frontend.registerOperator(o4);
    frontend.registerOperator(o5);
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

    for (int i = 0; i < 3; ++i) {
        std::cout << "Iteration: " << i+1 <<std::endl;
        session.newIteration();
        
        // Forward propagation op1
        void* t1_ptr = allocate(mem_manager, session, 1024);
        device_addresses.emplace("t1", t1_ptr);
        session.setMemoryDataAllocated("o1", "t1", t1_ptr);
        auto r1 = session.createRequest("o1");
        r1.waitTensor("t1");
        r1.setMemoryDataAssigned("t1");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        r1.release();

        // Forward propagation op2
        void* t2_ptr = allocate(mem_manager, session, 512);
        device_addresses.emplace("t2", t2_ptr);
        session.setMemoryDataAllocated("o2", "t2", t2_ptr);
        auto r2 = session.createRequest("o2");
        r2.waitTensor("t1");
        r2.waitTensor("t2");
        r2.setMemoryDataAcquired("t1");
        r2.setMemoryDataAssigned("t2");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        r2.release();

        // Forward propagation op3
        void* t3_ptr = allocate(mem_manager, session, 256);
        device_addresses.emplace("t3", t3_ptr);
        session.setMemoryDataAllocated("o3", "t3", t3_ptr);
        auto r3 = session.createRequest("o3");
        r3.waitTensor("t2");
        r3.waitTensor("t3");
        r3.setMemoryDataAcquired("t2");
        r3.setMemoryDataAssigned("t3");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        r3.release();

        // Forward propagation op4
        void* t4_ptr = allocate(mem_manager, session, 256);
        device_addresses.emplace("t4", t4_ptr);
        session.setMemoryDataAllocated("o4", "t4", t4_ptr);
        auto r4 = session.createRequest("o4");
        r4.waitTensor("t2");
        r4.waitTensor("t4");
        r4.setMemoryDataAcquired("t2");
        r4.setMemoryDataAssigned("t4");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        r4.release();

        // Forward propagation op5
        void* t5_ptr = allocate(mem_manager, session, 128);
        device_addresses.emplace("t5", t5_ptr);
        session.setMemoryDataAllocated("o5", "t5", t5_ptr);
        auto r5 = session.createRequest("o5");
        r5.waitTensor("t3");
        r5.waitTensor("t4");
        r5.waitTensor("t5");
        r5.setMemoryDataAcquired("t3");
        r5.setMemoryDataAssigned("t4");
        r5.setMemoryDataAssigned("t5");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        r5.release();

        session.halfIteration();

        // Backward propagation op5
        auto r5b = session.createRequest("o5");
        r5b.waitTensor("t3");
        r5b.waitTensor("t4");
        r5b.waitTensor("t5");
        r5b.setMemoryDataAcquired("t3");
        r5b.setMemoryDataAcquired("t4");
        r5b.setMemoryDataAcquired("t5");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        r5b.release();
        free(device_addresses, mem_manager, "t5");
        session.setMemoryDataFreed("o5", "t5");

        // Backward propagation op4
        auto r4b = session.createRequest("o4");
        r4b.waitTensor("t2");
        r4b.waitTensor("t4");
        r4b.setMemoryDataAcquired("t2");
        r4b.setMemoryDataAcquired("t4");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        r4b.release();
        free(device_addresses, mem_manager, "t4");
        session.setMemoryDataFreed("o4", "t4");

        // Backward propagation op3
        auto r3b = session.createRequest("o3");
        r3b.waitTensor("t2");
        r3b.waitTensor("t3");
        r3b.setMemoryDataAcquired("t2");
        r3b.setMemoryDataAcquired("t3");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        r3b.release();
        free(device_addresses, mem_manager, "t3");
        session.setMemoryDataFreed("o3", "t3");

        // Backward propagation op2
        auto r2b = session.createRequest("o2");
        r2b.waitTensor("t1");
        r2b.waitTensor("t2");
        r2b.setMemoryDataAcquired("t1");
        r2b.setMemoryDataAcquired("t2");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        r2b.release();
        free(device_addresses, mem_manager, "t2");
        session.setMemoryDataFreed("o2", "t2");

        // Backward propagation op1
        auto r1b = session.createRequest("o1");
        r1b.waitTensor("t1");
        r1b.setMemoryDataAcquired("t1");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        r1b.release();
        free(device_addresses, mem_manager, "t1");
        session.setMemoryDataFreed("o1", "t1");

        frontend.updateSchedule();
        std::cout << "Iteration: " << i+1 << " end.\n\n";
        device_addresses.clear();
    }

    frontend.stop();
    frontend.terminate();
    
    std::cout<<"Hello world!\n";
    return 0;
}