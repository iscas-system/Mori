#include <iostream>
#include <map>
#include <utility>
#include <memory>
#include <cassert>
#include <thread>

#include "frontend/libmori.hpp"
#include "demo_memory_manager.hpp"

int main(int argc, char** argv) {
    mori::Context context;
    DemoMemoryManager mem_manager;
    mori::StdIOLogger logger;

    //context["path"] = "dylib://libmori.so";

    mori::Frontend frontend(context);
    frontend.setMemoryManager(&mem_manager);
    frontend.setLogger(&logger);

    std::unordered_map<std::string, mori::TensorStatus> tensor_status;
    tensor_status.insert(std::make_pair("t", mori::TensorStatus("t", 1024, mori::MemoryType::inout)));
    
    mori::OperatorStatus o1_status("o1", {}, {"o2"}, tensor_status);
    mori::OperatorStatus o2_status("o2", {"o1"}, {"o3"}, tensor_status);
    mori::OperatorStatus o3_status("o3", {"o2"}, {}, tensor_status);

    frontend.init();

    frontend.registerOperator(o1_status);
    frontend.registerOperator(o2_status);
    frontend.registerOperator(o3_status);

    mori::MemorySession& session = frontend.getSession();
    session.init();

    // Forward propagation op1
    session.allocateMemory("o1", "t");
    auto r1 = session.waitData("o1");
    r1.setMemoryDataAssigned("o1", "t");
    r1.releaseData();

    // Forward propagation op2
    session.allocateMemory("o2", "t");
    auto r2 = session.waitData("o2");
    r2.setMemoryDataAssigned("o2", "t");
    r2.releaseData();

    // Forward propagation op3
    session.allocateMemory("o3", "t");
    auto r3 = session.waitData("o3");
    r3.setMemoryDataAssigned("o3", "t");
    r3.releaseData();

    // Backward propagation op3
    auto r3b = session.waitData("o3");
    r3b.setMemoryDataAcquired("o3", "t");
    r3b.releaseData();
    session.freeMemory("o3", "t");

    // Backward propagation op2
    auto r2b = session.waitData("o2");
    r2b.setMemoryDataAcquired("o2", "t");
    r2b.releaseData();
    session.freeMemory("o2", "t");

    // Backward propagation op1
    auto r1b = session.waitData("o1");
    r1b.setMemoryDataAcquired("o1", "t");
    r1b.releaseData();
    session.freeMemory("o1", "t");

    session.terminate();
    frontend.terminate();
    
    std::cout<<"Hello world!\n";
    return 0;
}

