#include <iostream>
#include <map>
#include <utility>
#include <memory>
#include <cassert>
#include <thread>

#include "includes/context.hpp"
#include "frontend/frontend.hpp"
#include "frontend/memory_manager.hpp"
#include "demo_memory_manager.hpp"

int main(int argc, char** argv) {
    mori::Context context;
    DemoMemoryManager mem_manager;

    mori::Frontend frontend(context);
    frontend.setMemoryManager(&mem_manager);

    std::unordered_map<std::string, mori::TensorStatus> tensor_status;
    tensor_status.insert(std::make_pair("t", mori::TensorStatus("t", 1024, mori::MemoryType::all)));
    
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
    session.setMemoryDataAssigned("o1", "t");
    session.withData("o1", []() {
        std::cout<<"o1\n";
    });

    // Forward propagation op2
    session.allocateMemory("o2", "t");
    session.setMemoryDataAssigned("o2", "t");
    session.withData("o2", []() {
        std::cout<<"o2\n";
    });

    // Forward propagation op3
    session.allocateMemory("o3", "t");
    session.setMemoryDataAssigned("o3", "t");
    session.withData("o3", []() {
        std::cout<<"o3\n";
    });

    // Backward propagation op3
    session.withData("o3", []() {
        std::cout<<"o3\n";
    });
    session.freeMemory("o3", "t");

    // Backward propagation op2
    session.withData("o2", []() {
        std::cout<<"o2\n";
    });
    session.freeMemory("o2", "t");

    // Backward propagation op1
    session.withData("o1", []() {
        std::cout<<"o1\n";
    });
    session.freeMemory("o1", "t");

    session.terminate();
    frontend.terminate();
    
    std::cout<<"hello world!\n";
    return 0;
}

