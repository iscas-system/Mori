#include <iostream>

#include "frontend/libmori.hpp"
#include "demo_memory_manager.hpp"
#include "dl_model.hpp"

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
    frontend.init();

    Model model(frontend, mem_manager);
    
    // Create simulation operators
    Operator o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12;

    o1.name = "o1";
    o1.tensors.insert("t1");
    o1.posts.insert("o2");
    o1.posts.insert("o3");
    model.setTensor(Tensor("t1", 512));
    model.setOperator(o1);

    o2.name = "o2";
    o2.tensors.emplace("t2");
    o2.posts.insert("o4");
    o2.prevs.insert("o1");
    model.setTensor(Tensor("t2", 384));
    model.setOperator(o2);

    o3.name = "o3";
    o3.tensors.emplace("t3");
    o3.posts.insert("o5");
    o3.prevs.insert("o1");
    model.setTensor(Tensor("t3", 384));
    model.setOperator(o3);

    o4.name = "o4";
    o4.tensors.emplace("t4");
    o4.posts.insert("o6");
    o4.prevs.insert("o2");
    model.setTensor(Tensor("t4", 256));
    model.setOperator(o4);

    o5.name = "o5";
    o5.tensors.emplace("t5");
    o5.posts.insert("o6");
    o5.prevs.insert("o3");
    model.setTensor(Tensor("t5", 256));
    model.setOperator(o5);

    o6.name = "o6";
    o6.tensors.emplace("t6");
    o6.posts.insert("o7");
    o6.posts.insert("o8");
    o6.prevs.insert("o4");
    o6.prevs.insert("o5");
    model.setTensor(Tensor("t6", 256));
    model.setOperator(o6);

    o7.name = "o7";
    o7.tensors.emplace("t7");
    o7.posts.insert("o9");
    o7.prevs.insert("o6");
    model.setTensor(Tensor("t7", 258));
    model.setOperator(o7);

    o8.name = "o8";
    o8.tensors.emplace("t8");
    o8.posts.insert("o9");
    o8.prevs.insert("o6");
    model.setTensor(Tensor("t8", 258));
    model.setOperator(o8);

    o9.name = "o9";
    o9.tensors.emplace("t9");
    o9.posts.insert("o10");
    o9.prevs.insert("o7");
    o9.prevs.insert("o8");
    model.setTensor(Tensor("t9", 256));
    model.setOperator(o9);

    o10.name = "o10";
    o10.tensors.emplace("t10");
    o10.posts.insert("o11");
    o10.prevs.insert("o9");
    model.setTensor(Tensor("t10", 256));
    model.setOperator(o10);

    o11.name = "o11";
    o11.tensors.emplace("t11");
    o11.posts.insert("o12");
    o11.prevs.insert("o10");
    model.setTensor(Tensor("t11", 124));
    model.setOperator(o11);

    o12.name = "o12";
    o12.tensors.emplace("t12");
    o12.prevs.insert("o11");
    model.setTensor(Tensor("t12", 124));
    model.setOperator(o12);

    model.setEntry("o1");

    // Simulate memory pool.
    model.init();
    frontend.start();

    for (int i = 0; i < 3; ++i) {
        std::cout << "Iteration: " << i+1 <<std::endl;
        model.execute();
        frontend.updateSchedule();
        std::cout << "Iteration: " << i+1 << " end.\n\n";
    }

    frontend.stop();
    frontend.terminate();
    
    std::cout<<"Hello world!\n";
    return 0;
}