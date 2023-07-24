#include <iostream>

#include "frontend/libmori.hpp"
#include "demo_memory_manager.hpp"
#include "dl_model.hpp"

int main(int argc, char** argv) {
    mori::Context context;
    DemoMemoryManager mem_manager;
    mori::StdIOLogger logger;

    // context["path"] = "dylib://libmori.so";
    // context["exporters.events"]                   = "json";
    // context["exporters.events.path"]              = "build/libmori_exporter_events_json.so";
    // context["exporters.events.method"]            = "file";
    // context["exporters.events.method.filename"]   = "events_export.log";
    // context["exporters.tensors"]                  = "json";
    // context["exporters.tensors.path"]             = "build/libmori_exporter_tensors_json.so";
    // context["exporters.tensors.method"]           = "file";
    // context["exporters.tensors.method.filename"]  = "tensor_export.log";
    context["exporters.schedule"]                 = "json";
    context["exporters.schedule.path"]            = "build/libmori_exporter_schedule_json.so";
    context["exporters.schedule.method"]          = "file";
    context["exporters.schedule.method.filename"] = "schedule_export.log";

    mori::Frontend frontend(context);
    frontend.setMemoryManager(&mem_manager);
    frontend.setLogger(&logger);
    frontend.init();

    Model model(frontend, mem_manager);
    
    // Create simulation operators
    // Forward propagation
    Operator o1,  o2,  o3,  o4,  o5,  o6,  o7,  o8,  o9,  o10, o11, o12;
    // Backward propagation
    Operator o13, o14, o15, o16, o17, o18, o19, o20, o21, o22, o23, o24;

    o1.name = "o1";
    o1.tensors.insert("t1");
    o1.tensors.insert("w1");
    o1.posts.insert("o2");
    o1.posts.insert("o3");
    o1.posts.insert("o24");
    model.setTensor(Tensor("t1", 512));
    model.setOperator(o1);

    o2.name = "o2";
    o2.tensors.emplace("t2");
    o1.tensors.insert("w1");
    o2.posts.insert("o4");
    o2.posts.insert("o23");
    o2.prevs.insert("o1");
    model.setTensor(Tensor("t2", 384));
    model.setOperator(o2);

    o3.name = "o3";
    o3.tensors.emplace("t3");
    o1.tensors.insert("w1");
    o3.posts.insert("o5");
    o3.posts.insert("o22");
    o3.prevs.insert("o1");
    model.setTensor(Tensor("t3", 384));
    model.setOperator(o3);

    o4.name = "o4";
    o4.tensors.emplace("t4");
    o1.tensors.insert("w1");
    o4.posts.insert("o6");
    o4.posts.insert("o21");
    o4.prevs.insert("o2");
    model.setTensor(Tensor("t4", 256));
    model.setOperator(o4);

    o5.name = "o5";
    o5.tensors.emplace("t5");
    o1.tensors.insert("w1");
    o5.posts.insert("o6");
    o5.posts.insert("o20");
    o5.prevs.insert("o3");
    model.setTensor(Tensor("t5", 256));
    model.setOperator(o5);

    o6.name = "o6";
    o6.tensors.emplace("t6");
    o1.tensors.insert("w1");
    o6.posts.insert("o7");
    o6.posts.insert("o8");
    o6.posts.insert("o19");
    o6.prevs.insert("o4");
    o6.prevs.insert("o5");
    model.setTensor(Tensor("t6", 256));
    model.setOperator(o6);

    o7.name = "o7";
    o7.tensors.emplace("t7");
    o1.tensors.insert("w1");
    o7.posts.insert("o9");
    o7.posts.insert("o18");
    o7.prevs.insert("o6");
    model.setTensor(Tensor("t7", 258));
    model.setOperator(o7);

    o8.name = "o8";
    o8.tensors.emplace("t8");
    o1.tensors.insert("w1");
    o8.posts.insert("o9");
    o8.posts.insert("o17");
    o8.prevs.insert("o6");
    model.setTensor(Tensor("t8", 258));
    model.setOperator(o8);

    o9.name = "o9";
    o9.tensors.emplace("t9");
    o1.tensors.insert("w1");
    o9.posts.insert("o10");
    o9.posts.insert("o16");
    o9.prevs.insert("o7");
    o9.prevs.insert("o8");
    model.setTensor(Tensor("t9", 256));
    model.setOperator(o9);

    o10.name = "o10";
    o10.tensors.emplace("t10");
    o1.tensors.insert("w1");
    o10.posts.insert("o11");
    o10.posts.insert("o15");
    o10.prevs.insert("o9");
    model.setTensor(Tensor("t10", 256));
    model.setOperator(o10);

    o11.name = "o11";
    o11.tensors.emplace("t11");
    o1.tensors.insert("w1");
    o11.posts.insert("o12");
    o11.posts.insert("o14");
    o11.prevs.insert("o10");
    model.setTensor(Tensor("t11", 124));
    model.setOperator(o11);

    o12.name = "o12";
    o12.tensors.emplace("t12");
    o1.tensors.insert("w1");
    o12.posts.insert("o13");
    o12.prevs.insert("o11");
    model.setTensor(Tensor("t12", 124));
    model.setOperator(o12);

    o13.name = "o13";
    o13.posts.insert("o14");
    o13.prevs.insert("o12");
    o13.prevs.insert("o12");
    o13.backward = true;
    model.setOperator(o13);

    o14.name = "o14";
    o14.posts.insert("o15");
    o14.prevs.insert("o11");
    o14.prevs.insert("o13");
    o14.backward = true;
    model.setOperator(o14);

    o15.name = "o15";
    o15.posts.insert("o16");
    o15.prevs.insert("o10");
    o15.prevs.insert("o14");
    o15.backward = true;
    model.setOperator(o15);

    o16.name = "o16";
    o16.posts.insert("o17");
    o16.posts.insert("o18");
    o16.prevs.insert("o9");
    o16.prevs.insert("o15");
    o16.backward = true;
    model.setOperator(o16);

    o17.name = "o17";
    o17.posts.insert("o19");
    o17.prevs.insert("o8");
    o17.prevs.insert("o16");
    o17.backward = true;
    model.setOperator(o17);

    o18.name = "o18";
    o18.posts.insert("o19");
    o18.prevs.insert("o7");
    o18.prevs.insert("o16");
    o18.backward = true;
    model.setOperator(o18);

    o19.name = "o19";
    o19.posts.insert("o20");
    o19.posts.insert("o21");
    o19.prevs.insert("o6");
    o19.prevs.insert("o17");
    o19.prevs.insert("o18");
    o19.backward = true;
    model.setOperator(o19);

    o20.name = "o20";
    o20.posts.insert("o22");
    o20.prevs.insert("o5");
    o20.prevs.insert("o19");
    o20.backward = true;
    model.setOperator(o20);

    o21.name = "o21";
    o21.posts.insert("o23");
    o21.prevs.insert("o4");
    o21.prevs.insert("o19");
    o21.backward = true;
    model.setOperator(o21);

    o22.name = "o22";
    o22.posts.insert("o24");
    o22.prevs.insert("o3");
    o22.prevs.insert("o21");
    o22.backward = true;
    model.setOperator(o22);

    o23.name = "o23";
    o23.posts.insert("o24");
    o23.prevs.insert("o2");
    o23.prevs.insert("o21");
    o23.backward = true;
    model.setOperator(o23);

    o24.name = "o24";
    o24.prevs.insert("o22");
    o24.prevs.insert("o1");
    o24.prevs.insert("o23");
    o24.backward = true;
    model.setOperator(o24);

    model.setEntry("o1");

    // Simulate memory pool.
    model.init();
    frontend.start();
    std::cout<<std::endl;

    for (int i = 0; i < 3; ++i) {
        std::cout << "Iteration: " << i+1 << std::endl;
        model.execute();
        frontend.updateSchedule();
        std::cout << "Iteration: " << i+1 << " end.\n\n";
    }

    frontend.stop();
    frontend.terminate();
    
    std::cout<<"Hello world!\n";
    return 0;
}