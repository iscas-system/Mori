#pragma once

#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <mutex>

#include "frontend/libmori.hpp"

struct Tensor {
    std::string name = "";
    size_t size = 0;
    void* address = nullptr;

    Tensor() = default;
    Tensor(const std::string& _name, size_t _size): name(_name), size(_size) {}
};  // struct Tensor

struct Operator {
    std::string name;

    std::unordered_set<std::string> tensors;

    std::unordered_set<std::string> prevs;
    std::unordered_set<std::string> posts;

    bool backward = false;

    unsigned long process_time = 60;
};  // Operator

struct Model final {
private:
    mori::Frontend& frontend;
    DemoMemoryManager& mem_manager;

    std::string name;

    std::unordered_map<std::string, Tensor>   tensors;
    std::unordered_map<std::string, Operator> operators;
    std::vector<std::string> execution_order;
    std::string entry;

    std::mutex m;

protected:
    void* allocate(size_t size) {
        void* re = mem_manager.allocate(size);
        if (re != nullptr) return re;
        
        frontend.getSession().waitMemory(size);
        re = mem_manager.allocate(size);
        assert(re != nullptr);
        return re;
    }

public:
    Model(mori::Frontend& _frontend, DemoMemoryManager& _mem_manager): frontend(_frontend), mem_manager(_mem_manager) {}

    void setTensor(const Tensor& _tensor) {
        tensors.emplace(_tensor.name, _tensor);
    }

    void setOperator(const Operator& _operator) {
        mori::status::Operator mo(_operator.name);
        for (auto &s : _operator.posts) mo.setPost(s);
        for (auto &s : _operator.prevs) mo.setPrev(s);
        for (auto &x : _operator.tensors) {
            frontend.registerTensor(mori::status::Tensor(x, tensors.at(x).size, mori::status::MemoryDataType::inout));
            mo.setTensor(x);
        }
        if (_operator.backward) mo.setBackwardPropagation(true);
        frontend.registerOperator(mo);
        operators.emplace(_operator.name, _operator);
        execution_order.push_back(_operator.name);
    }

    void setEntry(const std::string& _entry) {
        frontend.setEntry(_entry);
        entry = _entry;
    }

    void init() {
        frontend.setCallback(mori::CallbackStage::postSwapOut, [this](const std::string& tensor, void* address) {
            std::unique_lock<std::mutex> l{m};
            auto& t = tensors.at(tensor);
            t.address = nullptr;
            return 0;
        });

        frontend.setCallback(mori::CallbackStage::postSwapIn, [this](const std::string& tensor, void* address) {
            std::unique_lock<std::mutex> l{m};
            auto& t = tensors.at(tensor);
            t.address = address;
            return 0;
        });
    }

    void execute() {
        mori::MemorySession& session = frontend.getSession();
        session.newIteration();

        // Forward propagation.
        auto p = execution_order.begin();
        while (p != execution_order.end()) {
            Operator& op = operators.at(*p);
            if (op.backward) break;

            for (auto &s : op.tensors) {
                Tensor& t = tensors.at(s);
                t.address = allocate(t.size);
                session.setMemoryDataAllocated(*p, s, t.address);
            }

            auto request = session.createRequest(*p);
            for (auto &s : op.tensors) {
                request.waitTensor(s);
                request.setMemoryDataAssigned(s);
            }

            for (auto &s : op.prevs) {
                for (auto &s1 : operators.at(s).tensors) {
                    request.waitTensor(s1);
                    request.setMemoryDataAcquired(s1);
                }
            }

            request.setOperationStarted();
            std::this_thread::sleep_for(std::chrono::milliseconds(op.process_time));
            request.release();

            ++p;
        }

        session.halfIteration();

        // Backward propagation.
        while (p != execution_order.end()) {
            Operator& op = operators.at(*p);
            auto request = session.createRequest(*p);
            for (auto &s : op.prevs) {
                for (auto &s1 : operators.at(s).tensors) {
                    request.waitTensor(s1);
                    request.setMemoryDataAcquired(s1);
                }
            }

            request.setOperationStarted();
            std::this_thread::sleep_for(std::chrono::milliseconds(op.process_time));
            request.release();

            for (auto &s : op.prevs) {
                for (auto &s1 : operators.at(s).tensors) {
                    Tensor& t = tensors.at(s1);
                    session.setMemoryDataFreed(*p, s1);
                    mem_manager.free(t.address);
                    t.address = nullptr;   
                }
            }

            ++p;
        }

        mem_manager.check();
    }

};  // struct Model