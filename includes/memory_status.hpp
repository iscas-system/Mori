#pragma once

#include "includes/stdlibs.hpp"
#include "includes/exceptions.hpp"

namespace mori {

enum MemoryType {
    all, inout, weight, workspace
};  // enum MemoryType

enum MemoryDataStatusType {
    none, empty, device, host, coexist, swapin, swapout
};  // enum MemoryDataStatusType

/**
 * TensorStatus
 * Memory status of an operator tensor.
 */
struct TensorStatus {
    // These four variables can be regarded as const variables.
    std::string name = "";
    void* host_address = nullptr;
    void* device_address = nullptr;
    size_t size = 0;
    MemoryType type = all;

    MemoryDataStatusType data_status = none;

    // If the tensor status is locked. Only the mutex holder can modify the status.
    // Specifically, update memory data status.
    std::shared_mutex status_mutex;

    TensorStatus(): name(""), host_address(nullptr), device_address(nullptr), size(0), type(MemoryType::all), data_status(MemoryDataStatusType::none) {}

    TensorStatus(const std::string& _name, size_t _size, MemoryType _type): name(_name), host_address(nullptr), device_address(nullptr), size(_size), type(_type), data_status(MemoryDataStatusType::none) {}

    TensorStatus(const TensorStatus& status) {
        name = status.name;
        host_address = status.host_address;
        device_address = status.device_address;
        size = status.size;
        type = status.type;
        data_status = status.data_status;
    }
    TensorStatus(TensorStatus&& status) {
        name = move(status.name);
        host_address = status.host_address;
        device_address = status.device_address;
        size = status.size;
        type = status.type;
        data_status = status.data_status;
    }

    void operator=(const TensorStatus& status) {
        name = status.name;
        host_address = status.host_address;
        device_address = status.device_address;
        size = status.size;
        type = status.type;
        data_status = status.data_status;
    }
    void operator=(TensorStatus&& status) {
        name = move(status.name);
        host_address = status.host_address;
        device_address = status.device_address;
        size = status.size;
        type = status.type;
        data_status = status.data_status;
    }

};  // struct Tensor Status

/**
 * OperatorStatus
 * Memory status of an operator.
 */
struct OperatorStatus {
    using iterator = std::unordered_map<std::string, TensorStatus>::iterator;
    using const_iterator = std::unordered_map<std::string, TensorStatus>::const_iterator;

    // These three variables be regarded as const variables.
    std::string name = "";
    // The prev and post operators in the graph.
    std::vector<std::string> prevs, posts;

    std::unordered_map<std::string, TensorStatus> status;

    std::shared_mutex status_mutex;

    OperatorStatus(): name(""), prevs({}), posts({}), status({}) {}

    OperatorStatus(const std::string& _name, const std::vector<std::string>& _prevs, const std::vector<std::string>& _posts, const std::unordered_map<std::string, TensorStatus>& _status): name(_name), prevs(_prevs), posts(_posts), status(_status) {}

    OperatorStatus(const OperatorStatus& op_status) {
        name = op_status.name;
        prevs = op_status.prevs;
        posts = op_status.posts;
        status = op_status.status;
    }

    OperatorStatus(OperatorStatus&& op_status) {
        name = std::move(op_status.name);
        prevs = std::move(op_status.prevs);
        posts = std::move(op_status.posts);
        status = std::move(op_status.status);
    }

    std::vector<std::string> getPrevs() { return prevs; }
    std::vector<std::string> getPosts() { return posts; }

    void operator=(const OperatorStatus& op_status) {
        name = op_status.name;
        prevs = op_status.prevs;
        posts = op_status.posts;
        status = op_status.status;
    }

    void operator=(OperatorStatus&& op_status) {
        name = std::move(op_status.name);
        prevs = std::move(op_status.prevs);
        posts = std::move(op_status.posts);
        status = std::move(op_status.status);
    }

    bool isTensorRegistered(const std::string& tensor) {
        // Shared lock operator status mutex, since no object insert / erase takes place.
        std::shared_lock<std::shared_mutex>{status_mutex};
        auto p = status.find(tensor);
        return p != status.end();
    }

    TensorStatus& operator[](const std::string& tensor) {
        return status[tensor];
    }

    TensorStatus& at(const std::string& tensor) {
        return status.at(tensor);
    }

    const TensorStatus& at(const std::string& tensor) const {
        return status.at(tensor);
    }

    iterator begin() {return status.begin();}
    const_iterator cbegin() {return status.cbegin();}
    iterator end() {return status.end();}
    const_iterator cend() {return status.cend();}

};  // struct OperatorStatus

/**
 * MemoryStatuses
 * Provide a storage for memory status based on operator name.
 */
struct MemoryStatuses {
    using iterator = std::unordered_map<std::string, OperatorStatus>::iterator;
    using const_iterator = std::unordered_map<std::string, OperatorStatus>::const_iterator;

    std::unordered_map<std::string, OperatorStatus> status;
    std::vector<std::string> exec_order;
    std::string operators_entry, operators_end;

    std::shared_mutex status_mutex;

    MemoryStatuses() = default;
    MemoryStatuses(const MemoryStatuses&) = delete;
    MemoryStatuses(MemoryStatuses&& status) = delete;

    // void operator=(const MemoryStatuses&) = default;
    // void operator=(MemoryStatuses&& status) {
    //     status = move(status.status);
    // }

    void registerOperator(const OperatorStatus& opstatus) {
        std::unique_lock<std::shared_mutex>{status_mutex};
        auto p = status.find(opstatus.name);
        if (p != status.end()) throw status_error("Operator already registered.");

        exec_order.push_back(opstatus.name);

        status.insert(std::make_pair(opstatus.name, opstatus));
    }

    void completeComputationGraph() {

    }

    void submitMemoryStatus(const std::string& op, const std::string& tensor, MemoryDataStatusType data_status) {
        // Shared lock storage status mutex, since no object insert / erase takes place.
        std::shared_lock<std::shared_mutex>{status_mutex};
        
        auto& op_status = status.at(op);
        // Shared lock operator status mutex, since no object insert / erase takes place.
        std::shared_lock<std::shared_mutex>{op_status.status_mutex};

        auto& tensor_status = op_status.at(tensor);
        // Unique lock tensor status mutex, since status update.
        std::unique_lock<std::shared_mutex>{tensor_status.status_mutex};
        tensor_status.data_status = data_status;
    }

    MemoryDataStatusType getMemoryStatus(const std::string& op, const std::string& tensor) {
        // Shared lock storage status mutex, since no object insert / erase takes place.
        std::shared_lock<std::shared_mutex>{status_mutex};
        
        OperatorStatus& op_status = status.at(op);
        // Shared lock operator status mutex, since no object insert / erase takes place.
        std::shared_lock<std::shared_mutex>{op_status.status_mutex};

        auto& tensor_status = op_status.at(tensor);
        // Shared lock operator status mutex, since no object update takes place.
        std::shared_lock<std::shared_mutex>{tensor_status.status_mutex};
        return tensor_status.data_status;
    }

    bool isOperatorRegistered(const std::string& op) {
        std::shared_lock<std::shared_mutex>{status_mutex};
        auto p = status.find(op);
        return p != status.end();
    }

    bool isTensorRegistered(const std::string& op, const std::string& tensor) {
        // Shared lock operator status mutex, since no object insert / erase takes place.
        std::shared_lock<std::shared_mutex>{status_mutex};
        auto p = status.find(op);
        if (p == status.end()) return false;

        return p->second.isTensorRegistered(tensor);
    }

    OperatorStatus& operator[](const std::string& op) {
        return status[op];
    }

    OperatorStatus& at(const std::string& op) {
        return status.at(op);
    }

    const OperatorStatus& at(const std::string& op) const {
        return status.at(op);
    }

    void unregisterOperator(const std::string& op) {
        std::unique_lock<std::shared_mutex>{status_mutex};
        auto p = status.find(op);
        if (p == status.end()) throw status_error("Operator not registered.");

        status.erase(p);
    }

    iterator inline begin() {return status.begin();}
    const_iterator inline cbegin() {return status.cbegin();}
    iterator inline end() {return status.end();}
    const_iterator inline cend() {return status.cend();}

    ~MemoryStatuses() = default;

};  // struct MemoryStatus

}   // namespace mori