#pragma once

#include <unordered_map>
#include <vector>
#include <string>
#include <shared_mutex>

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
    std::string name;
    void* host_address;
    void* device_address;
    size_t size;
    MemoryType type;

    MemoryDataStatusType data_status;

    // If the tensor status is locked. Only the mutex holder can modify the status.
    // Specifically, update memory data status.
    std::shared_mutex status_mutex;

    TensorStatus(): name(""), size(0), type(MemoryType::all), host_address(nullptr), device_address(nullptr), data_status(MemoryDataStatusType::none) {}

    TensorStatus(const std::string& _name, size_t _size, MemoryType _type): name(_name), size(_size), type(_type), host_address(nullptr), device_address(nullptr), data_status(MemoryDataStatusType::none) {}

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
    // These three variables be regarded as const variables.
    std::string name;
    // The prev and post operators in the graph.
    std::vector<std::string> prevs, posts;

    std::unordered_map<std::string, TensorStatus> tensor_status;

    std::shared_mutex status_mutex;

    OperatorStatus(): name(""), prevs({}), posts({}), tensor_status({}) {}

    OperatorStatus(const std::string& _name, const std::vector<std::string>& _prevs, const std::vector<std::string>& _posts, const std::unordered_map<std::string, TensorStatus>& _tensor_status): name(_name), prevs(_prevs), posts(_posts), tensor_status(_tensor_status) {}

    OperatorStatus(const OperatorStatus& status) {
        name = status.name;
        prevs = status.prevs;
        posts = status.posts;
        tensor_status = status.tensor_status;
    }

    OperatorStatus(OperatorStatus&& status) {
        name = std::move(status.name);
        prevs = std::move(status.prevs);
        posts = std::move(status.posts);
        tensor_status = std::move(status.tensor_status);
    }

    std::vector<std::string> getPrevs() { return prevs; }
    std::vector<std::string> getPosts() { return posts; }

    void operator=(const OperatorStatus& status) {
        name = status.name;
        prevs = status.prevs;
        posts = status.posts;
        tensor_status = status.tensor_status;
    }

    void operator=(OperatorStatus&& status) {
        name = std::move(status.name);
        prevs = std::move(status.prevs);
        posts = std::move(status.posts);
        tensor_status = std::move(status.tensor_status);
    }

    TensorStatus& operator[](const std::string& tensor) {
        return tensor_status[tensor];
    }

    TensorStatus& at(const std::string& tensor) {
        return tensor_status.at(tensor);
    }

    const TensorStatus& at(const std::string& tensor) const {
        return tensor_status.at(tensor);
    }

};  // struct OperatorStatus

/**
 * MemoryStatuses
 * Provide a storage for memory status based on operator name.
 */
struct MemoryStatuses {
    std::unordered_map<std::string, OperatorStatus> operator_status;
    std::vector<std::string> exec_order;
    std::string operators_entry, operators_end;

    std::shared_mutex status_mutex;

    MemoryStatuses() = default;
    MemoryStatuses(const MemoryStatuses&) = delete;
    MemoryStatuses(MemoryStatuses&& status) = delete;

    // void operator=(const MemoryStatuses&) = default;
    // void operator=(MemoryStatuses&& status) {
    //     operator_status = move(status.operator_status);
    // }

    void registerOperator(const OperatorStatus& opstatus) {
        std::unique_lock<std::shared_mutex>{status_mutex};
        auto p = operator_status.find(opstatus.name);
        if (p != operator_status.end()) throw std::exception();

        exec_order.push_back(opstatus.name);

        operator_status.insert(std::make_pair(opstatus.name, opstatus));
    }

    void completeComputationGraph() {

    }

    void submitMemoryStatus(const std::string& op, const std::string& tensor, MemoryDataStatusType data_status) {
        // Shared lock storage status mutex, since no object insert / erase takes place.
        std::shared_lock<std::shared_mutex>{status_mutex};
        
        auto& op_status = operator_status.at(op);
        // Shared lock operator status mutex, since no object insert / erase takes place.
        std::shared_lock(op_status.status_mutex);

        auto& tensor_status = op_status.at(tensor);
        // Unique lock tensor status mutex, since status update.
        std::unique_lock(tensor_status.status_mutex);
        tensor_status.data_status = data_status;
    }

    MemoryDataStatusType getMemoryStatus(const std::string& op, const std::string& tensor) {
        // Shared lock storage status mutex, since no object insert / erase takes place.
        std::shared_lock<std::shared_mutex>{status_mutex};
        
        OperatorStatus& op_status = operator_status.at(op);
        // Shared lock operator status mutex, since no object insert / erase takes place.
        std::shared_lock<std::shared_mutex>{op_status.status_mutex};

        auto& tensor_status = op_status.at(tensor);
        // Shared lock operator status mutex, since no object update takes place.
        std::shared_lock<std::shared_mutex>{tensor_status.status_mutex};
        return tensor_status.data_status;
    }

    bool isOperatorRegistered(const std::string& op) {
        std::shared_lock<std::shared_mutex>{status_mutex};
        auto p = operator_status.find(op);
        return p != operator_status.end();
    }

    bool isTensorRegistered(const std::string& op, const std::string& tensor) {
        // Shared lock operator status mutex, since no object insert / erase takes place.
        std::shared_lock<std::shared_mutex>{status_mutex};
        auto p = operator_status.find(op);
        if (p != operator_status.end()) return false;

        // Shared lock operator status mutex, since no object insert / erase takes place.
        std::shared_lock<std::shared_mutex>{p->second.status_mutex};
        auto q = p->second.tensor_status.find(tensor);
        return q != p->second.tensor_status.end();
    }

    OperatorStatus& operator[](const std::string& op) {
        return operator_status[op];
    }

    OperatorStatus& at(const std::string& op) {
        return operator_status.at(op);
    }

    const OperatorStatus& at(const std::string& op) const {
        return operator_status.at(op);
    }

    void unregisterOperator(const std::string& op) {
        std::unique_lock<std::shared_mutex>{status_mutex};
        auto p = operator_status.find(op);
        if (p == operator_status.end()) throw std::exception();

        operator_status.erase(p);
    }

    ~MemoryStatuses() = default;

};  // struct MemoryStatus

}   // namespace mori