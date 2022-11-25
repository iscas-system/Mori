#pragma once

#include <string>
#include <vector>
#include <deque>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <shared_mutex>

#include "includes/exceptions.hpp"

namespace mori {
namespace status {

enum MemoryDataType {
    all, inout, weight, workspace, constant
};  // enum MemoryType

enum MemoryStatusType {
    none, empty, device, host, coexist, swapin, swapout
};  // enum MemoryDataStatusType

/**
 * MemoryStatus
 * Describe memory data section on the specific computing acclerating device. 
 * A tensor consists of a series of memory data sections.
 */
struct MemorySection {
    void* host_address = nullptr;
    void* device_address = nullptr;
    MemoryStatusType status = none;

    size_t size = 0;
};

struct TensorPres;

/**
 * TensorStatus
 * Status of a tensor which consists of a series of data sections.
 */
struct Tensor final {
private:
    friend struct TensorPres;

private:
    std::string name = "";

    // Tensor memory region consists of a series of data sections.
    // Key is the offset of the section, value is the corresponding memory status.
    // When the tensor is fulfilly on device, the sections should be continuous.
    std::map<size_t, MemorySection> sections;
    // Quick reference to current memory status, which is aggregated from section statuses.
    MemoryStatusType status = none;

    size_t size = 0;
    MemoryDataType type = all;

    // Indicating if the tensor should be considered in swapping.
    bool persistant = false;

    std::shared_mutex m;

public:
    Tensor() {
        sections.emplace(0, MemorySection());
    }
    Tensor(const std::string& _name): Tensor() {
        name = _name;
    }
    Tensor(const std::string& _name, size_t _size): name(_name), size(_size) {
        sections.emplace(0, MemorySection{nullptr, nullptr, none, _size});
    }
    Tensor(const std::string& _name, size_t _size, MemoryDataType _type): name(_name), size(_size), type(_type) {
        sections.emplace(0, MemorySection{nullptr, nullptr, none, _size});
    }
    Tensor(const Tensor& _status) {
        name = _status.name;
        sections = _status.sections;
        status = _status.status;
        size = _status.size;
        type = _status.type;
        persistant = _status.persistant;
    }
    Tensor(Tensor&& _status) {
        name = std::move(_status.name);
        sections = std::move(_status.sections);
        status = _status.status;
        size = _status.size;
        type = _status.type;
        persistant = _status.persistant;
    }

    Tensor& operator=(const Tensor& _status) {
        name = _status.name;
        sections = _status.sections;
        status = _status.status;
        size = _status.size;
        type = _status.type;
        persistant = _status.persistant;
        return *this;
    }
    Tensor& operator=(Tensor&& _status) {
        name = std::move(_status.name);
        sections = std::move(_status.sections);
        status = _status.status;
        size = _status.size;
        type = _status.type;
        persistant = _status.persistant;
        return *this;
    }

    inline void setName(const std::string& _name) { name = _name; }
    inline void setType(MemoryDataType _type) { type = _type; }
    inline void setPersistant(bool _persistant) { persistant = _persistant; }

    inline std::string getName() const noexcept { return name; }
    inline size_t getSize() const noexcept { return sections.at(0).size; }
    inline MemoryStatusType getStatus() const noexcept { return sections.at(0).status; }
    inline MemoryDataType getType() const noexcept { return type; }
    inline bool isPersistant() const noexcept { return persistant; }

    inline void*  getDevicePointer(int section) { return sections[section].device_address; }
    inline void*  getHostPointer(int section)   { return sections[section].host_address; }
    inline size_t getSectionSize(int section)   { return sections[section].size; }

    void acquire() { m.lock(); }

    void setAllocated(void* device_address) {
        // Since the allocation takes place in the beginning of the DL training procedure, there should be only one memory section.
        assert(sections.size() == 1);
        sections[0].device_address = device_address;
        sections[0].status = MemoryStatusType::empty;
    }
    void setAssigned() {
        switch(sections[0].status) {
            case empty:
            case coexist:
                sections[0].status = device;
            case device:
                break;
            default:
                throw status_exception("Accessing data not on device.");
        }
    }
    void setAcquired() {
        switch(sections[0].status) {
            case coexist:
            case device:
            case empty:
                break;
            default:
                throw status_exception("Acquiring data not on device.");
        }
    }
    void setAccessed() {
        setAssigned();
    }
    void setCopiedOut(void* host_address) {
        sections[0].host_address = host_address;
        switch(sections[0].status) {
            case coexist:
            case empty:
                break;
            case device:
                sections[0].status = MemoryStatusType::coexist;
                break;
            default:
                throw status_exception("No data on device while copying out memory data.");
        }
    }
    void setCopiedIn(void* device_address) {
        sections[0].device_address = device_address;
        switch(sections[0].status) {
            case host:
                sections[0].status = MemoryStatusType::coexist;
            case coexist:
                break;
            default:
                throw status_exception("No data on host while copying in memory data.");
                break;
        }
    }
    void setHostFreed() {
        sections[0].host_address = nullptr;
        switch (sections[0].status) {
            case coexist:
                sections[0].status = device;
                break;
            case host:
                sections[0].status = none;
                break;
            default:
                throw status_exception("No data on host while freeing host memory.");
        }
    }
    void setDeviceFreed() {
        sections[0].device_address = nullptr;
        switch (sections[0].status) {
            case coexist:
                sections[0].status = host;
                break;
            case empty:
            case device:
                sections[0].status = none;
                break;
            default:
                throw status_exception("No data on host while freeing host memory.");
        }
    }
    void setFreed() {
        sections[0].device_address = nullptr;
        switch (sections[0].status) {
            case coexist:
            case empty:
            case device:
            case host:
                sections[0].status = none;
                break;
            default:
                throw status_exception("No data on host and device while freeing memory.");
        }
    }

    void release() { m.unlock(); }

    void resize(size_t _size) {
        sections.clear();
        sections.emplace(0, MemorySection{nullptr, nullptr, none, _size});
    }

    ~Tensor() = default;

};  // struct TensorStatus

struct OperatorPres;

/**
 * Operator
 * Memory status of an operator.
 */
struct Operator final {
private:
    friend struct OperatorPres;

private:
    // Operator name.
    std::string name = "";
    // Prev and post dependencies.
    std::unordered_set<std::string> prevs, posts;  

    // Tensors consisting of this operator.
    std::unordered_set<std::string> tensors;

    std::shared_mutex m;

    // Indicate if the operator is a backward propagation operator.
    bool backward_propagation = false;

public:
    Operator() = default;
    Operator(const std::string& _name): name(_name) {}
    Operator(const Operator& _op) {
        name = _op.name;
        prevs = _op.prevs;
        posts = _op.posts;
        tensors = _op.tensors;
    }
    Operator(Operator&& _op) {
        name = std::move(_op.name);
        prevs = std::move(_op.prevs);
        posts = std::move(_op.posts);
        tensors = std::move(_op.tensors);
    }

    Operator& operator=(const Operator& _op) {
        name = _op.name;
        prevs = _op.prevs;
        posts = _op.posts;
        tensors = _op.tensors;
        return *this;
    }
    Operator& operator=(Operator&& _op) {
        name = std::move(_op.name);
        prevs = std::move(_op.prevs);
        posts = std::move(_op.posts);
        tensors = std::move(_op.tensors);
        return *this;
    }

    inline bool isBackwardPropagation() const noexcept { return backward_propagation; }
    inline void setBackwardPropagation(bool _backward_propagation) noexcept { backward_propagation = _backward_propagation; }

    inline void setPrev(const std::string& op) { prevs.insert(op); }
    template <typename T>
    inline void setPrevs(const T& ops) { prevs.insert(begin(ops), end(ops)); }
    inline void setPost(const std::string& op) { posts.insert(op); }
    template <typename T>
    inline void setPosts(const T& ops) { posts.insert(begin(ops), end(ops)); }

    bool isPrev(const std::string& op) const { return prevs.find(op) != prevs.end(); }
    bool isPost(const std::string& op) const { return posts.find(op) != posts.end(); }

    inline std::unordered_set<std::string> getPrevs() const noexcept { return prevs; }
    inline std::unordered_set<std::string> getPosts() const noexcept { return posts; }

    void removePrev(const std::string& op) {
        auto p = prevs.find(op);
        if (p == prevs.end()) return;
        prevs.erase(op);
    }

    void removePost(const std::string& op) {
        auto p = posts.find(op);
        if (p == posts.end()) return;
        posts.erase(op);
    }

    void clearPrevs() { prevs.clear(); }
    void clearPosts() { posts.clear(); }

    void setTensor(const std::string& tensor) { tensors.insert(tensor); }
    template <typename T>
    void setTensors(const T& _tensors) {
        for (auto &s : _tensors) tensors.insert(s);
    }

    bool  isTensorIncluded(const std::string& tensor) const { return tensors.find(tensor) != tensors.end(); }
    const std::unordered_set<std::string> getTensors() const noexcept { return tensors; }

    void removeTensor(const std::string& tensor) {
        auto p = tensors.find(tensor);
        if (p == tensors.end()) return;
        tensors.erase(tensor);
    }

    inline void clearTensors() { tensors.clear(); }

    inline void        setName(const std::string& _name) noexcept { name = _name; }
    inline std::string getName() const noexcept { return name; }

    ~Operator() = default;

};  // struct Operator

struct TensorPres final {
    Tensor& status;
    std::unique_lock<std::shared_mutex> l;

    TensorPres(Tensor& _status): status(_status) {
        l = std::unique_lock<std::shared_mutex>{status.m};
    }
    TensorPres(TensorPres&& _pres): status(_pres.status) {
        l = std::move(_pres.l);
    }

    inline void setAllocated(void* device_address) { status.setAllocated(device_address); }

    inline void setAssigned() { status.setAssigned(); }
    inline void setAcquired() { status.setAcquired(); }
    inline void setAccessed() { status.setAccessed(); }

    inline void setCopiedOut(void* host_address)  { status.setCopiedOut(host_address); }
    inline void setCopiedIn(void* device_address) { status.setCopiedIn(device_address); }
    inline void setHostFreed()   { status.setHostFreed(); }
    inline void setDeviceFreed() { status.setDeviceFreed(); }
    inline void setFreed()       { status.setFreed(); }

    inline std::string      getName()      const noexcept { return status.getName(); }
    inline size_t           getSize()      const noexcept { return status.getSize(); }
    inline MemoryStatusType getStatus()    const noexcept { return status.getStatus(); }
    inline MemoryDataType   getType()      const noexcept { return status.getType(); }
    inline bool             isPersistant() const noexcept { return status.isPersistant(); }

    inline void*  getDevicePointer(int section) { return status.getDevicePointer(section); }
    inline void*  getHostPointer(int section)   { return status.getHostPointer(section); }
    inline size_t getSectionSize(int section)   { return status.getSectionSize(section); }

    inline void release() { l.unlock(); }

    ~TensorPres() = default;
};  // struct TensorPres

struct OperatorPres final {
    Operator& status;
    // Operator is read-only during DL processing.
    // std::shared_lock<std::shared_mutex> l;

    OperatorPres(Operator& _status): status(_status) {
        // l = std::unique_lock<std::shared_mutex>(status.m);
    }
    OperatorPres(OperatorPres&& _pres): status(_pres.status) {
        // l = std::move(_pres.l);
    }

    inline std::string                     getName()    const noexcept { return status.getName(); }
    inline std::unordered_set<std::string> getPrevs()   const noexcept { return status.getPrevs(); }
    inline std::unordered_set<std::string> getPosts()   const noexcept { return status.getPosts(); }
    inline std::unordered_set<std::string> getTensors() const noexcept { return status.getTensors(); }

    inline bool isBackwardPropagation() const noexcept { return status.isBackwardPropagation(); }
};  // struct OperatorPres

/**
 * MemoryStatus
 * Storage of tensor status and corresponding operator status.
 */
struct MemoryStatus {
private:
    std::unordered_map<std::string, Tensor> tensor_statuses;
    std::unordered_map<std::string, Operator> operator_statuses;
    std::vector<std::string> execution_order;
    std::string operator_entry = "";

    // Protect the status map.
    std::shared_mutex tm, om;

public:
    MemoryStatus() = default;
    MemoryStatus(const MemoryStatus& _status) {
        tensor_statuses = _status.tensor_statuses;
        operator_statuses = _status.operator_statuses;
    }
    MemoryStatus(MemoryStatus&& _status) {
        std::unique_lock<std::shared_mutex>{_status.tm};
        std::unique_lock<std::shared_mutex>{_status.om};
        tensor_statuses = std::move(_status.tensor_statuses);
        operator_statuses = std::move(_status.operator_statuses);
    }

    MemoryStatus& operator=(const MemoryStatus& _status) {
        tensor_statuses = _status.tensor_statuses;
        operator_statuses = _status.operator_statuses;
        return *this;
    }
    MemoryStatus& operator=(MemoryStatus&& _status) {
        std::unique_lock<std::shared_mutex>{_status.tm};
        std::unique_lock<std::shared_mutex>{_status.om};
        tensor_statuses = std::move(_status.tensor_statuses);
        operator_statuses = std::move(_status.operator_statuses);
        return *this;
    }

    /**
     * registerTensor
     * Register a tensor to the storage.
     * Only can be invoked when tensor status storage not inited.
     * @param status tensorStatus
     */
    void registerTensor(const Tensor& status) {
        std::unique_lock<std::shared_mutex>{tm};

        auto p = tensor_statuses.find(status.getName());
        if (p != tensor_statuses.end()) throw status_exception("Tensor already registered.");
        tensor_statuses.emplace(status.getName(), status);
    }

    /**
     * registerTensor
     * Register a tensor to the storage, whose status information is empty.
     * Only can be invoked when tensor status storage not inited.
     * @param tensor tensor name
     */
    void registerTensor(const std::string& tensor) {
        std::unique_lock<std::shared_mutex>{tm};

        auto p = tensor_statuses.find(tensor);
        if (p != tensor_statuses.end()) throw status_exception("Tensor already registered.");
        tensor_statuses.emplace(tensor, Tensor(tensor));
    }

    /**
     * registerOperator
     * Register an operator to the storage.
     * An operator should be always registered later than its tensors,
     * since the status storage would check the validity of the tensors included in the operator.
     * @param status operator status
     */
    void registerOperator(const Operator& status) {
        std::unique_lock<std::shared_mutex>{om};

        auto p = operator_statuses.find(status.getName());
        if (p != operator_statuses.end()) throw status_exception("Operator already registered.");

        for (auto &s : status.getTensors())
            if (tensor_statuses.find(s) == tensor_statuses.end())
                // Tensor not registered.
                throw status_exception("Specified tensor not registered.");

        for (auto &s : status.getPrevs())
            if (operator_statuses.find(s) == operator_statuses.end())
                // Operator not registered.
                throw status_exception("Specified prev operator not registered.");

        operator_statuses.emplace(status.getName(), status);
        execution_order.push_back(status.getName());
    }

    void setEntry(const std::string& _op) {
        auto p = operator_statuses.find(_op);
        if (p == operator_statuses.end()) throw status_exception("Operator not registered.");
        operator_entry = _op;
    }

    inline std::string getEntry() const noexcept { return operator_entry; }

    inline std::vector<std::string> getExecutionOrder() const noexcept { return execution_order; }
    template<typename T>
    inline void setExecutionOrder(const T& _execution_order) { 
        std::vector<std::string> vect(begin(_execution_order), end(_execution_order));
        execution_order.swap(vect);
    }

    /**
     * isTensorRegistered
     * Check if the tensor is registered to the storage.
     * @param tensor tensor name
     * @return if the tensor is registered
     */
    bool isTensorRegistered(const std::string& tensor) const {
        return tensor_statuses.find(tensor) != tensor_statuses.end();
    }

    /**
     * isOperatorRegistered
     * Check if the operator is registered to the storage.
     * @param op operator name
     * @return if the operator is registered
     */
    // bool isOperatorRegistered(const std::string& op) const {
    //     return operator_statuses.find(op) != operator_statuses.end();
    // }

    /**
     * reference
     * Reference the tensor
     * @param tensor tensor name
     * @return reference to the specific tensor
     */
    TensorPres referenceTensor(const std::string& tensor) {
        auto p = tensor_statuses.find(tensor);
        if (p == tensor_statuses.end()) throw status_exception("Tensor not registered.");
        return TensorPres(p->second);
    }

    OperatorPres referenceOperator(const std::string& op) {
        auto p = operator_statuses.find(op);
        if (p == operator_statuses.end()) throw status_exception("Operator not registered.");
        return OperatorPres(p->second);
    }

    void unregisterOperator(const std::string& op) {
        std::unique_lock<std::shared_mutex>{om};
        
        auto p = operator_statuses.find(op);
        if (p == operator_statuses.end()) throw status_exception("Operator not registered.");
        operator_statuses.erase(p);

        auto q = std::find(execution_order.begin(), execution_order.end(), op);
        assert(q != execution_order.end());
        execution_order.erase(q);
    }

    /**
     * unregisterTensor
     * Unregister a tensor from the storage.
     * Only can be invoked when tensor status storage not inited.
     * @param tensor tensor name
    */
    void unregisterTensor(const std::string& tensor) {
        std::unique_lock<std::shared_mutex>{tm};
        
        auto p = tensor_statuses.find(tensor);
        if (p == tensor_statuses.end()) throw status_exception("Tensor not registered.");
        tensor_statuses.erase(p);
    }

    /**
     * @brief Clear all status information.
    */
    void clear() {
        tensor_statuses.clear();
        operator_statuses.clear();
    }

    ~MemoryStatus() = default;

};  // struct MemoryStatus

namespace util {

static std::string get_tensor_type_str(MemoryDataType type) {
    switch (type) {
        case MemoryDataType::all:
            return "all";
        case MemoryDataType::constant:
            return "constant";
        case MemoryDataType::inout:
            return "inout";
        case MemoryDataType::weight:
            return "weight";
        case MemoryDataType::workspace:
            return "workspace";
    }

    assert(0);
    return "";
}

}   // namespace util

}   // namespace status

using TensorPres   = status::TensorPres;
using MemoryStatus = status::MemoryStatus;

}   // namespace mori