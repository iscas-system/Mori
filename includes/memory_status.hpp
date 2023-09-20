#pragma once

#include <string>
#include <vector>
#include <deque>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <shared_mutex>
#include <cassert>

#include "includes/memory_info.hpp"
#include "includes/exceptions/status_exceptions.hpp"
#include "includes/exceptions/memory_status_exceptions.hpp"

namespace mori {
namespace status {

enum struct MemoryDataType {
    all, inout, weight, workspace, constant
};  // enum struct MemoryType

enum struct MemoryStatusType {
    none, empty, device, host, coexist, swapin, swapout
};  // enum struct MemoryDataStatusType

struct Tensor;

/**
 * Describe memory data section on the specific computing acclerating device. 
 * A tensor consists of a series of memory data sections.
 */
struct MemorySection final {
private:
    friend struct Tensor;

private:
    MemorySection* prev_sect = nullptr;
    MemorySection* post_sect = nullptr;

public:
    size_t offset = 0;
    size_t size = 0;

    void* host_address   = nullptr;
    void* device_address = nullptr;

    MemoryStatusType status = MemoryStatusType::none;

public:
    MemorySection() = default;
    MemorySection(size_t _offset, size_t _size, void* _host_address, void* _device_address, MemoryStatusType _status): offset(_offset), size(_size), host_address(_host_address), device_address(_device_address), status(_status) {}
    MemorySection(const MemorySection&) = default;
    MemorySection(MemorySection&&) = default;
    MemorySection& operator=(const MemorySection&) = default;
    MemorySection& operator=(MemorySection&&) = default;

    inline bool hasPrev() const { return prev_sect != nullptr; }
    inline MemorySection* prev() { return prev_sect; }
    inline const MemorySection* prev() const { return prev_sect; }

    inline bool hasPost() const { return post_sect != nullptr; }
    inline MemorySection* next() { return post_sect; }
    inline const MemorySection* next() const { return post_sect; }

    ~MemorySection() = default;
};

/**
 * Fragment is a simplified MemorySection
 */
struct Fragment final {
    size_t size = 0;
    void* address = nullptr;
    MemoryStatusType status = MemoryStatusType::none;

    Fragment() = default;
    Fragment(size_t _size, void* _address, MemoryStatusType _status = MemoryStatusType::none): size(_size), address(_address), status(_status) {} 
};  // struct Fragment

struct TensorPres;
struct MemoryStatus;

/**
 * TensorStatus
 * Status of a tensor which consists of a series of data sections.
 */
struct Tensor final {
private:
    friend struct TensorPres;
    friend struct MemoryStatus;

private:
    std::string name = "";

    // Tensor memory region consists of a series of data sections.
    // Key is the offset of the section, value is the corresponding memory status.
    // When the tensor is fulfilly on device, the sections should be continuous.
    std::map<size_t, MemorySection> sections;
    Fragment fragment;

    // Tensor size
    size_t size = 0;
    // Remaining tensor size in memory
    size_t device_size = 0;
    size_t host_size = 0;
    MemoryDataType type = MemoryDataType::all;

    // Indicating if the tensor should be considered in swapping.
    bool persistent = false;
    bool transient = false;

    std::string op = "";

public:
    Tensor() {
        sections.emplace(0, MemorySection());
    }
    Tensor(const std::string& _name): Tensor() {
        name = _name;
    }
    Tensor(const std::string& _name, size_t _size): name(_name), size(_size), device_size(_size), host_size(0) {
        sections.emplace(0, MemorySection{0, _size, nullptr, nullptr, MemoryStatusType::none});
        // if (_size < 1048576 * 4) transient = true; 
    }
    Tensor(const std::string& _name, size_t _size, MemoryDataType _type): Tensor(_name, _size) {
        type = _type;
        if (_type == MemoryDataType::constant || _type == MemoryDataType::weight) persistent = true;
        if (_type == MemoryDataType::workspace) transient = true;
    }
    Tensor(const Tensor& _status) = default;
    Tensor(Tensor&& _status) = default;

    Tensor& operator=(const Tensor& _status) = default;
    Tensor& operator=(Tensor&& _status) = default;

    inline void setName(const std::string& _name) { name = _name; }
    inline void setType(MemoryDataType _type) { type = _type; }
    inline void setSize(size_t _size) { size = _size; sections[0].size = _size; }
    inline void setPersistent(bool _persistent) { persistent = _persistent; }
    inline void setTransient(bool _transient) { transient = _transient; }

    inline std::string      getName()          const noexcept { return name; }
    inline std::string      getOperatorName()  const noexcept { return op; }
    inline size_t           getSize()          const noexcept { return size; }
    inline size_t           getDeviceSize()    const noexcept { return device_size; }
    inline size_t           getHostSize()      const noexcept { return host_size; }
    inline MemoryDataType   getType()          const noexcept { return type; }
    inline bool             isPersistent()     const noexcept { return persistent; }
    inline bool             isTransient()      const noexcept { return transient; }

    inline const MemorySection& getSection(size_t offset) const { return sections.at(offset); }
    inline int getSectionCount() const noexcept { return sections.size(); }
    inline const MemorySection& getFirstSection() const { return sections.begin()->second; }
    inline const MemorySection& getLastSection() const { return sections.rbegin()->second; }
    
    std::vector<size_t> getSections() const {
        std::vector<size_t> re;
        for (auto &x : sections) re.push_back(x.first);
        return re;
    }

    inline bool isSectionExist(size_t offset) const noexcept { return sections.find(offset) != sections.end(); }

    /**
     * If tensor has data located on device.
     */
    bool isDeviceLocated() const noexcept {
        for (auto &x : sections) {
            if (x.second.status == MemoryStatusType::empty || x.second.status == MemoryStatusType::device || x.second.status == MemoryStatusType::coexist) return true;
        }
        return false;
    }
    /**
     * If tensor has all data located on device.
     */
    bool isDeviceAllLocated() const noexcept {
        for (auto &x : sections) {
            if (x.second.status == MemoryStatusType::none || x.second.status == MemoryStatusType::host) return false;
        }
        return true;
    }
    /**
     * If tensor has data located on host.
     */
    bool isHostLocated() const noexcept {
        for (auto &x : sections) {
            if (x.second.status == MemoryStatusType::host || x.second.status == MemoryStatusType::coexist) return true;
        }
        return false;
    }
    /**
     * If tensor has all data located on host.
     */
    bool isHostAllLocated() const noexcept {
        for (auto &x : sections) {
            if (x.second.status == MemoryStatusType::none || x.second.status == MemoryStatusType::empty || x.second.status == MemoryStatusType::device) return false;
        }
        return true;
    }
    /**
     * If tensor has data located on host or device.
     */
    bool isMemoryLocated() const noexcept {
        for (auto &x : sections) {
            if (x.second.status != MemoryStatusType::none) return true;
        }
        return false;
    }

    void split(size_t offset, size_t size) {
        assert(size != 0);
        MemorySection& memory_section = sections.at(offset);

        if (memory_section.size < size) {
            throw memory_section_invalid("Sectioning size larger than section size.");
        }
        if (memory_section.size == size) return;

        MemorySection new_section;
        new_section.offset         = memory_section.offset + size;
        new_section.size           = memory_section.size - size;
        if (memory_section.host_address   != nullptr) new_section.host_address   = (uint8_t*)memory_section.host_address   + size;
        if (memory_section.device_address != nullptr) new_section.device_address = (uint8_t*)memory_section.device_address + size;
        new_section.status         = memory_section.status;
        new_section.prev_sect      = &memory_section;
        new_section.post_sect      = memory_section.post_sect;

        memory_section.size        = size;

        sections.emplace(offset + size, new_section);
        memory_section.post_sect = &(sections.at(offset + size));
    }

    inline bool isMergeable(size_t offset) {
        auto pb = sections.find(offset);
        if (pb == sections.end()) throw memory_section_invalid("Invalid section offset.");
        MemorySection& memory_section = pb->second;
        auto pe = ++pb;

        if (pe == sections.end()) return false;
        if (pe->second.status != memory_section.status) return false;
        if (memory_section.status == MemoryStatusType::host || memory_section.status == MemoryStatusType::coexist) return false;

        assert((uint8_t*)memory_section.device_address + memory_section.size == pe->second.device_address);
        return true;
    }

    MemorySection& merge(size_t offset = 0) {
        if (!isMergeable(offset)) throw memory_section_invalid("Invalid section merging.");

        MemorySection& memory_section = sections.at(offset);
        MemorySection& post_section = *memory_section.post_sect;
        size_t post_offset = post_section.offset;

        memory_section.size += post_section.size;
        memory_section.post_sect = post_section.post_sect;
        if (memory_section.post_sect != nullptr) memory_section.post_sect->prev_sect = &memory_section;
        sections.erase(post_offset);

        return memory_section;
    }

    void setReshaped(size_t _size) {
        // Since the allocation takes place in the beginning of the application procedure, there should be only one memory section.
        if (sections.size() != 1) throw status_exception("Set reshaped for sectioned tensor.");
        assert(sections.begin()->first == 0);
        assert(sections.begin()->second.offset == 0);
        size = _size;
        sections.begin()->second.size = size;
        if (sections[0].status != MemoryStatusType::none) throw status_exception("Set reshaped for allocated tensor.");
        device_size = size;
    }
    void setAllocated(void* device_address) {
        // Since the allocation takes place in the beginning of the application procedure, there should be only one memory section.
        if (sections.size() != 1) throw status_exception("Set allocated for sectioned tensor.");
        assert(sections.begin()->first == 0);
        assert(sections.begin()->second.offset == 0);
        assert(sections.begin()->second.size == size);
        if (sections[0].status != MemoryStatusType::none) throw status_exception("Set allocated for allocated tensor.");
        sections[0].device_address = device_address;
        sections[0].status = MemoryStatusType::empty;
        device_size = size;
    }
    void setAssigned() {
        for (auto &x : sections) {
            switch(x.second.status) {
                case MemoryStatusType::empty:
                    if (size != 0) x.second.status = MemoryStatusType::device;
                case MemoryStatusType::device:
                    break;
                case MemoryStatusType::coexist:
                    throw status_exception("Accessing data not released on host.");
                default:
                    throw status_exception("Accessing data not on device.");
            }
        }
    }
    void setAcquired() {
        for (auto &x : sections) {
            switch(x.second.status) {
                case MemoryStatusType::coexist:
                case MemoryStatusType::device:
                case MemoryStatusType::empty:
                    break;
                default:
                    throw status_exception("Acquiring data not on device.");
            }
        }
    }
    void setAccessed() {
        setAssigned();
    }
    void setCopiedOut(size_t offset, void* host_address) {
        MemorySection& memory_section = sections.at(offset);
        memory_section.host_address = host_address;
        switch(memory_section.status) {
            case MemoryStatusType::device:
                memory_section.status = MemoryStatusType::coexist;
                host_size += memory_section.size;
            case MemoryStatusType::coexist:
            case MemoryStatusType::empty:
                break;
            default:    // none host
                throw status_exception("No data on device while copying out memory data.");
        }

    }
    void setCopiedOut(void* host_address) {
        if (sections.size() != 1) throw status_exception("Set copied out for sectioned tensor.");
        assert(sections.begin()->first == 0);
        assert(sections.begin()->second.offset == 0);
        assert(sections.begin()->second.size == size);
        setCopiedOut(0, host_address);
    }
    void setCopiedIn(size_t offset, void* device_address) {
        MemorySection& memory_section = sections.at(offset);
        memory_section.device_address = device_address;
        switch(memory_section.status) {
            case MemoryStatusType::none:
                memory_section.status = MemoryStatusType::empty;
                break;
            case MemoryStatusType::host:
                memory_section.status = MemoryStatusType::coexist;
            case MemoryStatusType::coexist:
                break;
            default:    // device empty
                throw status_exception("No data on host while copying in memory data.");
                break;
        }
        device_size += memory_section.size;
    }
    void setCopiedIn(void* device_address) {
        if (sections.size() != 1) throw status_exception("Set copied in for sectioned tensor.");
        assert(sections.begin()->first == 0);
        assert(sections.begin()->second.offset == 0);
        assert(sections.begin()->second.size == size);
        setCopiedIn(0, device_address);
    }
    void setMoved(size_t offset, void* dst_address) {
        MemorySection& memory_section = sections.at(offset);
        memory_section.device_address = dst_address;
        switch(memory_section.status) {
            case MemoryStatusType::empty:
            case MemoryStatusType::device:
            case MemoryStatusType::coexist:
                break;
            default:    // device none
                throw status_exception("No data on device while moving memory data.");
                break;
        }
    }
    void setHostFreed(size_t offset) {
        MemorySection& memory_section = sections.at(offset);
        switch (memory_section.status) {
            case MemoryStatusType::coexist:
                memory_section.status = MemoryStatusType::device;
                break;
            case MemoryStatusType::host:
                memory_section.status = MemoryStatusType::none;
                break;
            default:    // none empty device
                throw status_exception("No data on host while freeing host memory.");
        }
        host_size -= memory_section.size;
        
    }
    void setDeviceFreed(size_t offset) {
        MemorySection& memory_section = sections.at(offset);
        switch (memory_section.status) {
            case MemoryStatusType::coexist:
                memory_section.status = MemoryStatusType::host;
                break;
            case MemoryStatusType::empty:
            case MemoryStatusType::device:
                memory_section.status = MemoryStatusType::none;
                break;
            default:    // none host
                throw status_exception("No data on host while freeing host memory.");
        }

        device_size -= memory_section.size;
    }
    void setFreed(size_t offset) {
        MemorySection& memory_section = sections.at(offset);
        switch (memory_section.status) {
            case MemoryStatusType::coexist:
                device_size -= memory_section.size;
                host_size -= memory_section.size;
                break;
            case MemoryStatusType::empty:
            case MemoryStatusType::device:
                device_size -= memory_section.size;
                break;
            case MemoryStatusType::host:
                host_size -= memory_section.size;
                break;
            default:    // none
                throw status_exception("No data on host and device while freeing memory.");
        }
        memory_section.status = MemoryStatusType::none;
    }

    inline bool hasFragment() const noexcept { return fragment.size != 0; }
    inline const Fragment& getFragment() const noexcept { return fragment; }
    inline void setFragment(size_t _size) {
        if (fragment.status != MemoryStatusType::none) throw status_exception("Setting existed fragment size.");
        fragment.size = _size; 
    }

    void setFragmentPlaced(void* address) {
        if (fragment.status != MemoryStatusType::none) throw status_exception("Placing existed fragment.");
        fragment.status = MemoryStatusType::empty;
        fragment.address = address;
    }

    inline void setFragmentPlaced() { setFragmentPlaced((uint8_t*)(sections.at(0).device_address) + size); }

    void setFragmentRemoved() {
        if (fragment.status == MemoryStatusType::none) throw status_exception("Removing non-exist fragment.");
        fragment.status = MemoryStatusType::none;
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

    // std::shared_mutex m;

    // Indicate if the operator is a backward propagation operator.
    bool backward_propagation = false;

public:
    Operator() = default;
    Operator(const std::string& _name): name(_name) {}
    Operator(const Operator& _op) = default;
    Operator(Operator&& _op) = default;
    Operator& operator=(const Operator& _op) = default;
    Operator& operator=(Operator&& _op) = default;

    inline bool isBackwardPropagation() const noexcept { return backward_propagation; }
    inline void setBackwardPropagation(bool _backward_propagation) noexcept { backward_propagation = _backward_propagation; }

    inline void setPrev(const std::string& op) { prevs.insert(op); }
    template <typename T>
    inline void setPrevs(const T& ops) { prevs.insert(begin(ops), end(ops)); }
    inline void setPost(const std::string& op) { posts.insert(op); }
    template <typename T>
    inline void setPosts(const T& ops) { posts.insert(begin(ops), end(ops)); }

    inline bool isPrev(const std::string& op) const { return prevs.find(op) != prevs.end(); }
    inline bool isPost(const std::string& op) const { return posts.find(op) != posts.end(); }

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

    inline void clearPrevs() noexcept { prevs.clear(); }
    inline void clearPosts() noexcept { posts.clear(); }

    inline void setTensor(const std::string& tensor) { tensors.insert(tensor); }
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

    inline void clearTensors() noexcept { tensors.clear(); }

    inline void        setName(const std::string& _name) noexcept { name = _name; }
    inline std::string getName() const noexcept { return name; }

    ~Operator() = default;

};  // struct Operator

struct MemoryStatus;

struct TensorPres final {
private:
    friend struct MemoryStatus;

public:
    using target_type = Tensor;

private:
    Tensor& status;
    std::unique_lock<std::shared_mutex> l;

    TensorPres(Tensor& _status, std::shared_mutex& m): status(_status) {
        l = std::unique_lock<std::shared_mutex>{m, std::try_to_lock};
    }

public:
    TensorPres(TensorPres&& _pres): status(_pres.status) {
        l = std::move(_pres.l);
    }

    inline bool isReferenced() const noexcept { return l.owns_lock(); }
    inline void reference() { l.lock(); }

public:
    inline void setReshaped(size_t size) { status.setReshaped(size); }
    inline void setAllocated(void* device_address) { status.setAllocated(device_address); }

    inline void setAssigned() { status.setAssigned(); }
    inline void setAcquired() { status.setAcquired(); }
    inline void setAccessed() { status.setAccessed(); }

    inline void setCopiedOut(size_t offset, void* host_address) { status.setCopiedOut(offset, host_address); }
    inline void setCopiedOut(void* host_address) { status.setCopiedOut(host_address); }
    inline void setCopiedIn(size_t offset, void* device_address) { status.setCopiedIn(offset, device_address); }
    inline void setCopiedIn(void* device_address) { status.setCopiedIn(device_address); }
    inline void setMoved(size_t offset, void* dst_address) { status.setMoved(offset, dst_address); }
    inline void setHostFreed(size_t offset = 0)   { status.setHostFreed(offset); }
    inline void setDeviceFreed(size_t offset = 0) { status.setDeviceFreed(offset); }
    inline void setFreed(size_t offset = 0)       { status.setFreed(offset); }

    inline std::string      getName()           const noexcept { return status.getName(); }
    inline std::string      getOperatorName()   const noexcept { return status.getOperatorName(); }
    inline size_t           getSize()           const noexcept { return status.getSize(); }
    inline size_t           getDeviceSize()     const noexcept { return status.getDeviceSize(); }
    inline size_t           getHostSize()       const noexcept { return status.getHostSize(); }
    inline MemoryDataType   getType()           const noexcept { return status.getType(); }
    inline bool             isPersistent()      const noexcept { return status.isPersistent(); }
    inline bool             isTransient()       const noexcept { return status.isTransient(); }

    inline const MemorySection& getSection(size_t offset) const { return status.getSection(offset); }
    inline int getSectionCount() const noexcept { return status.getSectionCount(); }
    inline const MemorySection& getFirstSection() const { return status.getFirstSection(); }
    inline const MemorySection& getLastSection() const { return status.getLastSection(); }

    inline bool isSectionExist(size_t offset) const noexcept { return status.isSectionExist(offset); }

    inline bool isDeviceLocated()    const noexcept { return status.isDeviceLocated(); }
    inline bool isDeviceAllLocated() const noexcept { return status.isDeviceAllLocated(); }
    inline bool isHostLocated()      const noexcept { return status.isHostLocated(); }
    inline bool isHostAllLocated()   const noexcept { return status.isHostAllLocated(); }
    inline bool isMemoryLocated()    const noexcept { return status.isMemoryLocated(); }

    inline Tensor& get() noexcept { return status; }

    inline void split(size_t offset, size_t size) { status.split(offset, size); }
    inline bool isMergeable(size_t offset) const { return status.isMergeable(offset); }
    inline MemorySection& merge(size_t offset = 0) { return status.merge(offset); }

    inline bool hasFragment() const noexcept { return status.hasFragment(); }
    inline const Fragment& getFragment() const noexcept { return status.getFragment(); }
    inline void setFragment(size_t _size) { status.setFragment(_size); }

    inline void setFragmentPlaced(void* address) { status.setFragmentPlaced(address); }
    inline void setFragmentPlaced()  { status.setFragmentPlaced(); }
    inline void setFragmentRemoved() { status.setFragmentRemoved(); }

    inline void release() { l.unlock(); }

    ~TensorPres() { if (l.owns_lock()) release(); }
};  // struct TensorPres

struct OperatorPres final {
private:
    friend struct MemoryStatus;

public:
    using target_type = Operator;

private:
    Operator& status;
    // Operator is read-only during DL processing.
    std::unique_lock<std::shared_mutex> l;

    OperatorPres(Operator& _status, std::shared_mutex& m): status(_status) {
        l = std::unique_lock<std::shared_mutex>(m, std::try_to_lock);
    }

public:
    OperatorPres(OperatorPres&& _pres): status(_pres.status) {
        l = std::move(_pres.l);
    }

    inline bool isReferenced() const noexcept { return l.owns_lock(); }
    inline void reference() { l.lock(); }

public:
    inline std::string                     getName()    const noexcept { return status.getName(); }
    inline std::unordered_set<std::string> getPrevs()   const noexcept { return status.getPrevs(); }
    inline std::unordered_set<std::string> getPosts()   const noexcept { return status.getPosts(); }
    inline std::unordered_set<std::string> getTensors() const noexcept { return status.getTensors(); }

    inline bool isBackwardPropagation() const noexcept { return status.isBackwardPropagation(); }

    inline Operator& get() noexcept { return status; }
};  // struct OperatorPres

/**
 * MemoryStatus
 * Storage of tensor status and corresponding operator status.
 */
struct MemoryStatus final {
private:
    template <typename T>
    struct View final {
        T pres;
        
        View(typename T::target_type& _target, std::shared_mutex& m): pres(_target, m) {}

        inline bool isReferenced() { return pres.isReferenced(); }
        inline T&& reference() { 
            if (!pres.isReferenced())
                pres.reference();
            return std::move(pres);
        }
    };  // inner struct View

    template <typename T>
    struct Hold {
        T target;
        std::shared_mutex m;
        Hold(const T& _target): target(_target) {}
        Hold(T&& _target): target(_target) {}
        Hold(const Hold& _hold): target(_hold.target) {}
        Hold(Hold&&) = default;

        Hold& operator=(const Hold& _hold) { target = _hold.target; return *this; }
        Hold& operator=(Hold&& _hold) = default;
    };  // inner struct Hold

public:
    using TensorView   = View<TensorPres>;
    using OperatorView = View<OperatorPres>;

private:
    std::unordered_map<std::string, Hold<Tensor>> tensor_statuses;
    std::unordered_map<std::string, Hold<Operator>> operator_statuses;
    std::vector<std::string> execution_order;
    std::string operator_entry = "";

    // Protect the status map.
    // std::shared_mutex tm, om;

    MemoryInfo memory_info;

public:
    MemoryStatus() = default;
    MemoryStatus(const MemoryStatus& _status) = default;
    MemoryStatus(MemoryStatus&& _status) {
        // std::unique_lock<std::shared_mutex> tl{_status.tm};
        // std::unique_lock<std::shared_mutex> ol{_status.om};
        tensor_statuses = std::move(_status.tensor_statuses);
        operator_statuses = std::move(_status.operator_statuses);
        execution_order = std::move(_status.execution_order);
        operator_entry = std::move(_status.operator_entry);
        memory_info = _status.memory_info;
    }

    MemoryStatus& operator=(const MemoryStatus& _status) = default;
    MemoryStatus& operator=(MemoryStatus&& _status) {
        // std::unique_lock<std::shared_mutex> tl{_status.tm};
        // std::unique_lock<std::shared_mutex> ol{_status.om};
        tensor_statuses = std::move(_status.tensor_statuses);
        operator_statuses = std::move(_status.operator_statuses);
        execution_order = std::move(_status.execution_order);
        operator_entry = std::move(_status.operator_entry);
        memory_info = _status.memory_info;
        return *this;
    }

    void setMemoryInfo(const MemoryInfo& _memory_info) { memory_info = _memory_info; }
    MemoryInfo getMemoryInfo() const { return memory_info; }

    /**
     * registerTensor
     * Register a tensor to the storage.
     * Only can be invoked when tensor status storage not inited.
     * @param status tensorStatus
     */
    void registerTensor(const Tensor& status) {
        // std::unique_lock<std::shared_mutex> l{tm};

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
        // std::unique_lock<std::shared_mutex> l{tm};

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
        // std::unique_lock<std::shared_mutex> l{om};

        auto p = operator_statuses.find(status.getName());
        if (p != operator_statuses.end()) throw status_exception("Operator already registered.");

        for (auto &s : status.getTensors()) {
            if (tensor_statuses.find(s) == tensor_statuses.end())
                // Tensor not registered.
                throw status_exception("Specified tensor not registered.");
            // Set operator information for tensor.
            tensor_statuses.at(s).target.op = status.getName();
        }

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

    inline bool hasExecutionPost(const std::string& op) const {
        auto p = std::find(execution_order.begin(), execution_order.end(), op);
        if (p == execution_order.end()) throw status_exception("Operator not registered.");
        return ++p != execution_order.end();
    }
    inline std::string getExecutionPost(const std::string& op) const {
        auto p = std::find(execution_order.begin(), execution_order.end(), op);
        if (p == execution_order.end()) throw status_exception("Operator not registered.");
        if (++p == execution_order.end()) return "";
        return *p;
    }

    inline bool hasExecutionPrev(const std::string& op) const {
        auto p = std::find(execution_order.begin(), execution_order.end(), op);
        if (p == execution_order.end()) throw status_exception("Operator not registered.");
        return p != execution_order.begin();
    }
    inline std::string getExecutionPrev(const std::string& op) const {
        auto p = std::find(execution_order.begin(), execution_order.end(), op);
        if (p == execution_order.end()) throw status_exception("Operator not registered.");
        if (p == execution_order.begin()) return "";
        return *--p;
    }

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
    bool isOperatorRegistered(const std::string& op) const {
        return operator_statuses.find(op) != operator_statuses.end();
    }

    TensorView tryReferenceTensor(const std::string& tensor) {
        auto p = tensor_statuses.find(tensor);
        if (p == tensor_statuses.end()) throw status_exception("Tensor not registered.");
        return TensorView(p->second.target, p->second.m);
    }

    /**
     * Reference the tensor
     * @param tensor tensor name
     * @return reference to the specific tensor
     */
    inline TensorPres referenceTensor(const std::string& tensor) { return tryReferenceTensor(tensor).reference(); }

    OperatorView tryReferenceOperator(const std::string& op) {
        auto p = operator_statuses.find(op);
        if (p == operator_statuses.end()) throw status_exception("Operator not registered.");
        return OperatorView(p->second.target, p->second.m);
    }

    inline OperatorPres referenceOperator(const std::string& op) { return tryReferenceOperator(op).reference(); }

    inline std::unordered_set<std::string> getTensors() const {
        std::unordered_set<std::string> re;
        for (auto &x : tensor_statuses) re.insert(x.first);
        return re;
    }
    inline std::unordered_set<std::string> getOperators() const {
        std::unordered_set<std::string> re;
        for (auto &x : operator_statuses) re.insert(x.first);
        return re;
    }

    void unregisterOperator(const std::string& op) {
        // std::unique_lock<std::shared_mutex> l{om};
        
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
        // std::unique_lock<std::shared_mutex> l{tm};
        
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

namespace utils {

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

}   // namespace utils

using TensorView   = status::MemoryStatus::TensorView;
using OperatorView = status::MemoryStatus::OperatorView;

}   // namespace status

using TensorPres   = status::TensorPres;
using MemoryStatus = status::MemoryStatus;

}   // namespace mori