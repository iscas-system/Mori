#pragma once

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <shared_mutex>

#include "includes/memory_info.hpp"
#include "includes/address_utils.hpp"
#include "includes/exceptions/memory_exceptions.hpp"

namespace mori {
namespace decisions {
    struct Model;
}   // namespace decisions

struct Region final {
    std::string name;
    size_t size = 0;

    std::vector<size_t> sections;
    size_t fragment_size = 0;

    Region() = default;
    Region(const std::string& _name, size_t _size): name(_name), size(_size) {}
};  // struct Region

struct Layer final {
public:
    using iterator       = std::vector<std::string>::iterator;
    using const_iterator = std::vector<std::string>::const_iterator;
public:
    std::vector<std::string> regions;
    size_t size = 0;
    size_t requested_size = 0;

public:
    Layer() = default;
    Layer(size_t _size): size(_size) {}

    inline void submit(const std::string& name, size_t size) {
        regions.push_back(name);
        requested_size += size;
    }

    inline bool isAccomodatable() const noexcept { return requested_size <= size; }

    iterator       begin() { return regions.begin(); }
    const_iterator begin() const { return regions.begin(); }

    iterator       end() { return regions.end(); }
    const_iterator end() const { return regions.end(); }
};  // struct Layer

/**
 * Describe the layout for all tensors in the memory.
 */
struct MemoryMap final {
private:
    friend struct decisions::Model;

private:
    std::unordered_map<std::string, Region> regions;
    std::vector<Layer> layers;

    size_t memory_size = 0;

    int current_layer = 0;

public:
    MemoryMap() { layers.emplace_back(); }

    inline void setMemorySize(size_t _size) {
        memory_size = _size;
        layers[0].size = _size;
    }
    inline size_t getMemorySize() { return memory_size; }

    inline void createLayer() { 
        layers.emplace_back(memory_size);
        ++current_layer;
    }
    inline Layer& referenceLayer(int _layer) { return layers[_layer]; }
    inline Layer& referenceLayer() { return referenceLayer(current_layer); }

    inline void submitMemoryRegion(int _layer, const Region& _region) {
        layers[_layer].submit(_region.name, _region.size);
        regions.emplace(_region.name, _region);
    }
    inline void submitMemoryRegion(const Region& _region) { submitMemoryRegion(current_layer, _region); }

    inline int getLayersCount() const { return layers.size(); }
    inline Layer& getLayer(int layer) { return layers[layer]; }
    inline const Layer& getLayer(int layer) const { return layers[layer]; }
    inline Layer& getCurrentLayer() { return getLayer(current_layer); }
    inline std::vector<size_t> getSections(const std::string& tensor) const { return regions.at(tensor).sections; }
    inline size_t getFragmentSize(const std::string& tensor) const { return regions.at(tensor).fragment_size; }

    std::unordered_map<std::string, size_t> getFragmentInfo() const {
        std::unordered_map<std::string, size_t> re;
        for (auto &x : regions) {
            if (x.second.fragment_size != 0) re.emplace(x.first, x.second.fragment_size);
        }
        return re;
    }

    void clear() {
        regions.clear();
        layers.clear();
    }
};  // struct MemoryMap

namespace layout {

struct MemorySection final {
    std::string name = "";  // Tensor information

    void* address = nullptr;
    size_t size = 0;

    bool allocated = false;
};  // struct MemorySection

struct Block final {
    std::map<void*, MemorySection> sections;

    Block(void* address, size_t size) {
        MemorySection s;
        s.address = address;
        s.size    = size;
        sections.emplace(s.address, s);
    }
};  // struct Block

struct MemoryLayout final {
private:
    std::map<void*, Block> blocks;

    std::shared_mutex m;

    size_t device_size;
    size_t block_size;
    size_t align_size;

protected:
    inline std::map<void*, Block>::iterator locateMemoryBlock(void* address) {
        auto bp = blocks.upper_bound(address);
        if (bp == blocks.begin()) return blocks.end();
        return --bp;
    }

public:
    MemoryLayout() = default;

    inline void setMemoryInfo(const MemoryInfo& info) {
        device_size = info.device.total_size;
        block_size  = info.device.block_size;
        align_size  = info.device.align_size;
    }

    inline bool isSectionExist(void* address) {
        std::shared_lock<std::shared_mutex> l{m};
        auto bp = locateMemoryBlock(address);
        if (bp == blocks.end()) return false;
        auto& sections = bp->second.sections;
        return sections.find(address) != sections.end();
    }
    inline MemorySection getMemorySection(void* address) {
        std::shared_lock<std::shared_mutex> l{m};
        auto bp = locateMemoryBlock(address);
        if (bp == blocks.end()) throw memory_unmanaged();
        auto& sections = bp->second.sections;
        auto sp = sections.find(address);
        if (sp == sections.end()) throw memory_unmanaged();
        return sp->second;
    }

    inline void recordMemoryAllocateEvent(void* address, size_t size, const std::string& tensor, size_t alignment) {
        // Since MemoryLayout is only a recorder of memory layout information, no need to implement for malloc and salloc seperately.
        if (size == 0) {
            recordMemoryAllocateEvent(address, alignment, tensor, alignment);
            return;
        }

        std::unique_lock<std::shared_mutex> l{m};

        auto bp = blocks.upper_bound(address);
        if (bp == blocks.begin()) bp = blocks.emplace(address, Block(address, block_size)).first;
        else {
            --bp;
            if (utils::address_offset(bp->first, block_size) <= address) bp = blocks.emplace(address, Block(address, block_size)).first;
        }

        auto& sections = bp->second.sections;
        auto p = sections.begin();
        while (p != sections.end() && utils::address_offset(p->first, p->second.size) <= address) ++p;
        if (p == sections.end() || p->first > address || p->second.allocated) throw memory_allocated(address);
        if (utils::address_offset(p->first, p->second.size) < utils::address_offset(address, size)) throw memory_operation_invalid(address, "Memory cannot be allocated at specificied address with size.");

        // The original unallocated space should be splited to three parts.
        if (p->first < address) {
            // Left part exists.
            MemorySection s;
            s.address = address;
            s.size    = (uint8_t*)p->first - (uint8_t*)address + p->second.size;
            auto q = sections.emplace(address, s);
            assert(q.second);
            p->second.size = (uint8_t*)address - (uint8_t*)p->first;
            p = q.first;
        }
        // Now p->first == address
        if (p->second.size > size) {
            // Right part exists.
            // Create empty section
            MemorySection s;
            s.address = (uint8_t*)address + size;
            s.size    = p->second.size - size;
            auto q = sections.emplace(s.address, s);
            assert(q.second);
            p->second.size = size;
        }
        p->second.name      = tensor;
        p->second.allocated = true;
    }
    inline void recordMemoryAllocateEvent(void* address, size_t size, const std::string& tensor) {
        if (!utils::memory_address_aligned(address, align_size)) throw memory_exception(address, "Memory address not aligned.");
        size_t aligned_size = utils::get_memory_aligned_size(size, align_size);
        if (aligned_size == 0) aligned_size = align_size;
        recordMemoryAllocateEvent(address, aligned_size, tensor, align_size);
    }
    inline void recordMemoryFreeEvent(void* address, const std::string& tensor = "") {
        std::unique_lock<std::shared_mutex> l{m};
        
        auto bp = locateMemoryBlock(address);
        if (bp == blocks.end()) throw memory_not_allocated(address);

        auto& sections = bp->second.sections;
        // Check if allocated device memory.
        auto p = sections.find(address);
        // Device memory not allocated.
        if (p == sections.end() || !p->second.allocated) throw memory_not_allocated(address);
        p->second.name      = "";
        p->second.allocated = false;

        // Merging free sections.
        auto prev = p;
        auto post = p;
        ++post;
        if (post != sections.end() && !post->second.allocated) {
            p->second.size += post->second.size;
            sections.erase(post);
        }

        if (p == sections.begin()) return;
        --prev;
        if (!prev->second.allocated) {
            prev->second.size += p->second.size;
            sections.erase(p);
        }
    }
    inline void recordMemorySplitEvent(void* address, size_t size) {
        std::unique_lock<std::shared_mutex> l{m};

        auto bp = locateMemoryBlock(address);
        if (bp == blocks.end()) throw memory_not_allocated(address);

        auto& sections = bp->second.sections;
        auto p = sections.find(address);
        if (p == sections.end() || !p->second.allocated) throw memory_not_allocated(address);
        if (p->second.size <= size) throw memory_operation_invalid(address, "Memory section equals or be smaller than spliting size.");

        MemorySection s = p->second;
        s.address = utils::address_offset(address, size);
        s.size   -=  size;
        sections.emplace(s.address, s);
        p->second.size = size;
    }
    inline void recordMemoryMergeEvent(void* left, void* right) {
        std::unique_lock<std::shared_mutex> l{m};

        auto bp = locateMemoryBlock(left);
        if (bp == blocks.end()) throw memory_not_allocated(left);

        auto& sections = bp->second.sections;
        auto q = sections.find(left);
        if (q == sections.end() || !q->second.allocated) throw memory_not_allocated(left, "Memory for left section not allocated."); 
        auto p = q++;
        if (q == sections.end() || q->first != right || !q->second.allocated) throw memory_not_allocated(right, "Memory for right section not allocated."); 
        if ((uint8_t*)left + p->second.size != (uint8_t*)right) throw memory_operation_invalid(left, "Memory sections not continuous.");

        p->second.size += q->second.size;
        sections.erase(q);
    }

};  // struct MemoryLayout

}   // namespace layout
}   // namespace mori