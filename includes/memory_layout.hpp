#pragma once

#include <string>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <shared_mutex>

#include "includes/memory_info.hpp"
#include "includes/symbols.hpp"
#include "includes/utils.hpp"
#include "includes/exceptions/memory_exceptions.hpp"
#include "includes/exceptions/status_exceptions.hpp"

namespace mori {
namespace layout {

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

struct MemoryMap;

struct MemoryMapBuilder final {
    std::unordered_map<std::string, Region> regions;
    std::vector<Layer> layers;

    MemoryInfo memory_info;

    int current_layer = 0;

    MemoryMapBuilder() { layers.emplace_back(); }

    inline void setMemoryInfo(const MemoryInfo& _memory_info) {
        if (layers.size() != 1 || !layers[0].regions.empty()) throw inited_exception("Memory map on built.");
        memory_info = _memory_info;
        layers[0].size = _memory_info.device.common_block.size;
    }
    inline const MemoryInfo& getMemoryInfo() const noexcept { return memory_info; };

    inline void createLayer() { 
        layers.emplace_back(memory_info.device.common_block.size);
        ++current_layer;
    }

    inline void submitMemoryRegion(int _layer, const Region& _region) {
        layers[_layer].submit(_region.name, _region.size);
        regions.emplace(_region.name, _region);
    }
    inline void submitMemoryRegion(const Region& _region) { submitMemoryRegion(current_layer, _region); }

    inline int getLayersCount() const { return layers.size(); }
    inline Layer& getLayer(int layer) { return layers[layer]; }
    inline const Layer& getLayer(int layer) const { return layers[layer]; }
    inline Layer& getCurrentLayer() { return getLayer(current_layer); }
    inline const Layer& getCurrentLayer() const { return getLayer(current_layer); }
    inline std::vector<Layer>& getLayers() { return layers; }
    inline const std::vector<Layer>& getLayers() const { return layers; }
    inline const std::vector<size_t>& getSections(const std::string& tensor) const { return regions.at(tensor).sections; }
    inline size_t getFragmentSize(const std::string& tensor) const { return regions.at(tensor).fragment_size; }
    inline std::unordered_map<std::string, Region>& getRegions() { return regions; }
    inline const std::unordered_map<std::string, Region>& getRegions() const { return regions; }

    std::unordered_map<std::string, size_t> getFragmentInfo() const {
        std::unordered_map<std::string, size_t> re;
        for (auto &x : regions) {
            if (x.second.fragment_size != 0) re.emplace(x.first, x.second.fragment_size);
        }
        return re;
    }

    inline MemoryMap build();

    void clear() {
        regions.clear();
        layers.clear();
    }
};  // struct MemoryMapBuilder

/**
 * Describe the layout for all tensors in the memory.
 */
struct MemoryMap {
private:
    std::unordered_map<std::string, Region> regions;
    std::vector<Layer> layers;

    MemoryInfo memory_info;

    int current_layer = 0;

public:
    MemoryMap() = default;
    MemoryMap(const MemoryMapBuilder& builder): regions(builder.regions), layers(builder.layers), memory_info(builder.memory_info), current_layer(builder.current_layer) {}

    inline const MemoryInfo& getMemoryInfo() const noexcept { return memory_info; };

    inline const Layer& referenceLayer(int _layer) const { return layers.at(_layer); }
    inline const Layer& referenceLayer() const { return referenceLayer(current_layer); }
    inline const Region& referenceRegion(const std::string& _region) const { return regions.at(_region); }

    inline int getLayersCount() const { return layers.size(); }
    inline const Layer& getLayer(int layer) const { return layers[layer]; }
    inline const Layer& getCurrentLayer() const { return getLayer(current_layer); }
    inline const std::vector<Layer>& getLayers() const { return layers; }
    inline const std::vector<size_t>& getSections(const std::string& tensor) const { return regions.at(tensor).sections; }
    inline size_t getFragmentSize(const std::string& tensor) const { return regions.at(tensor).fragment_size; }
    inline const std::unordered_map<std::string, Region>& getRegions() const { return regions; }

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

MemoryMap MemoryMapBuilder::build() { return MemoryMap(*this); }

enum struct MemoryBlockType {
    common, persistent, transient
};  // enum struct MemoryBlockType

struct MemoryRegion final {
    std::string name = "";  // Tensor information

    void* address = nullptr;
    size_t size = 0;

    bool allocated = false;
};  // struct MemoryRegion

struct Block final {
    MemoryBlockType type;
    std::map<void*, MemoryRegion> regions;
    mutable std::shared_mutex m;
    size_t total_size;

    Block(MemoryBlockType block_type, void* address, size_t size) {
        type = block_type;
        total_size = size;
        MemoryRegion s;
        s.address = address;
        s.size    = size;
        regions.emplace(s.address, s);
    }
    Block(const Block& block) {
        type    = block.type;
        regions = block.regions;
        total_size = block.total_size;
    }
    Block(Block&& block) {
        type    = block.type;
        regions = std::move(block.regions);
        total_size = block.total_size;
    }
};  // struct Block

struct MemoryDefragmentationExecutor;

struct MemoryLayout final {
private:
    friend struct MemoryDefragmentationExecutor;

private:
    std::map<void*, Block> blocks;
    // std::shared_mutex m;

    size_t align_size;

protected:
    inline std::map<void*, Block>::const_iterator locateMemoryBlock(void* address) const {
        auto bp = blocks.upper_bound(address);
        if (bp == blocks.begin()) return blocks.cend();
        return std::prev(bp);
    }
    inline std::map<void*, Block>::iterator locateMemoryBlock(void* address) {
        auto bp = blocks.upper_bound(address);
        if (bp == blocks.begin()) return blocks.end();
        return std::prev(bp);
    }

public:
    MemoryLayout() = default;

    inline void setMemoryInfo(const MemoryInfo& info) {
        assert(blocks.empty());

        blocks.emplace(info.device.common_block.address,     Block(MemoryBlockType::common,     info.device.common_block.address,     info.device.common_block.size));
        blocks.emplace(info.device.persistent_block.address, Block(MemoryBlockType::persistent, info.device.persistent_block.address, info.device.persistent_block.size));
        blocks.emplace(info.device.transient_block.address,  Block(MemoryBlockType::transient,  info.device.transient_block.address,  info.device.transient_block.size));

        align_size  = info.device.align_size;
    }

    bool isRegionExist(void* address, Direction direction = Direction::post) const {
        if (address == nullptr) throw memory_address_invalid();
        auto bp = locateMemoryBlock(address);
        if (bp == blocks.end()) return false;
        std::shared_lock<std::shared_mutex> l{bp->second.m};
        auto& regions = bp->second.regions;
        if (direction == Direction::post) return regions.find(address) != regions.end();
        else return regions.find(address) != regions.begin();
    }
    MemoryRegion getMemoryRegion(void* address, Direction direction = Direction::post) const {
        if (address == nullptr) throw memory_address_invalid();
        auto bp = locateMemoryBlock(address);
        if (bp == blocks.end()) throw memory_unmanaged();
        std::shared_lock<std::shared_mutex> l{bp->second.m};
        auto& regions = bp->second.regions;
        auto sp = regions.find(address);
        if (direction == Direction::post) {
            if (sp == regions.end()) throw memory_unmanaged();
            return sp->second;
        } else {
            if (sp == regions.begin()) throw memory_unmanaged();
            return std::prev(sp)->second;
        }
    }

    bool isPersistent(void* address) const {
        if (address == nullptr) throw memory_address_invalid();
        auto bp = locateMemoryBlock(address);
        if (bp == blocks.end()) throw memory_unmanaged();
        std::shared_lock<std::shared_mutex> l{bp->second.m};
        return bp->second.type == MemoryBlockType::persistent;
    }
    bool isTransient(void* address) const {
        if (address == nullptr) throw memory_address_invalid();
        auto bp = locateMemoryBlock(address);
        if (bp == blocks.end()) throw memory_unmanaged();
        std::shared_lock<std::shared_mutex> l{bp->second.m};
        return bp->second.type == MemoryBlockType::transient;
    }
    bool isCommon(void* address) const {
        if (address == nullptr) throw memory_address_invalid();
        auto bp = locateMemoryBlock(address);
        if (bp == blocks.end()) throw memory_unmanaged();
        std::shared_lock<std::shared_mutex> l{bp->second.m};
        return bp->second.type == MemoryBlockType::common;
    }

    void recordMemoryAllocateEvent(void* address, size_t size, const std::string& tensor, size_t alignment) {
        if (address == nullptr) throw memory_address_invalid();
        // Since MemoryLayout is only a recorder of memory layout information, no need to implement for malloc and salloc seperately.
        if (size == 0) return recordMemoryAllocateEvent(address, alignment, tensor, alignment);

        auto bp = blocks.upper_bound(address);
        assert(bp != blocks.begin());
        --bp;

        std::unique_lock<std::shared_mutex> l{bp->second.m};
        auto& regions = bp->second.regions;
        auto p = regions.begin();
        while (p != regions.end() && utils::address_offset(p->first, p->second.size) <= address) ++p;
        if (p == regions.end() || p->first > address || p->second.allocated) throw memory_allocated(address);
        if (utils::address_offset(p->first, p->second.size) < utils::address_offset(address, size)) throw memory_operation_invalid(address, "Memory cannot be allocated at specificied address with size.");

        // The original unallocated space should be splited to three parts.
        if (p->first < address) {
            // Left part exists.
            MemoryRegion s;
            s.address = address;
            s.size    = (uint8_t*)p->first - (uint8_t*)address + p->second.size;
            auto q = regions.emplace(address, s);
            assert(q.second);
            p->second.size = (uint8_t*)address - (uint8_t*)p->first;
            p = q.first;
        }
        // Now p->first == address
        if (p->second.size > size) {
            // Right part exists.
            // Create empty region
            MemoryRegion s;
            s.address = (uint8_t*)address + size;
            s.size    = p->second.size - size;
            auto q = regions.emplace(s.address, s);
            assert(q.second);
            p->second.size = size;
        }
        p->second.name      = tensor;
        p->second.allocated = true;
    }
    void recordMemoryAllocateEvent(void* address, size_t size, const std::string& tensor) {
        if (address == nullptr) throw memory_address_invalid();
        if (!utils::memory_address_aligned(address, align_size)) throw memory_exception(address, "Memory address not aligned.");
        size_t aligned_size = utils::get_memory_aligned_size(size, align_size);
        if (aligned_size == 0) aligned_size = align_size;
        recordMemoryAllocateEvent(address, aligned_size, tensor, align_size);
    }
    void recordMemoryFreeEvent(void* address, const std::string& tensor = "") {
        if (address == nullptr) throw memory_address_invalid();        
        auto bp = locateMemoryBlock(address);
        if (bp == blocks.end()) throw memory_not_allocated(address);

        std::unique_lock<std::shared_mutex> l{bp->second.m};
        auto& regions = bp->second.regions;
        // Check if allocated device memory.
        auto p = regions.find(address);
        // Device memory not allocated.
        if (p == regions.end() || !p->second.allocated) throw memory_not_allocated(address);
        p->second.name      = "";
        p->second.allocated = false;

        // Merging free regions.
        auto prev = p;
        auto post = p;
        ++post;
        if (post != regions.end() && !post->second.allocated) {
            p->second.size += post->second.size;
            regions.erase(post);
        }

        if (p == regions.begin()) return;
        --prev;
        if (!prev->second.allocated) {
            prev->second.size += p->second.size;
            regions.erase(p);
        }
    }
    void recordMemorySplitEvent(void* address, size_t size) {
        if (address == nullptr) throw memory_address_invalid();
        auto bp = locateMemoryBlock(address);
        if (bp == blocks.end()) throw memory_not_allocated(address);

        std::unique_lock<std::shared_mutex> l{bp->second.m};
        auto& regions = bp->second.regions;
        auto p = regions.find(address);
        if (p == regions.end() || !p->second.allocated) throw memory_not_allocated(address);
        if (p->second.size <= size) throw memory_operation_invalid(address, "Memory section equals or be smaller than spliting size.");

        MemoryRegion s = p->second;
        s.address = utils::address_offset(address, size);
        s.size   -=  size;
        regions.emplace(s.address, s);
        p->second.size = size;
    }
    void recordMemoryMergeEvent(void* left, void* right) {
        if (left == nullptr || right == nullptr) throw memory_address_invalid();
        auto bp = locateMemoryBlock(left);
        if (bp == blocks.end()) throw memory_not_allocated(left);

        std::unique_lock<std::shared_mutex> l{bp->second.m};
        auto& regions = bp->second.regions;
        auto q = regions.find(left);
        if (q == regions.end() || !q->second.allocated) throw memory_not_allocated(left, "Memory for left section not allocated."); 
        auto p = q++;
        if (q == regions.end() || q->first != right || !q->second.allocated) throw memory_not_allocated(right, "Memory for right section not allocated."); 
        if ((uint8_t*)left + p->second.size != (uint8_t*)right) throw memory_operation_invalid(left, "Memory sections not continuous.");

        p->second.size += q->second.size;
        regions.erase(q);
    }

};  // struct MemoryLayout

}   // namespace layout
}   // namespace mori