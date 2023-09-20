#pragma once

#include "frontend/memory_operation_executor.hpp"
#include "includes/memory_layout.hpp"

namespace mori {
namespace layout {

struct MemoryDefragmentationExecutor {
private:
    status::MemoryStatus& status;
    layout::MemoryLayout& layout;
    MemoryManager* memory_manager;

    std::map<size_t, std::set<void*>> allocated_regions;
    std::map<size_t, std::set<void*>> idle_regions;
    
    std::shared_mutex m;

protected:
    void performCopyDevice(void* src, void* dst, size_t size) {
        assert(src >= dst);

        MemoryRegion region = layout.getMemoryRegion(src);

        status::TensorView tensor_view = status.tryReferenceTensor(region.name); 
        if (!tensor_view.isReferenced()) return;     // Cannot move this tensor
        status::TensorPres tensor_pres =  tensor_view.reference();

        const status::MemorySection* section = &(tensor_pres.getFirstSection());
        while (section != nullptr) {
            if (section->device_address == src) break;
            section = section->next();
        }
        if (section == nullptr) throw memory_unmanaged();

        if (utils::address_offset(dst, size) >= src) {
            // No interleaving of the two memory regions.
            memory_manager->salloc(dst, size);
            memory_manager->copyDevice(src, dst, size);
            memory_manager->freeDevice(src);

            layout.recordMemoryAllocateEvent(dst, size, tensor_pres.getName());
            layout.recordMemoryFreeEvent(src);

            allocated_regions[size].insert(dst);
            allocated_regions[size].erase(src);
            idle_regions[size].insert(src);
            idle_regions[size].erase(dst);

        } else {
            // Interleaving of memory regions.
            memory_manager->salloc(dst, utils::address_distance(src, dst));
            bool res = memory_manager->merge(dst, src);
            assert(res);
            memory_manager->copyDevice(src, dst, size);
            void* right = memory_manager->split(dst, size);
            memory_manager->freeDevice(right);

            layout.recordMemoryAllocateEvent(dst, utils::address_distance(src, dst), tensor_pres.getName());
            layout.recordMemoryMergeEvent(dst, src);
            layout.recordMemorySplitEvent(dst, size);
            layout.recordMemoryFreeEvent(right);

            allocated_regions[size].insert(dst);
            allocated_regions[size].erase(src);
            idle_regions[utils::address_distance(src, dst)].insert(src);
            idle_regions[utils::address_distance(src, dst)].erase(dst);
        }

        tensor_pres.setMoved(section->offset, dst);
    }

public:
    MemoryDefragmentationExecutor(status::MemoryStatus& _status, layout::MemoryLayout& _layout): status(_status), layout(_layout) {}

    inline void setMemoryManager(MemoryManager* _memory_manager) {
        memory_manager = _memory_manager;
        
        assert(allocated_regions.empty());
        assert(idle_regions.empty());

        MemoryInfo&& info = memory_manager->getMemoryInfo();
        idle_regions[info.device.transient_block.size].insert(info.device.transient_block.address);
    }

    inline void recordMemoryAllocateEvent(void* address) {
        if (!layout.isTransient(address)) throw memory_unmanaged();
        MemoryRegion region = layout.getMemoryRegion(address);

        std::unique_lock<std::shared_mutex> l{m};

        assert(allocated_regions[region.size].count(address) == 0);
        assert(idle_regions[region.size].count(address) == 1);

        allocated_regions[region.size].insert(address);
        idle_regions[region.size].erase(address);
    }

    inline void recordMemoryFreeEvent(void* address) {
        if (!layout.isTransient(address)) throw memory_unmanaged();
        MemoryRegion region = layout.getMemoryRegion(address);

        std::unique_lock<std::shared_mutex> l{m};

        assert(allocated_regions[region.size].count(address) == 1);
        assert(idle_regions[region.size].count(address) == 0);

        allocated_regions[region.size].erase(address);
        idle_regions[region.size].insert(address);
    }

    std::pair<size_t, size_t> getTransientBlockAllocatableSize(size_t granularity) const noexcept {
        std::pair<size_t, size_t> re;
        re.first  = 0;
        re.second = 0;
        for (auto &x : idle_regions) {
            if (x.first >= granularity) re.first += x.first * x.second.size();
            else re.second += x.first * x.second.size();
        }
        return re;
    }

    void performDefragmentation(size_t granularity) {
        auto bp = std::find_if(layout.blocks.begin(), layout.blocks.end(), [](const std::pair<void*, Block>& p) { return p.second.type == MemoryBlockType::transient; });
        assert(bp != layout.blocks.end());

        std::unique_lock<std::shared_mutex> l{bp->second.m};
        auto& regions = bp->second.regions;

        for (auto p = regions.begin(); p != regions.end(); ++p) {
            if (p->second.allocated) continue;
            if (p->second.size >= granularity) continue;
            // Find fragmentation
            if (!allocated_regions[p->second.size].empty()) {
                auto q = allocated_regions[p->second.size].rbegin();
                if (*q != p->first) {
                    // Fast path
                    assert(*q >= p->first);
                    performCopyDevice(*q, p->first, p->second.size);
                    continue;
                }
            }
            // Slow path
            auto q = p;
            ++q;
            if (q == regions.end()) break;
            assert(q->second.allocated);
            performCopyDevice(q->first, p->first, q->second.size);
        }

    }

};  // struct MemoryDefragmentationExecutor

}   // namespace layout
}   // namespace mori
