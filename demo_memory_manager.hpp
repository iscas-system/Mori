#include <iostream>
#include <set>
#include <map>
#include <unordered_map>
#include <utility>
#include <exception>
#include <shared_mutex>
#include <cstdlib>
#include <cassert>

#include "frontend/libmori.hpp"

#define DEVICE_MEM 2048
#define COMMON_MEM 1536

struct DemoMemoryManager : public mori::MemoryManager {
private:
    struct MemoryStatus {
        size_t size = 0;
        bool allocated = false;
    };  // inner struct MemoryStatus

private:
    std::shared_mutex dm;
    std::shared_mutex hm;

    void* device_base = nullptr;
    std::map<void*, MemoryStatus> device;
    std::set<void*> host;

    std::map<void*, MemoryStatus>::iterator current_posi;

protected:
    inline bool isCommonMemCovered(void* address) {
        assert(device_base >= address);
        assert(mori::utils::address_offset(device_base, DEVICE_MEM) <= address);

        return mori::utils::address_offset(device_base, COMMON_MEM) > address;
    }

    void* allocateDeviceBestFit(size_t size) {
        // Slower version of best-fit strategy.
        auto p = device.rbegin();
        std::map<size_t, std::vector<std::map<void*, MemoryStatus>::reverse_iterator>> candidates;
        while (p != device.rend()) {
            if (!p->second.allocated && p->second.size >= size) candidates[p->second.size].push_back(p);
            ++p;
        }

        if (candidates.empty()) return nullptr;
        auto re = candidates.begin()->second.front();

        if (re->second.size > size) device.emplace((uint8_t*)p->first + size, MemoryStatus{p->second.size - size, false});
        re->second.allocated = true;
        re->second.size = size;
        return re->first;
    }

    void* allocateDeviceGTM(size_t size) {
        auto p = current_posi;
        bool continued = false;
        while (true) {
            if (p == device.end()) {
                continued = true;
                p = device.begin();
            }
            if (continued && p == current_posi) break;
            if (!p->second.allocated && p->second.size >= size) {
                // Allocatable
                if (p->second.size > size) device.emplace((uint8_t*)p->first + size, MemoryStatus{p->second.size - size, false});
                p->second.allocated = true;
                p->second.size = size;
                current_posi = p;
                if (++current_posi == device.end()) current_posi = device.begin();
                return p->first;
            }
            ++p;
        }
        return nullptr;
    }

public:
    DemoMemoryManager() {
        device_base = malloc(DEVICE_MEM);
        if (device_base == nullptr) throw mori::memory_device_insufficience("Memory on device insufficient.", DEVICE_MEM);
        device.emplace(device_base, MemoryStatus{DEVICE_MEM, false});
        current_posi = device.begin();
    }

    virtual void* allocateDevice(size_t size, bool transient = false) override {
        std::unique_lock<std::shared_mutex> l{dm};

        void* re = nullptr;
        if (!transient) re = allocateDeviceGTM(size);
        if (re == nullptr) re = allocateDeviceBestFit(size);
        return re;
    }

    virtual void* allocateHost(size_t size) override {
        std::unique_lock<std::shared_mutex> l{hm};
        void* ret = malloc(size);
        host.insert(ret);
        return ret;
    }

    virtual void copyIn(void* host_address, void* device_address, size_t size) override {
        // std::shared_lock<std::shared_mutex> dl{dm};
        // std::shared_lock<std::shared_mutex> hl{hm};
        std::this_thread::sleep_for(std::chrono::milliseconds(size >> 2));
        std::cout<<size<< " copied in.\n";
    }

    virtual void copyOut(void* device_address, void* host_address, size_t size) override {
        // std::shared_lock<std::shared_mutex> dl{dm};
        // std::shared_lock<std::shared_mutex> hl{hm};
        std::this_thread::sleep_for(std::chrono::milliseconds(size >> 2));
        std::cout<<size<< " copied out.\n";
    }

    virtual void freeDevice(void* address) override {
        std::unique_lock<std::shared_mutex> l{dm};
        // Check if allocated device memory.
        auto p = device.find(address);
        // Device memory not allocated.
        if (p == device.end() || !p->second.allocated) throw mori::memory_not_allocated();
        p->second.allocated = false;
        std::cout<<p->second.size<<" freed.\n";
        auto prev = p;
        auto post = p;
        ++post;
        if (post != device.end() && !post->second.allocated) {
            p->second.size += post->second.size;
            if (current_posi == post) current_posi = p;
            device.erase(post);
        }

        if (p == device.begin()) return;
        --prev;
        if (!prev->second.allocated) {
            prev->second.size += p->second.size;
            if (current_posi == p) current_posi = prev;
            device.erase(p);
        }
    }

    virtual void freeHost(void* address) override {
        std::unique_lock<std::shared_mutex> l{hm};
        auto p = host.find(address);
        if (p == host.end()) throw mori::memory_not_allocated();
        std::free(address);
        host.erase(p);
    }

    virtual bool isMemorySectionSupported() const override { return false; }

    virtual void copyDevice(void* src, void* dst, size_t size) override {
        // std::shared_lock<std::shared_mutex> l{dm};
        std::this_thread::sleep_for(std::chrono::milliseconds(size >> 2));
    }

    virtual void* split(void* address, size_t size) override {
        std::unique_lock<std::shared_mutex> l{dm};
        // Check if allocated device memory.
        auto p = device.find(address);
        // Device memory not allocated.
        if (p == device.end() || !p->second.allocated) throw mori::memory_not_allocated();
        if (p->second.size <= size) throw mori::memory_operation_invalid("Memory section equals or be smaller than spliting size.");

        device.emplace((uint8_t*)address + size, MemoryStatus{p->second.size - size, true});
        p->second.size = size;
        return (uint8_t*)address + size;
    }

    virtual void* salloc(void* address, size_t size) override {
        std::unique_lock<std::shared_mutex> l{dm};
        auto p = device.begin();
        while (p != device.end()) {
            if (p == device.end()) return nullptr;
            if ((uint8_t*)p->first + p->second.size <= (uint8_t*)address) {
                ++p;
                continue;
            }
            if (p->first > address) return nullptr;
            if (p->second.allocated) return nullptr;
            break;
        }

        current_posi = p;

        if ((uint8_t*)p->first + p->second.size < (uint8_t*)address + size) return nullptr;

        // The original unallocated space should be splited to three parts.
        if (p->first < address) {
            // Left part exists.
            auto q = device.emplace(address, MemoryStatus{(uint8_t*)p->first - (uint8_t*)address + p->second.size, false});
            p->second.size = (uint8_t*)address - (uint8_t*)p->first;
            p = q.first;
        }
        // Now p->first == address
        if (p->second.size > size) {
            // Right part exists.
            auto q = device.emplace((uint8_t*)address + size, MemoryStatus{p->second.size - size, false});
            p->second.size = size;
        }
        p->second.allocated = true;
        return address;
    }

    virtual bool merge(void* left, void* right) override {
        std::unique_lock<std::shared_mutex> l{dm};
        auto q = device.find(left);
        if (q == device.end() || !q->second.allocated) return false;
        auto p = q++;
        if (q == device.end() || q->first != right || !q->second.allocated) return false;
        if ((uint8_t*)left + p->second.size != (uint8_t*)right) return false;

        p->second.size += q->second.size;
        device.erase(q);
        return true;
    }

    virtual mori::MemoryInfo getMemoryInfo() const override {
        auto re = mori::create_default_memory_info(DEVICE_MEM, 32 * 1024);
        re.device.common_block.address = device.begin()->first;
        re.device.common_block.size    = DEVICE_MEM;
        re.device.align_size           = 1;
        return re;
    }

    void check() const {
        assert(device.size() == 1);
        assert(device.begin()->second.size == DEVICE_MEM);
        assert(!device.begin()->second.allocated);
        assert(host.empty());
    }

    ~DemoMemoryManager() {
        std::free(device.begin()->first);
    }
};  // struct DemoMemoryManager