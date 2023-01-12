#include <iostream>
#include <map>
#include <utility>
#include <exception>
#include <mutex>
#include <cstdlib>

#include "frontend/libmori.hpp"

#define DEVICE_MEM 2048

struct DemoMemoryManager : public mori::MemoryManager {
private:
    struct MemoryStatus {
        size_t ori_size = 0;
        size_t size = 0;
        bool allocated = false;
    };  // struct MemoryStatus

private:
    std::recursive_mutex m;

    std::map<void*, MemoryStatus> host_map;
    std::map<void*, MemoryStatus> device_map;

    size_t total_size = 0;

public:
    DemoMemoryManager() {}

    virtual void* allocateDevice(size_t size) override {
        std::unique_lock<std::recursive_mutex>{m};

        // Assume 2 MB of device memory.
        if (total_size + size > DEVICE_MEM) return nullptr;

        // Sufficient device memory
        void* ret_address = malloc(size);
        device_map.insert(std::make_pair(ret_address, MemoryStatus{size, size, true}));

        total_size += size;
        return ret_address;
    }

    virtual void* allocateHost(size_t size) override {
        // Allocate host memory and copy data.
        return malloc(size);
    }

    virtual void copyIn(void* host_address, void* device_address, size_t size) override {
        std::unique_lock<std::recursive_mutex>{m};

        // Check if allocated host memory.
        auto p = host_map.find(host_address);
        // Host memory not allocated.
        if (p == host_map.end() || !p->second.allocated) throw std::exception();

        std::memcpy(device_address, host_address, size);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        std::cout<<size<< " copied in.\n";
        device_map.emplace(device_address, MemoryStatus{size, size, true});
    }

    virtual void copyOut(void* device_address, void* host_address, size_t size) override {
        std::unique_lock<std::recursive_mutex>{m};

        // Check if allocated device memory.
        auto p = device_map.find(device_address);
        // Device memory not allocated.
        if (p == device_map.end() || !p->second.allocated) throw std::exception();

        std::memcpy(host_address, device_address, size);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        std::cout<<size<< " copied out.\n";
        host_map.emplace(host_address, MemoryStatus{size, size, true});
    }

    virtual void freeDevice(void* address) override {
        std::unique_lock<std::recursive_mutex>{m};

        // Check if allocated device memory.
        auto p = device_map.find(address);
        // Device memory not allocated.
        if (p == device_map.end() || !p->second.allocated) throw std::exception();

        size_t ori_size   = p->second.ori_size;
        size_t freed_size = 0;
        while (freed_size < ori_size) {
            if (p == device_map.end() || !p->second.allocated) throw std::exception();
            std::free(p->first);
            total_size -= p->second.size;
            freed_size += p->second.size;
            device_map.erase(p++);
        }
        assert(freed_size == ori_size);
        std::cout<<ori_size<<" freed.\n";
    }

    virtual void freeHost(void* address) override {
        std::unique_lock<std::recursive_mutex>{m};

        // Check if allocated host memory.
        auto p = host_map.find(address);
        // Device memory not allocated.
        if (p == host_map.end() || !p->second.allocated) throw std::exception();

        std::free(p->first);
        host_map.erase(p);
    }

    virtual bool isMemorySectionSupported() const override { return true; }

    virtual void freeSection(void* address, size_t size) override {
        std::unique_lock<std::recursive_mutex>{m};

        // Check if allocated device memory.
        auto p = device_map.find(address);
        // Device memory not allocated.
        if (p == device_map.end() || !p->second.allocated) throw std::exception();

        device_map.emplace((uint8_t*)address + size, MemoryStatus{p->second.size - size, p->second.size - size, true});
        p->second.size = size;
        p->second.allocated = false;

        total_size -= size;
        std::cout<<size<<" freed.\n";
    }
    virtual void* realloc(void* address, size_t size) override { 
         std::unique_lock<std::recursive_mutex>{m};

        // Check if allocated device memory.
        auto p = device_map.find(address);
        // Device memory not allocated before.
        if (p == device_map.end() || p->second.allocated) throw std::exception();

        auto cur = p++;
        if (cur->second.size != size) throw std::exception();
        cur->second.allocated = true;
        total_size += size;

        if (p != device_map.end()) {
            // Try merge sections.
            // Continueous memory sections.
            if ((uint8_t*)address + size != (uint8_t*)p->first) throw std::exception();
            // Not-allocated memory section.
            if (p->second.allocated == false) return address;
            cur->second.size += p->second.size;
            device_map.erase(p);
        }

        return address;
    }

    ~DemoMemoryManager() {}
};  // struct DemoMemoryManager