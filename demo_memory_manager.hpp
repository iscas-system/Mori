#include <iostream>
#include <map>
#include <set>
#include <utility>
#include <exception>
#include <mutex>
#include <cstdlib>
#include <cassert>

#include "frontend/libmori.hpp"

#define DEVICE_MEM 2048

struct DemoMemoryManager : public mori::MemoryManager {
private:
    struct MemoryStatus {
        size_t size = 0;
        bool allocated = false;
    };  // inner struct MemoryStatus

private:
    std::recursive_mutex m;

    std::map<void*, MemoryStatus> device;
    std::set<void*> host;

public:
    DemoMemoryManager() {
        device.emplace(malloc(DEVICE_MEM), MemoryStatus{DEVICE_MEM, false});
    }

    virtual void* allocateDevice(size_t size) override {
        std::unique_lock<std::recursive_mutex>{m};

        for (auto &x : device) {
            if (x.second.allocated || x.second.size < size) continue;
            device.emplace((uint8_t*)x.first + size, MemoryStatus{x.second.size - size, false});
            x.second.allocated = true;
            x.second.size = size;
            return x.first;
        }

        return nullptr;
    }

    virtual void* allocateHost(size_t size) override {
        void* ret = malloc(size);
        host.insert(ret);
        return ret;
    }

    virtual void copyIn(void* host_address, void* device_address, size_t size) override {
        std::this_thread::sleep_for(std::chrono::milliseconds(size >> 2));
        std::cout<<size<< " copied in.\n";
    }

    virtual void copyOut(void* device_address, void* host_address, size_t size) override {
        std::this_thread::sleep_for(std::chrono::milliseconds(size >> 2));
        std::cout<<size<< " copied out.\n";
    }

    virtual void freeDevice(void* address) override {
        // Check if allocated device memory.
        auto p = device.find(address);
        // Device memory not allocated.
        if (p == device.end() || !p->second.allocated) throw std::exception();
        p->second.allocated = false;
        std::cout<<p->second.size<<" freed.\n";

        auto prev = p;
        auto post = p;
        ++post;
        --prev;
        if (post != device.end() && !post->second.allocated) {
            p->second.size += post->second.size;
            device.erase(post);
        }

        if (p != device.begin() && !prev->second.allocated) {
            prev->second.size += p->second.size;
            device.erase(p);
        }
    }

    virtual void freeHost(void* address) override {
        auto p = host.find(address);
        if (p == host.end()) throw std::exception();
        std::free(address);
        host.erase(p);
    }

    virtual bool isMemorySectionSupported() const override { return true; }

    virtual void copyDevice(void* src, void* dst, size_t size) override {
        std::this_thread::sleep_for(std::chrono::milliseconds(size >> 4));
    }

    virtual void* split(void* address, size_t size) override {
        // Check if allocated device memory.
        auto p = device.find(address);
        // Device memory not allocated.
        if (p == device.end() || !p->second.allocated) throw std::exception();

        if (p->second.size != size) {
            assert(p->second.size > size);
            device.emplace((uint8_t*)address + size, MemoryStatus{p->second.size - size, true});
            p->second.size = size;
        }

        return (uint8_t*)address + size;
    }

    virtual void* salloc(void* address, size_t size) override {
        auto p = device.begin();
        while (true) {
            if (p == device.end()) return nullptr;
            if ((uint8_t*)p->first + size < (uint8_t*)address) {
                ++p;
                continue;
            }
            if (p->first > address) return nullptr;
            if (p->second.allocated) return nullptr;
            break;
        }

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
        auto q = device.find(left);
        if (q == device.end() || !q->second.allocated) return false;
        auto p = q++;
        if (q == device.end() || q->first != right || !q->second.allocated) return false;
        if ((uint8_t*)left + p->second.size != (uint8_t*)right) return false;

        p->second.size += q->second.size;
        device.erase(q);
        return true;
    }

    ~DemoMemoryManager() { 
        assert(device.size() == 1);
        assert(device.begin()->second.size == DEVICE_MEM);
        assert(!device.begin()->second.allocated);
     }
};  // struct DemoMemoryManager