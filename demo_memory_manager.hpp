#include <map>
#include <utility>
#include <exception>
#include <mutex>
#include <cstdlib>

#include "libmori.hpp"

struct LoggingMemoryManager : public mori::MemoryManager {
    virtual void* allocate(size_t size) {
        std::cout<<size<<" allocated.\n";
        return reinterpret_cast<void*>(10086);
    }
    virtual void* copyIn(void* address, size_t size) {
        std::cout<<size<<" copied in.\n";
        return reinterpret_cast<void*>(10086);
    }
    virtual void* copyOut(void* address, size_t size) {
        std::cout<<size<<" copied out.\n";
        return reinterpret_cast<void*>(10001);
    }
    virtual void freeDevice(void* address) {
        std::cout<<address<<" freed on device.\n";
    }
    virtual void freeHost(void* address) {
        std::cout<<address<<" freed on host.\n";
    }
};  // struct LoggingMemoryManager

struct DemoMemoryManager : public mori::MemoryManager{
    std::mutex m;

    void* host_memory_pool;

    std::map<void*, size_t> host_map;
    std::map<void*, size_t> device_map;

    size_t total_size = 0;

    DemoMemoryManager() {
        // 4 MB
        host_memory_pool = malloc(1024 * 4);
    }

    virtual void* allocate(size_t size) {
        std::unique_lock<std::mutex>{m};

        // Assume 2 MB of device memory.
        if (total_size + size > 1024*2) return nullptr;

        // Sufficient device memory
        void* ret_address = malloc(size);
        device_map.insert(std::make_pair(ret_address, size));

        total_size += size;
        return ret_address;
    }

    virtual void* copyIn(void* address, size_t size) {
        std::unique_lock<std::mutex>{m};

        // Check if allocated host memory.
        auto p = host_map.find(address);
        // Host memory not allocated.
        if (p == host_map.end()) throw std::exception();
        // Not tensor size
        if (p->second != size) throw std::exception();

        // Allocate device memory and copy data.
        void* device_address = malloc(size);
        std::memcpy(device_address, address, size);
        device_map.insert(std::make_pair(device_address, size));
        total_size += size;

        return device_address;
    }

    virtual void* copyOut(void* address, size_t size) {
        std::unique_lock<std::mutex>{m};

        // Check if allocated device memory.
        auto p = device_map.find(address);
        // Device memory not allocated.
        if (p == device_map.end()) throw std::exception();
        // Not tensor size
        if (p->second != size) throw std::exception();

        // Allocate host memory and copy data.
        void* host_address = malloc(size);
        std::memcpy(host_address, address, size);
        host_map.insert(std::make_pair(host_address, size));

        return host_address;
    }

    virtual void freeDevice(void* address) {
        std::unique_lock<std::mutex>{m};

        // Check if allocated device memory.
        auto p = device_map.find(address);
        // Device memory not allocated.
        if (p == device_map.end()) throw std::exception();

        std::free(p->first);
        total_size -= p->second;
        device_map.erase(p);
    }

    virtual void freeHost(void* address) {
        std::unique_lock<std::mutex>{m};

        // Check if allocated host memory.
        auto p = host_map.find(address);
        // Device memory not allocated.
        if (p == host_map.end()) throw std::exception();

        std::free(p->first);
        host_map.erase(p);
    }

    ~DemoMemoryManager() {
        std::free(host_memory_pool);
    }
};  // struct DemoMemoryManager