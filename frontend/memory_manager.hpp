#pragma once

#include "includes/memory_info.hpp"

namespace mori {

struct MemoryManager {
public:
    // Basie memory management methods.
    virtual void* allocateDevice(size_t size) = 0;
    virtual void* allocateHost(size_t size) = 0;
    virtual void* allocate(size_t size) { return allocateDevice(size); }

    virtual void copyIn(void* host_address, void* device_address, size_t size) = 0;
    virtual void copyOut(void* device_address, void* host_address, size_t size) = 0;

    virtual void freeDevice(void* address) = 0;
    virtual void freeHost(void* address) = 0;

    virtual void* swapIn(void* host_address, void* device_address, size_t size) {
        copyIn(host_address, device_address, size);
        freeHost(host_address);
        return device_address;
    }

    virtual void* swapOut(void* device_address, void* host_address, size_t size) {
        copyOut(device_address, host_address, size);
        freeDevice(device_address);
        return host_address;
    }

    virtual void free(void* address) { freeDevice(address); }

    // Memory section methods.
    virtual bool isMemorySectionSupported() const = 0;

    virtual void copyDevice(void* src, void* dst, size_t size) {
        void* host_address = allocateHost(size);
        copyOut(src, host_address, size);
        copyIn(host_address, dst, size);
        freeHost(host_address);
    }
    virtual void* split(void* address, size_t size) { return nullptr; }
    virtual void* salloc(void* address, size_t size) { return nullptr; }
    virtual bool  merge(void* left, void* right) { return false; }

    // Memory info methods.

    virtual MemoryInfo getMemoryInfo() const = 0;
    
    virtual ~MemoryManager() {}
};

}   //namespace mori