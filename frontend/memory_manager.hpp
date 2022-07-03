#pragma once

namespace mori {

struct MemoryManager {
    virtual void* allocate(size_t size) =0;

    virtual void* copyIn(void* address, size_t size) =0;
    virtual void* copyOut(void* address, size_t size) =0;

    virtual void freeDevice(void* address) = 0;
    virtual void freeHost(void* address) = 0;

    virtual void* swapIn(void* address, size_t size) {
        void* re = copyIn(address, size);
        freeHost(address);
        return re;
    }

    virtual void* swapOut(void* address, size_t size) {
        void* re = copyOut(address, size);
        freeDevice(address);
        return re;
    }

    virtual void free(void* address) {freeDevice(address);}
    
    virtual ~MemoryManager() {}
};

}   //namespace mori