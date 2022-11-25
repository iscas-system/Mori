#pragma once

#include "frontend/memory_manager.hpp"
#include "includes/memory_status.hpp"

namespace mori {

struct MemoryOperationExecutor final {
    MemoryManager* memory_manager = nullptr;

    MemoryOperationExecutor() {}

    inline void setMemoryManager(MemoryManager* _memory_manager) { memory_manager = _memory_manager; };

    void allocate(status::TensorPres& tensor) {        
        void* device_address = memory_manager->allocate(tensor.getSize());
        if (device_address == nullptr) throw memory_device_insufficience();
        tensor.setAllocated(device_address);
    }

    void copyIn(status::TensorPres& tensor) {
        void* device_address = memory_manager->copyIn(tensor.getHostPointer(0), tensor.getSectionSize(0));
        if (device_address == nullptr) throw memory_device_insufficience(); 
        tensor.setCopiedIn(device_address);
    }

    void copyOut(status::TensorPres& tensor) {
        void* host_address = memory_manager->copyOut(tensor.getDevicePointer(0), tensor.getSectionSize(0));
        if (host_address == nullptr) throw memory_host_insufficience();
        tensor.setCopiedOut(host_address);
    }

    void freeDevice(status::TensorPres& tensor) {
        memory_manager->freeDevice(tensor.getDevicePointer(0));
        tensor.setDeviceFreed();
    }

    void freeHost(status::TensorPres& tensor) {
        memory_manager->freeHost(tensor.getHostPointer(0));
        tensor.setHostFreed();
    }

    void swapIn(status::TensorPres& tensor) {
        copyIn(tensor);
        freeHost(tensor);
    }

    void swapOut(status::TensorPres& tensor) {
        copyOut(tensor);
        freeDevice(tensor);
    }

    void free(status::TensorPres& tensor) {
        assert(0);
    }

};  // struct MemoryOperationExecutor

}   // namespace mori