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

    void copyIn(status::TensorPres& tensor, size_t size) {
        if (tensor.getSize() < size) throw status::tensor_invalid("Copying in size larger than tensor size.");
        int remaining_size = size;
        for (auto &x : tensor.getSections()) {
            // There may be section merging.
            if (!tensor.isSectionExist(x)) continue;
            const status::MemorySection& section = tensor.getSection(x);
            void* device_address = nullptr;
            switch (section.status) {
                case status::MemoryStatusType::host: {
                    // Copy in this section
                    if (memory_manager->isMemorySectionSupported()) {
                        // There may be a number of sections, and the section should be smaller or equal to the specified size.
                        assert(size >= section.size);
                        std::cout<<"op_executor: " << section.size<<std::endl;
                        device_address = memory_manager->realloc(section.device_address, section.size);
                        // TODO: Support failed realloc
                        assert(device_address == section.device_address);
                    } else {
                        // There must be only one section, and the section size is equal to the tensor size.
                        assert(size == tensor.getSize());
                        assert(size == section.size);
                        device_address = memory_manager->allocateDevice(section.size);
                    }
                    memory_manager->copyIn(section.host_address, device_address, section.size);
                    // Less possible to happen since copying in usually takes place in backward propagation, while the peak memory usage is gone through.
                    if (device_address == nullptr) throw memory_device_insufficience();
                    tensor.setCopiedIn(x, section.size, device_address);
                    // This tensor's status will be coexist.
                    assert(section.status == status::MemoryStatusType::coexist);
                }
                case status::MemoryStatusType::coexist: {
                    remaining_size -= section.size;
                    if (remaining_size <= 0) return;
                    break;
                }
                default:
                    break;
            }
        }
    }

    void copyOut(status::TensorPres& tensor, size_t size) {
        if (tensor.getSize() < size) throw status::tensor_invalid("Copying out size larger than tensor size.");
        int remaining_size = size;
        for (auto &x : tensor.getSections()) {
            // There may be section merging.
            if (!tensor.isSectionExist(x)) continue;
            const status::MemorySection& section = tensor.getSection(x);
            int copying_size = 0;
            switch (section.status) {
                case status::MemoryStatusType::device: {
                    // Copy out this section
                    if (section.size <= remaining_size) copying_size = section.size;
                    else copying_size = remaining_size;
                    void* host_address = memory_manager->allocateHost(copying_size);
                    if (host_address == nullptr) throw memory_host_insufficience();
                    memory_manager->copyOut(section.device_address, host_address, copying_size);
                    tensor.setCopiedOut(x, copying_size, host_address);
                    // This tensor's status will be coexist.
                    assert(section.status == status::MemoryStatusType::coexist);
                }
                case status::MemoryStatusType::coexist: {
                    remaining_size -= copying_size;
                    if (remaining_size <= 0) return;
                    break;
                }
                default:
                    break;
            }
        }
    }

    void freeDevice(status::TensorPres& tensor, size_t size) {
        if (tensor.getSize() < size) throw status::tensor_invalid("Freeing size larger than tensor size.");
        int remaining_size = size;
        for (auto &x : tensor.getSections()) {
            // There may be section merging.
            if (!tensor.isSectionExist(x)) continue;
            const status::MemorySection& section = tensor.getSection(x);
            switch (section.status) {
                case status::MemoryStatusType::device:
                case status::MemoryStatusType::coexist:
                    if (memory_manager->isMemorySectionSupported())
                        memory_manager->freeSection(section.device_address, section.size);
                    else
                        memory_manager->freeDevice(section.device_address);
                    tensor.setDeviceFreed(x);
                    remaining_size -= section.size;
                    if (remaining_size <= 0) return;
                    break;
                default:
                    break;
            }
        }
    }

    void freeHost(status::TensorPres& tensor, size_t size) {
        if (tensor.getSize() < size) throw status::tensor_invalid("Freeing size larger than tensor size.");
        int remaining_size = size;
        for (auto &x : tensor.getSections()) {
            // There may be section merging.
            if (!tensor.isSectionExist(x)) continue;
            const status::MemorySection& section = tensor.getSection(x);
            switch (section.status) {
                case status::MemoryStatusType::host:
                case status::MemoryStatusType::coexist:
                    memory_manager->freeHost(section.host_address);
                    tensor.setHostFreed(x);
                    remaining_size -= section.size;
                    if (remaining_size <= 0) return;
                    break;
                default:
                    break;
            }
        }
    }

    void swapIn(status::TensorPres& tensor, size_t size) {
        copyIn(tensor, size);
        freeHost(tensor, size);
    }

    void swapOut(status::TensorPres& tensor, size_t size) {
        copyOut(tensor, size);
        freeDevice(tensor, size);
    }

    void free(status::TensorPres& tensor, size_t size) {
        freeDevice(tensor, size);
        freeHost(tensor, size);
    }

};  // struct MemoryOperationExecutor

}   // namespace mori