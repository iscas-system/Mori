#pragma once

#include <cassert>

#include "frontend/memory_manager.hpp"
#include "includes/memory_status.hpp"

namespace mori {

struct MemoryOperationExecutor final {
private:
    MemoryManager* memory_manager = nullptr;

protected:
    void relocate(status::TensorPres& tensor) {
        void* device_address = memory_manager->allocateDevice(tensor.getSize());
        if (device_address == nullptr) {
            if (tensor.getDeviceSize() != 0) swapOut(tensor, tensor.getDeviceSize());
            assert(tensor.getDeviceSize() == 0);
            device_address = memory_manager->allocateDevice(tensor.getSize());
            if (device_address == nullptr) throw memory_device_insufficience("Relocation of tensor failed.", tensor.getSize());
        }
        
        const status::MemorySection* section = &(tensor.getSection(0));
        do {
            switch (section->status) {
                case status::MemoryStatusType::empty:
                    memory_manager->freeDevice(section->device_address);
                    tensor.setDeviceFreed();
                case status::MemoryStatusType::none:
                    tensor.setCopiedIn(section->offset, device_address);
                    break;
                case status::MemoryStatusType::host:
                    memory_manager->copyIn(section->host_address, device_address, section->size);
                    tensor.setCopiedIn(section->offset, device_address);
                    break;
                default:    // coexist device
                    memory_manager->copyDevice(section->device_address, device_address, section->size);
                    memory_manager->freeDevice(section->device_address);
                    tensor.setMoved(section->offset, device_address);
                    break;
            }
            device_address = (uint8_t*)device_address + section->size;

            const status::MemorySection* section_prev = section->prev();
            if (section_prev != nullptr) {
                if (tensor.isMergeable(section_prev->offset)) {
                    // The sections are already in continuous memory region, hence only update memory section information.
                    tensor.merge(section_prev->offset);
                    section = section_prev;
                } else {
                    memory_manager->split(section_prev->device_address, section_prev->size);
                }
            }
            section = section->next();
        } while (section != nullptr);
    }

public:
    MemoryOperationExecutor() {}

    inline void setMemoryManager(MemoryManager* _memory_manager) { memory_manager = _memory_manager; };

    // inline bool isMemorySectionSupported() const { return memory_manager->isMemorySectionSupported(); }

    void allocate(status::TensorPres& tensor) {        
        void* device_address = memory_manager->allocate(tensor.getSize());
        if (device_address == nullptr) throw memory_device_insufficience("Device memory insufficient.", tensor.getSize());
        tensor.setAllocated(device_address);
    }

    void copyIn(status::TensorPres& tensor, size_t size) {
        if (tensor.getSize() < size) throw status::tensor_invalid("Copying in size larger than tensor size.");
        size_t copied_size = 0;
        const status::MemorySection* section = &(tensor.getSection(0));
        do {
            switch(section->status) {
                case status::MemoryStatusType::none:
                case status::MemoryStatusType::host: {
                    void* device_address = nullptr;
                    // Allocate this section
                    if (memory_manager->isMemorySectionSupported()) {
                        // Make a quick path here.
                        if (tensor.getSize() == size && tensor.getDeviceSize() == 0) {
                            // If copying size equals to tensor size, and tensor currently has no data on device, the whole tensor must have been swapped out before and be copied in now.
                            // Hence, just relocate this tensor.
                            relocate(tensor);
                            return;
                        }
                        // There may be a number of sections, and the section should be smaller or equal to the specified size.
                        device_address = memory_manager->salloc(section->device_address, section->size);
                        if (device_address == nullptr) {
                            // Allocate memory for the tensor and copy in all the data.
                            relocate(tensor);
                            return;
                        }
                        assert(device_address == section->device_address);
                    } else {
                        // There must be only one section, and the section size is equal to the tensor size.
                        assert(size == tensor.getSize());
                        assert(size == section->size);
                        device_address = memory_manager->allocateDevice(section->size);
                    }
                    // Less possible to happen since copying in usually takes place in backward propagation, while the peak memory usage is gone through.
                    if (device_address == nullptr) throw memory_device_insufficience("Device memory insufficient.", section->size);
                    
                    if (section->status == status::MemoryStatusType::host) {
                        // Copy in this section.
                        memory_manager->copyIn(section->host_address, device_address, section->size);
                        tensor.setCopiedIn(section->offset, device_address);
                        // This tensor's status will be coexist.
                        assert(section->status == status::MemoryStatusType::coexist);
                    } else {
                        tensor.setCopiedIn(section->offset, device_address);
                        // This tensor's status will be empty.
                        assert(section->status == status::MemoryStatusType::empty);
                    }

                    // Process memory section merging.
                    if (memory_manager->isMemorySectionSupported()) {
                        if (tensor.isMergeable(section->offset)) {
                            assert(memory_manager->merge(device_address, (uint8_t*)device_address + section->size));
                            tensor.merge(section->offset);
                        }
                        const status::MemorySection* section_prev = section->prev();
                        if (section_prev != nullptr && tensor.isMergeable(section_prev->offset)) {
                            assert(memory_manager->merge(section_prev->device_address, device_address));
                            tensor.merge(section_prev->offset);
                            section = section_prev;
                        }
                    }
                }
                case status::MemoryStatusType::coexist:
                case status::MemoryStatusType::empty:
                    copied_size += section->size;
                    if (copied_size >= size) return;
                default:
                    break;
            }

            section = section->next();
        } while (section != nullptr);
    }

    void copyOut(status::TensorPres& tensor, size_t size) {
        if (tensor.getSize() < size) throw status::tensor_invalid("Copying out size larger than tensor size.");
        size_t copied_size = 0;
        const status::MemorySection* section = &(tensor.getSection(0));
        do {
            switch(section->status) {
                case status::MemoryStatusType::device: {
                    if (memory_manager->isMemorySectionSupported()) {
                        if (copied_size + section->size > size) {
                            memory_manager->split(section->device_address, size - copied_size);
                            tensor.split(section->offset, size - copied_size);
                        }
                    }
                    void* host_address = memory_manager->allocateHost(section->size);
                    if (host_address == nullptr) throw memory_host_insufficience("Host memory insufficient.", section->size);
                    memory_manager->copyOut(section->device_address, host_address, section->size);
                    tensor.setCopiedOut(section->offset, host_address);
                    // This tensor's status will be coexist.
                    assert(section->status == status::MemoryStatusType::coexist);

                    copied_size += section->size;
                    if (copied_size >= size) return;
                }
                case status::MemoryStatusType::coexist:
                case status::MemoryStatusType::empty:
                    // Empty can be regarded as a new kind of 'coexist', the data on device is empty (not assigned), the data on host is empty (do not need host memory space). 
                default:
                    break;
            }
            section = section->next();
        } while (section != nullptr);
    }

    void freeDevice(status::TensorPres& tensor, size_t size) {
        if (tensor.getSize() < size) throw status::tensor_invalid("Freeing size larger than tensor size.");
        size_t freed_size = 0;
        const status::MemorySection* section = &(tensor.getSection(0));
        do {
            switch (section->status) {
                case status::MemoryStatusType::device:
                case status::MemoryStatusType::coexist:
                case status::MemoryStatusType::empty: {
                    memory_manager->freeDevice(section->device_address);
                    tensor.setDeviceFreed(section->offset);

                    if (tensor.isMergeable(section->offset)) tensor.merge(section->offset);
                    const status::MemorySection* section_prev = section->prev();
                    if (section_prev != nullptr && tensor.isMergeable(section_prev->offset)) tensor.merge(section->offset);

                    freed_size += section->size;
                    if (freed_size >= size) return;
                    break;
                }
                default:
                    break;
            }
            section = section->next();
        } while (section != nullptr);
    }

    void freeHost(status::TensorPres& tensor, size_t size) {
        if (tensor.getSize() < size) throw status::tensor_invalid("Freeing size larger than tensor size.");
        size_t freed_size = 0;
        const status::MemorySection* section = &(tensor.getSection(0));
        do {
            switch (section->status) {
                case status::MemoryStatusType::host:
                case status::MemoryStatusType::coexist: {
                    memory_manager->freeHost(section->host_address);
                    tensor.setHostFreed(section->offset);
                    freed_size += section->size;
                    if (tensor.isMergeable(section->offset)) {
                        if (section->status != status::MemoryStatusType::none) assert(memory_manager->merge(section->device_address, section->next()->device_address));
                        tensor.merge(section->offset);
                    }
                    const status::MemorySection* section_prev = section->prev();
                    if (section_prev != nullptr && tensor.isMergeable(section_prev->offset)) {
                        if (section->status != status::MemoryStatusType::none) assert(memory_manager->merge(section_prev->device_address, section->device_address));
                        tensor.merge(section_prev->offset);
                        section = section_prev;
                    }
                    if (freed_size >= size) return;
                    break;
                }
                default:
                    break;
            }
            section = section->next();
        } while (section != nullptr);
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