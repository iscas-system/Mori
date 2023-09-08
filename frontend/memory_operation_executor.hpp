#pragma once

#include <cassert>

#include "frontend/memory_manager.hpp"
#include "includes/memory_status.hpp"
#include "includes/memory_layout.hpp"

namespace mori {

struct MemoryOperationExecutor final {
private:
    struct MemoryOperationExecutorImpl {
        MemoryOperationExecutor& executor;
        
        MemoryOperationExecutorImpl(MemoryOperationExecutor& _executor): executor(_executor) {}

        virtual void copyIn(status::TensorPres& tensor, size_t size) = 0;
        virtual void copyOut(status::TensorPres& tensor, size_t size) = 0;
        virtual void freeDevice(status::TensorPres& tensor, size_t size) = 0;
        virtual void freeHost(status::TensorPres& tensor, size_t size) = 0;
        virtual void fragment(status::TensorPres& tensor) = 0;
        virtual void fuse(status::TensorPres& tensor) = 0;
    };  // struct MemoryOperationExecutorImpl

    struct MemoryOperationExecutorDefaultImpl final : public MemoryOperationExecutorImpl {
        MemoryOperationExecutorDefaultImpl(MemoryOperationExecutor& _executor): MemoryOperationExecutorImpl(_executor) {}

        virtual void copyIn(status::TensorPres& tensor, size_t size) override {
            if (tensor.getSize() < size) throw status::tensor_invalid("Copying in size larger than tensor size.");
            assert(tensor.getSectionCount() == 1);
            const status::MemorySection& section = tensor.getFirstSection();

            switch (section.status) {
                case status::MemoryStatusType::none:
                case status::MemoryStatusType::host: {
                    void *device_address = nullptr;
                    // Allocate this section
                    device_address = executor.memory_manager->allocateDevice(section.size);
                    if (device_address == nullptr) throw memory_device_insufficience("Device memory insufficient.", section.size);
                    executor.layout.recordMemoryAllocateEvent(device_address, section.size, tensor.getName());
                    // if (tensor.hasFragment()) {
                    //     memory_manager->split(device_address, section->size);
                    //     tensor.setFragmentPlaced((uint8_t*)device_address + section->size);
                    // }
                    // Less possible to happen since copying in usually takes place in backward propagation, while the peak memory usage is gone through.

                    if (section.status == status::MemoryStatusType::host) {
                        // Copy in this section.
                        executor.memory_manager->copyIn(section.host_address, device_address, section.size);
                        tensor.setCopiedIn(section.offset, device_address);
                        // This tensor's status will be coexist.
                        assert(section.status == status::MemoryStatusType::coexist);
                    } else {
                        tensor.setCopiedIn(section.offset, device_address);
                        // This tensor's status will be empty.
                        assert(section.status == status::MemoryStatusType::empty);
                    }
                }
                case status::MemoryStatusType::coexist:
                case status::MemoryStatusType::empty:
                default:
                    break;
            }
        }
        void copyOut(status::TensorPres& tensor, size_t size) override {
            if (tensor.getSize() < size) throw status::tensor_invalid("Copying out size larger than tensor size.");
            assert(tensor.getSectionCount() == 1);
            const status::MemorySection* section = &(tensor.getFirstSection());

            switch(section->status) {
                case status::MemoryStatusType::device: {
                    void* host_address = executor.memory_manager->allocateHost(section->size);
                    if (host_address == nullptr) throw memory_host_insufficience("Host memory insufficient.", section->size);
                    executor.memory_manager->copyOut(section->device_address, host_address, section->size);
                    tensor.setCopiedOut(section->offset, host_address);
                    // This tensor's status will be coexist.
                    assert(section->status == status::MemoryStatusType::coexist);
                }
                case status::MemoryStatusType::coexist:
                case status::MemoryStatusType::empty:
                    // Empty can be regarded as a new kind of 'coexist', the data on device is empty (not assigned), the data on host is empty (do not need host memory space). 
                default:
                    break;
            }
        }
        virtual void freeDevice(status::TensorPres& tensor, size_t size) override {
            if (tensor.getSize() < size) throw status::tensor_invalid("Freeing size larger than tensor size.");
            assert(tensor.getSectionCount() == 1);
            const status::MemorySection* section = &(tensor.getFirstSection());
            switch (section->status) {
                case status::MemoryStatusType::device:
                case status::MemoryStatusType::coexist:
                case status::MemoryStatusType::empty: {
                    executor.layout.recordMemoryFreeEvent(section->device_address);
                    executor.memory_manager->freeDevice(section->device_address);
                    tensor.setDeviceFreed(section->offset);
                }
                default:
                    break;
            }
        }
        virtual void freeHost(status::TensorPres& tensor, size_t size) override {
            if (tensor.getSize() < size) throw status::tensor_invalid("Freeing size larger than tensor size.");
            assert(tensor.getSectionCount() == 1);
            const status::MemorySection* section = &(tensor.getFirstSection());
            switch (section->status) {
                case status::MemoryStatusType::host:
                case status::MemoryStatusType::coexist:
                    executor.memory_manager->freeHost(section->host_address);
                    tensor.setHostFreed(section->offset);
                    break;
                default:
                    break;
            }
        }
        virtual void fragment(status::TensorPres& tensor) override {}
        virtual void fuse(status::TensorPres& tensor) override {}
    };  // struct MemoryOperationDefaultImpl

    struct MemoryOperationExecutorSectionedImpl final : public MemoryOperationExecutorImpl {
    protected:
        void relocate(status::TensorPres& tensor) {
            void* device_address = executor.memory_manager->allocateDevice(tensor.getSize());
            if (device_address == nullptr) {
                if (tensor.getDeviceSize() != 0) executor.swapOut(tensor, tensor.getDeviceSize());
                assert(!tensor.isDeviceLocated());
                device_address = executor.memory_manager->allocateDevice(tensor.getSize());
                if (device_address == nullptr) throw memory_device_insufficience("Relocation of tensor failed.", tensor.getSize());
            }
            executor.layout.recordMemoryAllocateEvent(device_address, tensor.getSize(), tensor.getName());
            // Remove fragment
            if (tensor.hasFragment()) {
                const status::Fragment& fragment = tensor.getFragment();
                if (tensor.getFragment().status == status::MemoryStatusType::empty) {
                    executor.layout.recordMemoryFreeEvent(fragment.address);
                    executor.memory_manager->freeDevice(fragment.address);
                    tensor.setFragmentRemoved();
                }
            }
        
            const status::MemorySection* section = &(tensor.getFirstSection());
            do {
                switch (section->status) {
                    case status::MemoryStatusType::empty:
                        executor.layout.recordMemoryFreeEvent(section->device_address);
                        executor.memory_manager->freeDevice(section->device_address);
                        tensor.setDeviceFreed(section->offset);
                    case status::MemoryStatusType::none:
                        tensor.setCopiedIn(section->offset, device_address);
                        break;
                    case status::MemoryStatusType::host:
                        executor.memory_manager->copyIn(section->host_address, device_address, section->size);
                        tensor.setCopiedIn(section->offset, device_address);
                        break;
                    case status::MemoryStatusType::coexist:
                    case status::MemoryStatusType::device:
                        executor.memory_manager->copyDevice(section->device_address, device_address, section->size);
                        executor.layout.recordMemoryFreeEvent(section->device_address);
                        executor.memory_manager->freeDevice(section->device_address);
                        tensor.setMoved(section->offset, device_address);
                    default:
                        break;
                }
                device_address = (uint8_t*)device_address + section->size;

                const status::MemorySection* section_prev = section->prev();
                if (section_prev != nullptr) {
                    if (tensor.isMergeable(section_prev->offset)) {
                        // The sections are already in continuous memory region, hence only update memory section information.
                        section = &(tensor.merge(section_prev->offset));
                    } else {
                        executor.memory_manager->split(section_prev->device_address, section_prev->size);
                        executor.layout.recordMemorySplitEvent(section_prev->device_address, section_prev->size);
                    }
                }
                section = section->next();
            } while (section != nullptr);
        }

    public:
        MemoryOperationExecutorSectionedImpl(MemoryOperationExecutor& _executor): MemoryOperationExecutorImpl(_executor) {}
        virtual void copyIn(status::TensorPres& tensor, size_t size) override {
            if (tensor.getSize() < size) throw status::tensor_invalid("Copying in size larger than tensor size.");
            size_t copied_size = 0;
            const status::MemorySection* section = &(tensor.getLastSection());
            do {
                switch(section->status) {
                    case status::MemoryStatusType::none:
                    case status::MemoryStatusType::host: {
                        void* device_address = nullptr;
                        // Allocate this section
                        // Make a quick path here.
                        if (tensor.getSize() == size && tensor.getDeviceSize() == 0) {
                            // If copying size equals to tensor size, and tensor currently has no data on device, the whole tensor must have been swapped out before and be copied in now.
                            // Hence, just relocate this tensor.
                            relocate(tensor);
                            return;
                        }
                        // There may be a number of sections, and the section should be smaller or equal to the specified size.
                        device_address = executor.memory_manager->salloc(section->device_address, section->size);
                        if (device_address == nullptr) {
                            // Allocate memory for the tensor and copy in all the data.
                            relocate(tensor);
                            return;
                        }
                        // Salloc do not need aligned allcoation.
                        executor.layout.recordMemoryAllocateEvent(device_address, section->size, tensor.getName(), 1);
                        assert(device_address == section->device_address);

                        // Less possible to happen since copying in usually takes place in backward propagation, while the peak memory usage is gone through.
                        
                        if (section->status == status::MemoryStatusType::host) {
                            // Copy in this section.
                            executor.memory_manager->copyIn(section->host_address, device_address, section->size);
                            tensor.setCopiedIn(section->offset, device_address);
                            // This tensor's status will be coexist.
                            assert(section->status == status::MemoryStatusType::coexist);
                        } else {
                            tensor.setCopiedIn(section->offset, device_address);
                            // This tensor's status will be empty.
                            assert(section->status == status::MemoryStatusType::empty);
                        }

                        // Process memory section merging.
                         if (tensor.isMergeable(section->offset)) {
                            bool res = executor.memory_manager->merge(device_address, (uint8_t*)device_address + section->size);
                            assert(res);
                            executor.layout.recordMemoryMergeEvent(device_address, (uint8_t*)device_address + section->size);
                            tensor.merge(section->offset);
                        }
                        const status::MemorySection* section_prev = section->prev();
                        if (section_prev != nullptr && tensor.isMergeable(section_prev->offset)) {
                            bool res = executor.memory_manager->merge(section_prev->device_address, device_address);
                            assert(res);
                            executor.layout.recordMemoryMergeEvent(section_prev->device_address, device_address);
                            section = &(tensor.merge(section_prev->offset));
                        }
                    }
                    case status::MemoryStatusType::coexist:
                    case status::MemoryStatusType::empty:
                        copied_size += section->size;
                    default:
                        break;
                }
                if (copied_size >= size) return;
                section = section->prev();
            } while (section != nullptr);
        }
        virtual void copyOut(status::TensorPres& tensor, size_t size) override {
            if (tensor.getSize() < size) throw status::tensor_invalid("Copying out size larger than tensor size.");
            size_t copied_size = 0;
            const status::MemorySection* section = &(tensor.getFirstSection());
            do {
                switch(section->status) {
                    case status::MemoryStatusType::device: {
                        if (copied_size + section->size > size) {
                            executor.memory_manager->split(section->device_address, size - copied_size);
                            executor.layout.recordMemorySplitEvent(section->device_address, size - copied_size);
                            tensor.split(section->offset, size - copied_size);
                        }

                        void* host_address = executor.memory_manager->allocateHost(section->size);
                        if (host_address == nullptr) throw memory_host_insufficience("Host memory insufficient.", section->size);
                        executor.memory_manager->copyOut(section->device_address, host_address, section->size);
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
        virtual void freeDevice(status::TensorPres& tensor, size_t size) override {
            if (tensor.getSize() < size) throw status::tensor_invalid("Freeing size larger than tensor size.");
            size_t freed_size = 0;
            const status::MemorySection* section = &(tensor.getFirstSection());
            do {
                switch (section->status) {
                    case status::MemoryStatusType::device:
                    case status::MemoryStatusType::coexist:
                    case status::MemoryStatusType::empty: {
                        executor.layout.recordMemoryFreeEvent(section->device_address);
                        executor.memory_manager->freeDevice(section->device_address);
                        tensor.setDeviceFreed(section->offset);

                        if (tensor.isMergeable(section->offset)) tensor.merge(section->offset);
                        const status::MemorySection* section_prev = section->prev();
                        if (section_prev != nullptr && tensor.isMergeable(section_prev->offset)) section = &(tensor.merge(section_prev->offset));

                        freed_size += section->size;
                    }
                    default:
                        break;
                }
                if (freed_size >= size) break;
                section = section->next();
            } while (section != nullptr);
            if (tensor.getDeviceSize() == 0 && tensor.hasFragment()) {
                executor.layout.recordMemoryFreeEvent(tensor.getFragment().address);
                executor.memory_manager->freeDevice(tensor.getFragment().address);
                tensor.setFragmentRemoved();
            }
        }
        virtual void freeHost(status::TensorPres& tensor, size_t size) override {
            if (tensor.getSize() < size) throw status::tensor_invalid("Freeing size larger than tensor size.");
            size_t freed_size = 0;
            const status::MemorySection* section = &(tensor.getLastSection());
            do {
                switch (section->status) {
                    case status::MemoryStatusType::host:
                    case status::MemoryStatusType::coexist: {
                        executor.memory_manager->freeHost(section->host_address);
                        tensor.setHostFreed(section->offset);
                        freed_size += section->size;
                        if (tensor.isMergeable(section->offset)) {
                            if (section->status != status::MemoryStatusType::none) {
                                bool res = executor.memory_manager->merge(section->device_address, section->next()->device_address);
                                assert(res);
                                executor.layout.recordMemoryMergeEvent(section->device_address, section->next()->device_address);
                            }
                            tensor.merge(section->offset);
                        }
                        const status::MemorySection* section_prev = section->prev();
                        if (section_prev != nullptr && tensor.isMergeable(section_prev->offset)) {
                            if (section->status != status::MemoryStatusType::none) {
                                bool res = executor.memory_manager->merge(section_prev->device_address, section->device_address);
                                assert(res);
                                executor.layout.recordMemoryMergeEvent(section_prev->device_address, section->device_address);
                            }
                            section = &(tensor.merge(section_prev->offset));
                        }
                        if (freed_size >= size) return;
                        break;
                    }
                    default:
                        break;
                }
                section = section->prev();
            } while (section != nullptr);
        }
        virtual void fragment(status::TensorPres& tensor) override {
            if (!tensor.hasFragment()) throw status::tensor_invalid("Tensor does not request fragment.");
            void* target_address = (uint8_t*)(tensor.getFirstSection().device_address) + tensor.getSize();
            void* device_address = executor.memory_manager->salloc(target_address, tensor.getFragment().size);
            if (device_address == nullptr) throw memory_exception("Allocation for fragment failed.");
            executor.layout.recordMemoryAllocateEvent(device_address, tensor.getFragment().size, tensor.getName(), 1);
            tensor.setFragmentPlaced();
        }
        virtual void fuse(status::TensorPres& tensor) override {
            if (!tensor.hasFragment()) throw status::tensor_invalid("Tensor does not request fragment.");
            executor.layout.recordMemoryFreeEvent(tensor.getFragment().address);
            executor.memory_manager->freeDevice(tensor.getFragment().address);
            tensor.setFragmentRemoved();
        }
    };  // struct MemoryOperationExecutorSectionedImpl

private:
    layout::MemoryLayout& layout;
    MemoryManager* memory_manager = nullptr;

    MemoryOperationExecutorDefaultImpl   default_impl;
    MemoryOperationExecutorSectionedImpl sectioned_impl;
    MemoryOperationExecutorImpl* impl = nullptr;

public:
    MemoryOperationExecutor(layout::MemoryLayout& _layout): layout(_layout), default_impl(*this), sectioned_impl(*this) {
        impl = &default_impl;
    }

    inline void setMemoryManager(MemoryManager* _memory_manager) { 
        memory_manager = _memory_manager;
        if (memory_manager->isMemorySectionSupported()) impl = &sectioned_impl;
    };

    // void allocate(status::TensorPres& tensor) {        
    //     void* device_address = memory_manager->allocate(tensor.getSize() + tensor.getFragment().size);
    //     if (device_address == nullptr) throw memory_device_insufficience("Device memory insufficient.", tensor.getSize());
    //     layout.recordMemoryAllocateEvent(device_address, tensor.getSize() + tensor.getFragment().size, tensor.getName());
    //     tensor.setAllocated(device_address);
    // }

    /**
     * Copy in tensor data from host memory to device memory with specific size.
     * Copy from the last section that located only on host.
     * @param tensor Tensor to be copied in
     * @param size   Size to be copied in
    */
    void copyIn(status::TensorPres& tensor, size_t size) {
        impl->copyIn(tensor, size);
    }

    /**
     * Copy out tensor data from device memory to host memory with specific size.
     * Copy from the first section that located only on device.
     * @param tensor Tensor to be copied out
     * @param size   Size to be copied out
    */
    void copyOut(status::TensorPres& tensor, size_t size) {
        impl->copyOut(tensor, size);
    }

    /**
     * Free device memory with specific size.
     * Free from the first section that located on device
     * @param tensor Tensor to be freed on device
     * @param size   Size to be freed on device
    */
    void freeDevice(status::TensorPres& tensor, size_t size) {
       impl->freeDevice(tensor, size);
    }

    /**
     * Free host memory with specific size.
     * Free from the last section that located on host
     * @param tensor Tensor to be freed on host
     * @param size   Size to be freed on host
    */
    void freeHost(status::TensorPres& tensor, size_t size) {
        impl->freeHost(tensor, size);
    }

    /**
     * Swap in tensor data from host memory to device memory with specific size.
     * Swap from the first section that located only on host.
     * @param tensor Tensor to be copied in
     * @param size   Size to be copied in
    */
    void swapIn(status::TensorPres& tensor, size_t size) {
        copyIn(tensor, size);
        freeHost(tensor, size);
    }

    /**
     * Swap out tensor data from device memory to host memory with specific size.
     * Swap from the first section that located only on device.
     * @param tensor Tensor to be copied out
     * @param size   Size to be copied out
    */
    void swapOut(status::TensorPres& tensor, size_t size) {
        copyOut(tensor, size);
        freeDevice(tensor, size);
    }

    /**
     * Free device and host memory with specific size.
     * Free from the first section that located on device or host
     * @param tensor Tensor to be freed on device and host
     * @param size   Size to be freed on device and host
    */
    void free(status::TensorPres& tensor, size_t size) {
        freeDevice(tensor, size);
        freeHost(tensor, size);
    }

    /**
     * Place fragment for the tensor
     * @param tensor Tensor to be placed fragment
    */
    void fragment(status::TensorPres& tensor) {
        impl->fragment(tensor);
    }

    /**
     * Release and fuse the fragment for the tensor
     * @param tensor Tensor to be fused
    */
    void fuse(status::TensorPres& tensor) {
        impl->fuse(tensor);
    }

};  // struct MemoryOperationExecutor

}   // namespace mori