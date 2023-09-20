#pragma once

#include <string>

namespace mori {

struct PerfModel {
    size_t read_speed;
    size_t write_speed;
};  // struct PerfModel

// static PerfModel create_gpu_performance_model() {
//     return PerfModel(12288, 12288);
// }
// static PerfModel create_cpu_performance_model() {
//     return PerfModel(65536, 65536);
// }
// static PerfModel create_nvm_performance_model() {
//     // CPU memory is not a limitor of memory swapping. The current spped is set to 64 GB/s.
//     return PerfModel{40960, 20480};
// }
// static PerfModel create_nvme_performance_model() {
//     return PerfModel{2560, 2048};
// }

struct MemoryInfo {
public:
    struct Block {
        void* address = nullptr;
        size_t size   = 0;
    };  // inner struct Block

    struct Device {
        std::string type;

        Block common_block;
        Block persistent_block;
        Block transient_block;

        size_t total_size    = 512;
        size_t align_size    = 512;
        size_t reserved_size = 0;
    };  // inner struct Device

    struct Host {
        std::string type;
        size_t total_size;
    };  // innter struct Host

public:
    Device device;
    Host   host;
};  // struct MemoryInfo

static MemoryInfo create_default_memory_info(size_t device, size_t host) {
    MemoryInfo re;
    re.device.type       = "gpu";
    re.device.total_size = device;
    re.device.align_size = 256;     // 256 B
    re.host.type         = "cpu";
    re.host.total_size   = host;
    return re;
}

}   // namespace mori
