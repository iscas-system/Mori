#pragma once

#include <cmath>
#include <sstream>

namespace mori {
namespace utils {

inline static void* address_offset(void* address, size_t size) {
    return (uint8_t*)address + size;
}

// inline static size_t get_memory_block(void* address, size_t size) {
//     return reinterpret_cast<size_t>(address) / size;
// }

// inline static void* get_block_base_address(size_t block, size_t size) {
//     return reinterpret_cast<void*>(block * size);
// }

inline static size_t get_memory_aligned_size(size_t size, size_t alignment) {
    if (size == 0) return 0; 
    return ((size - 1) / alignment + 1) * alignment;
}

inline static bool memory_address_aligned(void* address, size_t alignment) {
    return (reinterpret_cast<size_t>(address) % alignment) == 0;
}

inline static std::string make_pointer_string_hex(void* address) {
    std::stringstream ss;
    ss << std::hex << reinterpret_cast<size_t>(address);
    return ss.str();
}

}   // namespace utils
}   // namespace mori