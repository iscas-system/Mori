#pragma once

#include <string>
#include <cassert>

namespace mori {
namespace utils {
    static long get_timestamp_val(const std::chrono::steady_clock::time_point& timestamp) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(timestamp.time_since_epoch()).count();
    }

}   // namespace utils
}   // namespace mori