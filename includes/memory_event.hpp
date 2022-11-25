#pragma once

#include <string>
#include <sstream>
#include <chrono>

namespace mori {
namespace events {

enum class MemoryEventType {
    allocate, write, read, access, swapin, swapout, free
};  // enum MemoryEventType

namespace util {
    static std::string get_event_type_str(const MemoryEventType& type) {
        switch (type) {
            case MemoryEventType::allocate:
                return "allocate";
            case MemoryEventType::write:
                return "write";
            case MemoryEventType::read:
                return "read";
            case MemoryEventType::access:
                return "access";
            case MemoryEventType::swapin:
                return "swapin";
            case MemoryEventType::swapout:
                return "swapout";
            case MemoryEventType::free:
                return "free";
        }

        assert(0);
        return "";
    }

    static long get_timestamp_val(const std::chrono::steady_clock::time_point& timestamp) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(timestamp.time_since_epoch()).count();
    }
}   // namespace util

struct MemoryEvent final {
    std::string tensor;
    MemoryEventType type;
    std::chrono::steady_clock::time_point timestamp;

    MemoryEvent() {
        tensor = "";
        type = MemoryEventType::access;
        timestamp = std::chrono::steady_clock::now();
    }

    MemoryEvent(const std::string& _tensor, MemoryEventType _type, const std::chrono::steady_clock::time_point& _timestamp) {
        tensor = _tensor;
        type = _type;
        timestamp = _timestamp;
    }

    MemoryEvent(const std::string& _tensor, MemoryEventType _type) {
        tensor = _tensor;
        type = _type;
        timestamp = std::chrono::steady_clock::now();
    } 

    MemoryEvent(const MemoryEvent& event) = default;
    MemoryEvent& operator=(const MemoryEvent& event) = default;

    bool operator<(const MemoryEvent& event) const {return timestamp < event.timestamp;}

    operator std::string() const {
        std::string typestr = util::get_event_type_str(type);

        std::stringstream ss;
        ss<<"Timestamp: "<<util::get_timestamp_val(timestamp)<<" tensor: "<<tensor<<" type: "<<typestr;
        return ss.str();
    }

};  // struct MemoryEvents

}   // namespace events
}   // namespace mori