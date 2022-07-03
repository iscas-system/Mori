#pragma once

#include <string>
#include <chrono>

namespace mori {

enum MemoryEventType {
    allocate, write, read, access, free
};  // enum MemoryEventType

struct MemoryEvent {
    std::string op;
    std::string tensor;
    MemoryEventType type;
    std::chrono::steady_clock::time_point timestamp;

    MemoryEvent() {
        op = "";
        tensor = "";
        type = MemoryEventType::access;
        timestamp = std::chrono::steady_clock::now();
    }

    MemoryEvent(const std::string& _op, const std::string& _tensor, MemoryEventType _type, const std::chrono::steady_clock::time_point& _timestamp) {
        op = _op;
        tensor = _tensor;
        type = _type;
        timestamp = _timestamp;
    }

    MemoryEvent(const std::string& _op, const std::string& _tensor, MemoryEventType _type) {
        op = _op;
        tensor = _tensor;
        type = _type;
        timestamp = std::chrono::steady_clock::now();
    } 

    MemoryEvent(const MemoryEvent& event) = default;
    MemoryEvent& operator=(const MemoryEvent& event) = default;

    bool operator<(const MemoryEvent& event) const {return timestamp < event.timestamp;}

};  // struct MemoryEvents

static MemoryEvent make_memory_event(const std::string& _op, const std::string& _tensor, MemoryEventType _type) {
    return MemoryEvent(_op, _tensor, _type);
}

}   // namespace mori