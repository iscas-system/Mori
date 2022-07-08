#pragma once

#include "stdlibs.hpp"

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

    operator std::string() const {
        std::string typestr;
        switch (type) {
            case allocate:
                typestr = "allocate";
                break;
            case write:
                typestr = "write";
                break;
            case read:
                typestr = "read";
                break;
            case access:
                typestr = "access";
                break;
            case free:
                typestr = "free";
                break;
            default:
                typestr = "access";
                break;
        }

        std::stringstream ss;
        ss<<"Timestamp: "<<std::chrono::duration_cast<std::chrono::milliseconds>(timestamp.time_since_epoch()).count()<<" operator: "<<op<<" tensor: "<<tensor<<" type: "<<typestr;
        return ss.str();
    }

};  // struct MemoryEvents

static MemoryEvent make_memory_event(const std::string& _op, const std::string& _tensor, MemoryEventType _type) {
    return MemoryEvent(_op, _tensor, _type);
}

}   // namespace mori