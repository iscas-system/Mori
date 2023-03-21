#pragma once

#include <string>
#include <sstream>
#include <chrono>
#include <cassert>

#include "includes/application_stage.hpp"

namespace mori {
namespace events {

enum class MemoryEventType {
    allocate, write, read, access, swapin, swapout, free, reshape
};  // enum MemoryEventType

namespace util {
    static std::string get_event_type_str(MemoryEventType type) {
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
            case MemoryEventType::reshape:
                return "reshape";
        }

        assert(0);
        return "";
    }

    static long get_timestamp_val(const std::chrono::steady_clock::time_point& timestamp) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(timestamp.time_since_epoch()).count();
    }
}   // namespace util

struct MemoryEvent final {
    std::string op;
    std::string tensor;
    size_t size;
    MemoryEventType type;
    ApplicationStage stage;
    std::chrono::steady_clock::time_point timestamp;

    MemoryEvent() {
        op = "";
        tensor = "";
        size = 0;
        type = MemoryEventType::access;
        stage = ApplicationStage::all;
        timestamp = std::chrono::steady_clock::now();
    }

    MemoryEvent(const std::string& _op, const std::string& _tensor, size_t _size, MemoryEventType _type, ApplicationStage _stage, const std::chrono::steady_clock::time_point& _timestamp) {
        op = _op;
        tensor = _tensor;
        size = _size;
        type = _type;
        stage = _stage;
        timestamp = _timestamp;
    }

    MemoryEvent(const std::string& _op, const std::string& _tensor, size_t _size, MemoryEventType _type, ApplicationStage _stage) {
        op = _op;
        tensor = _tensor;
        size = _size;
        type = _type;
        stage = _stage;
        timestamp = std::chrono::steady_clock::now();
    } 

    MemoryEvent(const MemoryEvent& event) = default;
    MemoryEvent& operator=(const MemoryEvent& event) = default;

    bool operator<(const MemoryEvent& event) const {return timestamp < event.timestamp;}

    operator std::string() const {
        std::stringstream ss;
        ss<<"Timestamp: "<<util::get_timestamp_val(timestamp)<<" operator: "<<op<<" tensor: "<<tensor<<" size: "<<size<<" type: "<<util::get_event_type_str(type)<<" stage: "<<mori::util::get_application_stage_str(stage);
        return ss.str();
    }

};  // struct MemoryEvents

}   // namespace events
}   // namespace mori