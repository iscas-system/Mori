#pragma once

#include <string>
#include <set>
#include <map>
#include <utility>

#include "../includes/memory_event.hpp"

namespace mori {

struct EventSet {
    const std::set<MemoryEvent>& events;
    const std::string select_cond;
    EventSet(const std::set<MemoryEvent>& _events): events(_events) {}

    // EventSet select(const std::string& cond) const {
    //     return EventSet();
    // }

    // EventSet where(const std::string& cond) const {
    //     return EventSet();
    // }

    // EventSet get() const {
    //     return EventSet();
    // }
};  // struct EventSet

struct EventStorage {
    int iteration = 0;
    std::multimap<int, MemoryEvent> events;

    void submitEvent(const MemoryEvent& event) {
        events.insert(std::make_pair(iteration, event));
    }

    void increaseIteration() {
        ++iteration;
    }

};  // struct EventStorage

/**
 * from
 * Forms a from(xxx).where(xxx) query.
 */
static inline EventSet from(const EventSet& events) {
    return events;
}

}   // namespace mori