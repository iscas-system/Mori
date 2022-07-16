#pragma once

#include <functional>
#include <string>
#include <set>
#include <map>
#include <utility>

#include "includes/memory_event.hpp"

namespace mori {

struct EventSet {
    std::multimap<int, MemoryEvent> events;
    
    EventSet() {}
    EventSet(const std::multimap<int, MemoryEvent>& _events): events(_events) {}

    EventSet(const EventSet& event_set) {
        events = event_set.events;
    }
    EventSet(EventSet&& event_set) {
        events = std::move(event_set.events);
    }

    void operator=(const EventSet& event_set) {
        events = event_set.events;
    }

    void operator=(EventSet&& event_set) {
        events = std::move(event_set.events);
    }

    EventSet select(const std::string& cond) const {
        return EventSet();
    }

    EventSet where(const std::function<bool(const MemoryEvent&)> f) const {
        std::multimap<int, MemoryEvent> re;
        for (auto &x : events) {
            if (f(x.second)) re.insert(std::make_pair(x.first, x.second));
        }
        return EventSet(re);
    }

    EventSet where(int iter) const {
        auto p = events.lower_bound(iter);
        auto q = events.upper_bound(iter);
        if (p == q) return EventSet();
        return EventSet(std::multimap<int, MemoryEvent>(p, q));
    }

    // EventSet get() const {
    //     return EventSet();
    // }

    ~EventSet() {}
};  // struct EventSet

struct Events {
    int iteration = 0;
    std::multimap<int, MemoryEvent> events;

    void submitEvent(const MemoryEvent& event) {
        events.insert(std::make_pair(iteration, event));
    }

    void increaseIteration() {
        ++iteration;
    }

    EventSet from() const {
        return EventSet(events);
    }

    EventSet select(const std::string& cond) const {
        return from().select(cond);
    }

};  // struct Events

/**
 * from
 * Form a from(xxx).select(xxx).where(xxx) query.
 */
static inline EventSet from(const EventSet& events) {
    return events;
}

static inline EventSet from(const Events& events) {
    return events.from();
}

}   // namespace mori