#pragma once

#include <functional>
#include <string>
#include <set>
#include <map>
#include <utility>
#include <cassert>

#include "backend/events_exporter.hpp"
#include "includes/memory_event.hpp"

namespace mori {
namespace events {

struct EventSet;

struct Events final {
private:
    friend struct EventSet;

private:
    int iteration = 0;
    // key: iteration, value: MemoryEvent
    std::multimap<int, MemoryEvent> events;

public:
    Events() = default;

    void submitEvent(const MemoryEvent& event) {
        events.emplace(iteration, event);
    }

    EventSet select() const;

    int getIteration() const noexcept { return iteration; }
    void setIteration(int _iteration) noexcept { iteration = _iteration; }
    void newIteration() noexcept { ++iteration; }

    ~Events() = default;

};  // struct Events

struct EventSet final {
private:
    using event_iter = std::multimap<int, MemoryEvent>::const_iterator;

    struct Comparator final {
        bool operator()(const event_iter& p, const event_iter& q) const {
            if (p->first == q->first) return p->second.timestamp < q->second.timestamp;
            return p->first < q->first;
        }
    };  // struct Comparator

public:
    using item = std::pair<int, MemoryEvent>;
    using pred = std::function<bool(const item&)>;
    using res  = std::set<event_iter, Comparator>;

private:
    const std::multimap<int, MemoryEvent>& events_base;
    std::set<event_iter, Comparator>       events_cond;
    std::vector<pred>                      preds;

    bool first_query = true;

public:
    EventSet(const Events& events): events_base(events.events) {}

    EventSet(const EventSet&) = default;
    EventSet(EventSet&&)      = default;

    EventSet select() { return *this; }

    EventSet& where(const std::function<bool(const std::pair<int, MemoryEvent>&)> f) {
        preds.push_back(f);
        return *this;
    }

    EventSet& get() {
        auto p = preds.begin();

        if (first_query) {
            assert(events_cond.empty());
            auto q = events_base.begin();
            while (q != events_base.end()) {
                if ((*p)(*q)) events_cond.insert(q);
                ++q;
            }
            ++p;
            first_query = false;
        }

        while (p != preds.end()) {
            auto q = events_cond.begin();
            while (q != events_cond.end()) {
                if ((*p)(**q)) ++q;
                else q = events_cond.erase(q);
            }
            ++p;
        }

        preds.clear();

        return *this;
    }

    const res& ref() const noexcept { return events_cond; }

    size_t size() const noexcept { 
        if (first_query) return events_base.size();
        return events_cond.size();
    }

    void clear() {
        events_cond.clear();
        first_query = true;
    }

    ~EventSet() = default;
};  // struct EventSet

inline EventSet Events::select() const {
    return EventSet(*this);
}

/**
 * from
 * Form a from(xxx).select(xxx).where(xxx) query.
 */
static inline EventSet select_from(const EventSet& events) {
    return events;
}

static inline EventSet select_from(const Events& events) {
    return events.select();
}

}   // namespace events
}   // namespace mori