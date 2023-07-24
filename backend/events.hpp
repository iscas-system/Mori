#pragma once

#include <functional>
#include <string>
#include <set>
#include <map>
#include <utility>
#include <cassert>

#include "backend/exporters.hpp"
#include "includes/memory_event.hpp"

namespace mori {
namespace events {

template <typename T>
struct EventSet;

struct Events final {
private:
    friend struct EventSet<MemoryEvent>;
    friend struct EventSet<ExecutionEvent>;

private:
    int iteration = 0;
    // key: iteration, value: MemoryEvent / ExecutionEvent
    std::multimap<int, MemoryEvent> memory_events;
    std::multimap<int, ExecutionEvent> execution_events;

public:
    Events() = default;

    void submitEvent(const MemoryEvent& event) {
        memory_events.emplace(iteration, event);
    }
    void submitEvent(const ExecutionEvent& event) {
        execution_events.emplace(iteration, event);
    }

    EventSet<MemoryEvent> from_memory_events() const;
    EventSet<ExecutionEvent> from_execution_events() const;

    int getIteration() const noexcept { return iteration; }
    void setIteration(int _iteration) noexcept { iteration = _iteration; }
    void newIteration() noexcept { ++iteration; }

    ~Events() = default;

};  // struct Events

template <typename T>
struct EventSet final {
private:
    friend Events;

private:
    using event_base = std::multimap<int, T>;
    using event_iter = typename event_base::const_iterator;

    struct Comparator final {
        bool operator()(const event_iter& p, const event_iter& q) const {
            if (p->first == q->first) return p->second.timestamp < q->second.timestamp;
            return p->first < q->first;
        }
    };  // struct Comparator

public:
    using item = std::pair<int, T>;
    using pred = std::function<bool(const item&)>;
    using res  = std::set<event_iter, Comparator>;

private:
    const event_base&                events_base;
    std::set<event_iter, Comparator> events_cond;
    std::vector<pred>                preds;

    bool first_query = true;

private:
    EventSet(const event_base& events): events_base(events) {}

public:
    EventSet(const EventSet&) = default;
    EventSet(EventSet&&)      = default;

    EventSet select() { return *this; }

    EventSet& where(const std::function<bool(const std::pair<int, T>&)> f) {
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

    inline const res& ref() const noexcept { return events_cond; }

    inline size_t size() const noexcept { 
        if (first_query) return events_base.size();
        return events_cond.size();
    }

    inline bool empty() const noexcept { return events_cond.empty(); }

    inline void clear() {
        events_cond.clear();
        first_query = true;
    }

    ~EventSet() = default;
};  // struct EventSet

inline EventSet<MemoryEvent> Events::from_memory_events() const {
    return EventSet<MemoryEvent>{this->memory_events};
}

inline EventSet<ExecutionEvent> Events::from_execution_events() const {
    return EventSet<ExecutionEvent>{this->execution_events};
}

template <typename T>
inline static EventSet<T> select(const EventSet<T>& event_set) {
    return event_set;
}

}   // namespace events
}   // namespace mori