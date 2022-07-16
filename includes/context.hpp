#pragma once

#include "includes/stdlibs.hpp"
#include "includes/exceptions.hpp"

namespace mori {

struct Context final {
protected:
    std::map<std::string, std::string> defaults;
    std::map<std::string, std::string> contexts;

    std::string empty_string;

    void prepareDefaultParams() {
        defaults.insert(std::make_pair("path", "int://local"));
        defaults.insert(std::make_pair("scheduler", "fifo"));
        defaults.insert(std::make_pair("scheduler.trigger_event", "dependency"));
    }

public:
    Context() {
        prepareDefaultParams();
    }

    Context(const std::string& _path) {
        // std::ifstream fin(_path);
        // fin<<context;
        
        prepareDefaultParams();
    }

    Context(const Context& _context) {
        defaults = _context.defaults;
        contexts = _context.contexts;
    }

    Context(Context&& _context) {
        defaults = move(_context.defaults);
        contexts = move(_context.contexts);
    }

    void operator=(const Context& _context) {
        defaults = _context.defaults;
        contexts = _context.contexts;
    }

    void operator=(Context&& _context) {
        defaults = move(_context.defaults);
        contexts = move(_context.contexts);
    }

    std::string& at(const std::string& key) {
        auto p = contexts.find(key);
        if (p != contexts.end()) return p->second;

        p = defaults.find(key);
        if (p != defaults.end()) return p->second;

        throw context_missing(key);

    }

    const std::string& at(const std::string& key) const {
        auto p = contexts.find(key);
        if (p != contexts.end()) return p->second;

        p = defaults.find(key);
        if (p != defaults.end()) return p->second;

        throw context_missing(key);
    }

    std::string& operator[](const std::string& key) {
        auto p = contexts.find(key);
        if (p != contexts.end()) return p->second;

        p = defaults.find(key);
        if (p != defaults.end()) return p->second;

        return contexts[key];
    }

    const std::string& operator[](const std::string& key) const {
        auto p = contexts.find(key);
        if (p != contexts.end()) return p->second;

        p = defaults.find(key);
        if (p != defaults.end()) return p->second;

        return empty_string;
    }

    bool isParamExists(const std::string& key) {
        auto p = contexts.find(key);
        if (p != contexts.end()) return true;

        p = defaults.find(key);
        if (p != defaults.end()) return true;

        return false;
    }

    bool isDefaultParam(const std::string& key) {
        auto p = contexts.find(key);
        if (p != contexts.end()) return false;

        p = defaults.find(key);
        if (p != defaults.end()) return true;

        return false;
    }

    friend std::istream& operator>>(std::istream& in, const Context& context) {
        return in;
    }

    friend std::ostream& operator<<(std::ostream& out, const Context& context) {
        return out;
    }

};  // struct Context

}   // namespace mori