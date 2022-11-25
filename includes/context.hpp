#pragma once

#include "includes/exceptions.hpp"

namespace mori {

struct Context final {
public:
    struct View final {
    protected:
        friend struct Context;
        
    protected:
        const Context& context;
        std::string prefix;

        View(const Context& _context, const std::string& _prefix): context(_context), prefix(_prefix) {}

        std::string make_target_key(const std::string& key) const {
            std::string target_key = prefix;
            if (key != "") target_key = prefix + "." + key;
            return target_key;
        }

    public:
        const std::string& at(const std::string& key) const { return context.at(make_target_key(key)); }
        const std::string& at() const { return context.at(prefix); }
        bool signal(const std::string& key) const { return context.signal(make_target_key(key)); }
        bool isParamExists(const std::string& key) const { return context.isParamExists(make_target_key(key)); }
        bool isDefaultParam(const std::string& key) const { return context.isDefaultParam(make_target_key(key)); }

        View view(const std::string& _prefix) const {
            std::string target = prefix;
            if (_prefix != "") {
                target.push_back('.');
                target += _prefix;
            }

            return View(context, target);
        }
    };  // struct ContextView

protected:
    std::unordered_map<std::string, std::string> defaults;
    std::unordered_map<std::string, std::string> contexts;

    void prepareDefaultParams() {
        defaults.emplace("path", "int://local");
        defaults.emplace("scheduler", "fifo");
        defaults.emplace("scheduler.trigger_event", "dependency");

        defaults.emplace("exporters.events", "empty");
        defaults.emplace("exporters.events.method", "empty");
        defaults.emplace("exporters.tensors", "empty");
        defaults.emplace("exporters.tensors.method", "empty");
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

    bool signal(const std::string& key) const {
        return at(key) == "1" ? true : false;
    }

    bool isParamExists(const std::string& key) const {
        auto p = contexts.find(key);
        if (p != contexts.end()) return true;

        p = defaults.find(key);
        if (p != defaults.end()) return true;

        return false;
    }

    bool isDefaultParam(const std::string& key) const {
        auto p = contexts.find(key);
        if (p != contexts.end()) return false;

        p = defaults.find(key);
        if (p != defaults.end()) return true;

        return false;
    }

    View view(const std::string& prefix) const {
        return View(*this, prefix);
    }

    friend std::istream& operator>>(std::istream& in, const Context& context) {
        return in;
    }

    friend std::ostream& operator<<(std::ostream& out, const Context& context) {
        return out;
    }

};  // struct Context

}   // namespace mori