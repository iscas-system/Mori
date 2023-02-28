#pragma once

namespace mori {

struct event_exception : public std::exception {
    event_exception() = default;
    event_exception(const event_exception& exception) = default;

    event_exception& operator=(const event_exception& exception) = default;
    virtual const char* what() const throw() {
        return "Event exception.";
    }

    virtual ~event_exception() = default;
};  // struct event_exception

struct event_conflict : public event_exception {
protected:
    std::string reason;
    
public:
    event_conflict(const std::string _reason): reason(_reason) {}
    event_conflict(const event_conflict& exception) = default;

    event_conflict& operator=(const event_conflict& exception) = default;
    virtual const char* what() const throw() {
        return reason.c_str();
    }

    virtual ~event_conflict() = default;
};  // struct event_conflict

}   // namespace mori