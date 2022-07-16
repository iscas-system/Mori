#pragma once

namespace mori {

struct context_exception : public std::exception {
    context_exception() = default;
    context_exception(const context_exception& exception) = default;

    context_exception& operator=(const context_exception& exception) = default;
    virtual const char* what() const throw() {
        return "Context exception.";
    }
};  // struct status_exception

struct context_missing : public context_exception {
protected:
    std::string reason = "Context missing: ";
    
public:
    context_missing(const std::string& parameter) {
        reason += parameter;
    }
    context_missing(const context_missing& exception) = default;

    context_missing& operator=(const context_missing& exception) = default;
    virtual const char* what() const throw() {
        return reason.c_str();
    }

    ~context_missing() = default;
};  // struct context_missing

struct context_invalid : public context_exception {
protected:
    std::string reason = "Context invalid: ";

public:
    context_invalid(const std::string& parameter) {
        reason += parameter;
    }
    context_invalid(const context_invalid& exception) = default;

    context_invalid& operator=(const context_invalid& exception) = default;
    virtual const char* what() const throw() {
        return reason.c_str();
    }

    ~context_invalid() = default;
};  // struct context_invalid

}   // namespace mori