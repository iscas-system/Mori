#pragma once

namespace mori {

struct status_exception : public std::exception {
    status_exception() = default;
    status_exception(const status_exception& exception) = default;

    status_exception& operator=(const status_exception& exception) = default;
    virtual const char* what() const throw() {
        return "Status exception.";
    }

    virtual ~status_exception() = default;
};  // struct status_exception

struct uninited_exception : public status_exception {
    virtual const char* what() const throw() {
        return "Not inited.";
    }
};  // struct uninited_exception

struct inited_exception : public status_exception {
    virtual const char* what() const throw() {
        return "Already inited.";
    }
};  // struct inited_exception

struct status_error : public status_exception {
    std::string reason;

    status_error(const std::string& _reason): reason(_reason) {}
    status_error(const status_error& exception) = default;

    status_error& operator=(const status_error& exception) = default;

    virtual const char* what() const throw() {
        return reason.c_str();
    }
};  // struct status_error

}   // namespace mori