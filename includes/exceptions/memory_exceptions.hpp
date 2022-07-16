#pragma once

namespace mori {

struct memory_exception : public std::exception {
    memory_exception() = default;
    memory_exception(const memory_exception& exception) = default;

    memory_exception& operator=(const memory_exception& exception) = default;
    virtual const char* what() const throw() {
        return "Memory exception.";
    }

    virtual ~memory_exception() = default;
};  // struct memory_exception

struct memory_bad_alloc : public memory_exception {
    memory_bad_alloc() = default;
    memory_bad_alloc(const memory_bad_alloc& exception) = default;

    memory_bad_alloc& operator=(const memory_bad_alloc& exception) = default;
    virtual const char* what() const throw() {
        return "Memory bad alloc.";
    }

    virtual ~memory_bad_alloc() = default;
};  // struct memory_bad_alloc

struct memory_unmanaged : public memory_exception {

};  // struct memory_unmanaged

}   // namespace mori