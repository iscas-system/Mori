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

struct memory_host_insufficience : public memory_exception {
    memory_host_insufficience() = default;
    memory_host_insufficience(const memory_host_insufficience& exception) = default;

    memory_host_insufficience& operator=(const memory_host_insufficience& exception) = default;
    virtual const char* what() const throw() {
        return "Memory bad alloc.";
    }

    virtual ~memory_host_insufficience() = default;
};  // struct memory_host_insufficience

struct memory_device_insufficience : public memory_exception {
    memory_device_insufficience() = default;
    memory_device_insufficience(const memory_device_insufficience& exception) = default;

    memory_device_insufficience& operator=(const memory_device_insufficience& exception) = default;
    virtual const char* what() const throw() {
        return "Memory bad alloc.";
    }

    virtual ~memory_device_insufficience() = default;
};  // struct memory_device_insufficience

struct memory_unmanaged : public memory_exception {

};  // struct memory_unmanaged

}   // namespace mori