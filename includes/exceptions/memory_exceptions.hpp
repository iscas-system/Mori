#pragma once

#include <string>

#include "includes/utils.hpp"

namespace mori {

struct memory_exception : public std::exception {
protected:
    std::string reason = "Memory exception.";

public:
    memory_exception() = default;
    memory_exception(const std::string& _reason): reason(_reason) {}
    memory_exception(const memory_exception& exception) = default;

    memory_exception(void* address) {
        reason = utils::make_pointer_string_hex(address);
        reason += ": Memory exception.";
    }
    memory_exception(void* address, const std::string& _reason) {
        reason = utils::make_pointer_string_hex(address);
        reason += _reason;
    }

    memory_exception& operator=(const memory_exception& exception) = default;
    virtual const char* what() const throw() override {
        return reason.c_str();
    }

    virtual ~memory_exception() = default;
};  // struct memory_exception

struct memory_insufficience : public memory_exception {
protected:
    size_t size;

public:
    memory_insufficience(const std::string& _reason, size_t _size): memory_exception(_reason), size(_size) {}
    virtual const char* what() const noexcept override {
        return reason.c_str();
    }
    virtual size_t demand() const { return size; }
};  // struct memory_insufficience

struct memory_host_insufficience : public memory_insufficience {
    memory_host_insufficience(const std::string& _reason, size_t _size): memory_insufficience(_reason, _size) {}

    virtual ~memory_host_insufficience() = default;
};  // struct memory_host_insufficience

struct memory_device_insufficience : public memory_insufficience {
    memory_device_insufficience(const std::string& _reason, size_t _size): memory_insufficience(_reason, _size) {}

    virtual ~memory_device_insufficience() = default;
};  // struct memory_device_insufficience

struct memory_allocated : public memory_exception {
    memory_allocated() { reason = "Memory already allocated."; }
    memory_allocated(const std::string& _reason): memory_exception(_reason) {}
    memory_allocated(void* address): memory_exception(address, ": Memory already allocated.") {}
    memory_allocated(void* address, const std::string& _reason): memory_exception(address, _reason) {}
    virtual ~memory_allocated() = default;
};  // struct memory_allocated

struct memory_not_allocated : public memory_exception {
    memory_not_allocated() { reason = "Memory not allocated."; }
    memory_not_allocated(const std::string& _reason): memory_exception(_reason) {}
    memory_not_allocated(void* address): memory_exception(address, ": Memory not allocated.") {}
    memory_not_allocated(void* address, const std::string& _reason): memory_exception(address, _reason) {}
    virtual ~memory_not_allocated() = default;
};  // struct memory_not_allocated

struct memory_operation_invalid : public memory_exception {
    memory_operation_invalid() { reason = "Memory operation invalid."; }
    memory_operation_invalid(const std::string& _reason): memory_exception(_reason) {}
    memory_operation_invalid(void* address): memory_exception(address, ": Memory operation invalid.") {}
    memory_operation_invalid(void* address, const std::string& _reason): memory_exception(address, _reason) {}
    virtual ~memory_operation_invalid() = default;
};  // struct memory_operation_invalid

struct memory_unmanaged : public memory_exception {

};  // struct memory_unmanaged

}   // namespace mori