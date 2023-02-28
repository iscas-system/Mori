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

struct memory_insufficience : public memory_exception {
protected:
    std::string reason;
    size_t size;

public:
    memory_insufficience(const std::string& _reason, size_t _size): reason(_reason), size(_size) {}
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

struct memory_unmanaged : public memory_exception {

};  // struct memory_unmanaged

}   // namespace mori