#pragma once

namespace mori {

struct backend_exception : public std::exception {
    backend_exception() = default;
    backend_exception(const backend_exception& exception) = default;

    backend_exception& operator=(const backend_exception& exception) = default;
    virtual const char* what() const throw() {
        return "Backend exception.";
    }

    virtual ~backend_exception() = default;
};  // struct backend_exception

struct dynamic_library_exception : public backend_exception {
protected:
    std::string reason;
    
public:
    dynamic_library_exception(const std::string _reason): reason(_reason) {}
    dynamic_library_exception(const dynamic_library_exception& exception) = default;

    dynamic_library_exception& operator=(const dynamic_library_exception& exception) = default;
    virtual const char* what() const throw() {
        return reason.c_str();
    }

    virtual ~dynamic_library_exception() = default;
};  // struct dynamic_library_exception

}   // namespace mori