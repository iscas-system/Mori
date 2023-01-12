#pragma once

namespace mori {
namespace status {

struct memory_status_exception : public std::exception {
protected:
    std::string reason = "Memory status exception.";

public:
    memory_status_exception() = default;
    memory_status_exception(const std::string& _reason): reason(_reason) {}
    memory_status_exception(const memory_status_exception&) = default;
    memory_status_exception& operator=(const memory_status_exception&) = default;

    virtual const char* what() const noexcept override {
        return reason.c_str();
    }

    virtual ~memory_status_exception() = default;
};  // struct memory_status_exception

struct tensor_invalid : public memory_status_exception {
    tensor_invalid() { reason = "Tensor status invalid."; }
    tensor_invalid(const std::string& _reason): memory_status_exception(_reason) {}
};  // struct tensor_invalid

struct memory_section_invalid : public memory_status_exception {
    memory_section_invalid() { reason = "Memory section status invalid."; }
    memory_section_invalid(const std::string& _reason): memory_status_exception(_reason) {}
};  // struct memory_section_invalid

}   // namespace status
}   // namespace mori