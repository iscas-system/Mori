#pragma once

namespace mori {

struct status_exception : public std::exception {
protected:
    std::string reason = "Status exception.";

public:
    status_exception() = default;
    status_exception(const std::string& _reason): reason(_reason) {}
    status_exception(const status_exception&) = default;
    status_exception& operator=(const status_exception&) = default;

    virtual const char* what() const noexcept override {
        return reason.c_str();
    }

    virtual ~status_exception() = default;
};  // struct status_exception

struct uninited_exception : public status_exception {
    uninited_exception() { reason = "Not inited."; }
    uninited_exception(const std::string& _reason): status_exception(_reason) {}
};  // struct uninited_exception

struct inited_exception : public status_exception {
    inited_exception() { reason = "Already inited."; }
    inited_exception(const std::string& _reason): status_exception(_reason) {}
};  // struct inited_exception

}   // namespace mori