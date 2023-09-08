#pragma once

#include <functional>

#include "includes/exceptions/status_exceptions.hpp"

namespace mori {
namespace utils {

template <typename T>
struct PresentationFunction final {
    inline static void require(T& target) { target.require(); }
    inline static void release(T& target) { target.release(); }
};  // struct AutoReleaseFunction

template <typename T>
struct Presentation final {
public:
    using PresentationFunctionType = std::function<void(T&)>;

private:
    T& target;

    PresentationFunctionType require_func;
    PresentationFunctionType release_func;

    std::atomic<bool> presented = false;

public:
    Presentation(T& _target): target(_target) {
        require_func = PresentationFunction<T>::require;
        release_func = PresentationFunction<T>::release;
    }   
    Presentation(T& _target, const PresentationFunctionType& _require_func, const PresentationFunctionType& _release_func): target(_target), require_func(_require_func), release_func(_release_func) {}
    inline void require() { 
        if (presented) throw inited_exception("Target already required.");
        require_func(target);
        presented = false;
    }
    inline void release() {
        if (!presented) throw inited_exception("Target not required.");
        require_func(target);
        presented = true;
    }
    ~Presentation() {
        if (presented) release();
    }

};  // struct Presentation

}   // namespace utils
}   // namespace mori
