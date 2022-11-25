#pragma once

#include <dlfcn.h>

namespace mori {
namespace utils {

template <typename T, typename O>
void* load_dylib(const std::string& dylib, const std::string& path, const std::string& entry, std::unique_ptr<O>& ptr) {
    void* hInst = dlopen(path.c_str(), RTLD_LAZY);
    if (!hInst) throw dynamic_library_exception("Failed to open dynamic library: " + dylib);
    T f = (T)dlsym(hInst, entry.c_str());

    int ret;
    if (f) ret = f(ptr);
    else throw dynamic_library_exception("Failed to access entry: " + dylib);

    if (ret != 0) throw dynamic_library_exception("Failed to enter entry function: " + dylib);

    return hInst;
}

template <typename T, typename O, typename C>
void* load_dylib(const std::string& dylib, const std::string& path, const std::string& entry, std::unique_ptr<O>& ptr, C context) {
    void* hInst = dlopen(path.c_str(), RTLD_LAZY);
    if (!hInst) throw dynamic_library_exception("Failed to open dynamic library: " + dylib);
    T f = (T)dlsym(hInst, entry.c_str());

    int ret;
    if (f) ret = f(ptr, context);
    else throw dynamic_library_exception("Failed to access entry: " + dylib);

    if (ret != 0) throw dynamic_library_exception("Failed to enter entry function: " + dylib);

    return hInst;
}

}   // namespace utils
}   // namespace mori