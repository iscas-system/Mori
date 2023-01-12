#pragma once

#include <dlfcn.h>

#include "includes/context.hpp"

namespace mori {
namespace utils {

template <typename T>
void* load_dylib(const std::string& dylib, const std::string& path, const std::string& entry, std::unique_ptr<T>& ptr) {
    typedef int(*EntryType)(std::unique_ptr<T>&);
    void* hInst = dlopen(path.c_str(), RTLD_LAZY);
    if (!hInst) throw dynamic_library_exception("Failed to open dynamic library: " + dylib);
    EntryType f = (EntryType)dlsym(hInst, entry.c_str());

    int ret;
    if (f) ret = f(ptr);
    else throw dynamic_library_exception("Failed to access entry: " + dylib);

    if (ret != 0) throw dynamic_library_exception("Failed to enter entry function: " + dylib);

    return hInst;
}

template <typename T>
void* load_dylib(const std::string& dylib, const std::string& path, const std::string& entry, std::unique_ptr<T>& ptr, const Context::View& context) {
    typedef int(*EntryType)(std::unique_ptr<T>&, const Context::View&);
    void* hInst = dlopen(path.c_str(), RTLD_LAZY);
    if (!hInst) throw dynamic_library_exception("Failed to open dynamic library: " + dylib);
    EntryType f = (EntryType)dlsym(hInst, entry.c_str());

    int ret;
    if (f) ret = f(ptr, context);
    else throw dynamic_library_exception("Failed to access entry: " + dylib);

    if (ret != 0) throw dynamic_library_exception("Failed to enter entry function: " + dylib);

    return hInst;
}

}   // namespace utils
}   // namespace mori