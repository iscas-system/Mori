#pragma once

#include <iostream>
#include <chrono>

namespace mori {

/**
 * Log
 * Log Entry
 */
struct Log final {
    
};  // struct Log

/**
 * Logger
 * Basic logger interface
 */
struct Logger {
    virtual void submit(const std::string& log) = 0;
};  // struct Logger

/**
 * StdIOLogger
 * Submit logs to std streams
 */
struct StdIOLogger : public Logger {
    virtual void submit(const std::string& log) {
        std::cout<<" "<<log<<std::endl;
    }
};  // struct StdIOLogger
    
}   // namespace mori
