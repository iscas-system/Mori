#pragma once

#include "includes/stdlibs.hpp"

namespace mori {

/**
 * LogLevel
 * Describe the level of the log.
 */
enum LogLevel {
    debug, info, warning, error
};  // enum LogLevel;

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
    LogLevel default_level;
    std::string log_buffer;

    virtual void setDefaultLogLevel(LogLevel level) {default_level = level;}

    inline std::string getLogLevelStr(LogLevel level) {
        switch (level) {
            case debug:
                return "[Debug]  ";
            case info:
                return "[Info]   ";
            case warning:
                return "[Warning]";
            case error:
                return "[Error]  ";
            default:
                return "[Info]   ";
                break;
        }
    }

    virtual void submitInternal(const std::string& log) {log_buffer.append(log);}
    virtual void flush(LogLevel level) {}
    virtual void flush() {flush(default_level);}

    virtual void submit(LogLevel level, const std::string& log) {
        log_buffer.clear();
        submitInternal(log);
        flush(level);
    };
    virtual void submit(const std::string& log) {submit(default_level, log);}

    virtual Logger& operator<<(LogLevel level) {
        default_level = level;
        return *this;
    }
    virtual Logger& operator<<(const std::string& log) {
        submitInternal(log);
        return *this;
    }
};  // struct Logger

/**
 * StdIOLogger
 * Submit logs to std streams
 */
struct StdIOLogger : public Logger {
    virtual void flush(LogLevel level) {
        std::cout<<getLogLevelStr(level)<<" "<<std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()<<" "<<log_buffer<<std::endl;
        log_buffer.clear();
    }

};  // struct StdIOLogger
    
}   // namespace mori
