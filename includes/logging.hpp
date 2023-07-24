#pragma once

#include <iostream>
#include <unordered_map>
#include <thread>
#include <shared_mutex>

namespace mori {

/**
 * LogLevel
 * Describe the level of the log.
 */
enum struct LogLevel {
    debug, info, warning, error
};  // enum struct LogLevel

static std::string get_log_level_str(LogLevel level) {
    switch (level) {
        case LogLevel::debug:
            return "[Debug]  ";
        case LogLevel::warning:
            return "[Warning]";
        case LogLevel::error:
            return "[Error]  ";
        default:
            // info
            return "[Info]   ";
    }
}

/**
 * Logger
 * Basic logger interface
 */
struct Logger {
protected:
    typedef Logger&(*func)(Logger&);

protected:
    LogLevel default_level;

    std::unordered_map<std::thread::id, std::ostringstream> sls;
    std::shared_mutex tm;

    std::ostringstream sg;
    std::mutex sm;

    template <typename T>
    void submitInternal(const T& info) {
        std::shared_lock<std::shared_mutex> l{tm};
        auto p = sls.find(std::this_thread::get_id());
        if (p == sls.end()) {
            l.unlock();
            std::unique_lock<std::shared_mutex> lu{tm};
            p = sls.emplace(std::this_thread::get_id(), "").first;
            lu.unlock();
            l.lock();
        }

        auto& sl = p->second;
        l.unlock();
        sl << info;
    }

    virtual void log(LogLevel level, const std::string& log) {}

public:
    inline void setDefaultLogLevel(LogLevel level) { default_level = level; }
    inline LogLevel getDefaultLogLevel() const { return default_level; }

    void flush(LogLevel level) {
        std::unique_lock<std::mutex> ls{sm};

        std::shared_lock<std::shared_mutex> lt{tm};
        auto& sl = sls[std::this_thread::get_id()];
        lt.unlock();

        sg << get_log_level_str(level) << " " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() << " " << sl.str();
        sg.flush();
        std::string entry = sg.str();
        sg.str("");
        ls.unlock();

        log(level, entry);
        sl.str("");
        
    }
    void flush() { flush(default_level); }

    template <typename T>
    void submit(LogLevel level, const T& entry) {
        submitInternal(entry);
        flush(level);
    };
    template <typename T>
    inline void submit(const T& entry) { submit(default_level, entry); }

    Logger& operator<<(LogLevel level) {
        default_level = level;
        return *this;
    }
    Logger& operator<<(func _func) {
        return _func(*this);
    }
    template <typename T>
    Logger& operator<<(const T& info) {
        submitInternal(info);
        return *this;
    }

    void clear() {
        std::unique_lock<std::mutex> ls{sm};
        sg.str("");
        ls.unlock();

        std::unique_lock<std::shared_mutex> lt{tm};
        sls.clear();
        lt.unlock();
    }

};  // struct Logger

static Logger& endl(Logger& logger) {
    logger.flush();
    return logger;
}

/**
 * StdIOLogger
 * Submit logs to std streams
 */
struct StdIOLogger : public Logger {
protected:
    virtual void log(LogLevel level, const std::string& entry) override {
        switch (level) {
            case LogLevel::warning:
                std::clog << entry << std::endl;
                break;
            case LogLevel::error:
                std::cerr << entry << std::endl;
                break;
            default:
                // debug, info
                std::cout << entry << std::endl;
                break;
        }
    }

};  // struct StdIOLogger
    
}   // namespace mori