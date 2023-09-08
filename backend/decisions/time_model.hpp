#pragma once

#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <utility>

#include "includes/exceptions/status_exceptions.hpp"

namespace mori {
namespace decisions {

struct TransferringModel {
    long analyze(size_t size) {
        // long t = size / 1048576;
        // return t / 12;
        return size >> 2;
    }

};  // struct TransferModel

struct TimeModel final {
private:
    enum struct SynchronizationType {
        prev, post
    };  // enum struct SynchronizationType

public:
    struct Timespan final {
        std::string target;
        long span = 0;
        bool synchronization = false;
        long timepoint = 0;

        Timespan() = default;
        Timespan(const std::string& _target, long _span): target(_target), span(_span) {}

        inline bool isSynchronization() { return synchronization; }
        inline void setSynchronization(bool _synchronization) { synchronization = _synchronization; }
    };  // inner struct Timespan

    struct Lane final {
        SynchronizationType synchronization_type = SynchronizationType::prev;
        std::vector<std::pair<std::string, Timespan>> timespans;
        std::string current_synchronization_label = "";

        void submitSynchronizationLabel(const std::string synchronization_label) {
            if (synchronization_type == SynchronizationType::post) {
                for (auto p = timespans.rbegin(); p != timespans.rend(); ++p) {
                    if (p->second.synchronization) break;
                    if (p->first != synchronization_label) throw status_exception("Synchronization label mismatch.");
                }
            }
            Timespan timespan(synchronization_label, 0);
            auto& target = timespans.emplace_back(synchronization_label, timespan);
            target.second.synchronization = true;
            current_synchronization_label = synchronization_label;
        }
        void submitTimespan(const std::string synchronization_label, const Timespan& _timespan) {
            if (synchronization_type == SynchronizationType::post) {
                if (synchronization_label == current_synchronization_label) throw status_exception("Synchronization label mismatch.");
            } else {
                if (synchronization_label != current_synchronization_label) throw status_exception("Synchronization label mismatch.");
            }
            timespans.emplace_back(synchronization_label, _timespan);
        }
    };  // inner struct Lane

private:
    std::unordered_set<std::string> enabled_synchronization_labels;

    bool strong_synchronization = false;

public:
    Lane execution_lane;
    Lane transferring_lane;

protected:
    void analyzeSynchronization() {
        auto ptrans = transferring_lane.timespans.rbegin();

        long total_execution_time = 0;
        for (auto pexec = execution_lane.timespans.rbegin(); pexec != execution_lane.timespans.rend(); ++pexec) {
            if (pexec->second.synchronization) {
                auto penabled = enabled_synchronization_labels.find(pexec->second.target);
                if (penabled == enabled_synchronization_labels.end()) continue;
            } else {
                total_execution_time += pexec->second.span;
                continue;
            }
        
            // Synchronize execution and transferring lane.
            long total_transferring_time = 0;
            while (ptrans != transferring_lane.timespans.rend()) {
                if (ptrans->second.synchronization) {
                    auto penabled = enabled_synchronization_labels.find(ptrans->second.target);
                    if (penabled != enabled_synchronization_labels.end()) break;
                } else total_transferring_time += ptrans->second.span;
                ++ptrans;
            }
            if (ptrans != transferring_lane.timespans.rend()) assert(ptrans->second.target == pexec->second.target);
            if (total_execution_time >= total_transferring_time) {
                ptrans->second.span = total_execution_time - total_transferring_time;
                total_execution_time = 0;
            } else {
                ptrans->second.span = 0;
                total_execution_time = strong_synchronization ? (total_execution_time - total_transferring_time) : 0;
            }
            ++ptrans;
        }
    }

    void generateTimepoint() {
        long current_timepoint = 0;
        for (auto p = execution_lane.timespans.begin(); p != execution_lane.timespans.end(); ++p) {
            p->second.timepoint = current_timepoint;
            current_timepoint += p->second.span;
        }
        for (auto p = transferring_lane.timespans.rbegin(); p != transferring_lane.timespans.rend(); ++p) {
            current_timepoint -= p->second.span;
            p->second.timepoint = current_timepoint;
        }
    }

public:
    TimeModel() {
        execution_lane.synchronization_type    = SynchronizationType::prev;
        transferring_lane.synchronization_type = SynchronizationType::post;
    }

    inline void submitExecutionSynchronization(const std::string& synchronization_label) {
        execution_lane.submitSynchronizationLabel(synchronization_label);
    }
    inline void submitExecutionTimespan(const std::string& synchronization_label, const Timespan& timespan) {
        execution_lane.submitTimespan(synchronization_label, timespan);
    }
    inline void submitTransferringSynchronization(const std::string& synchronization_label) {
        transferring_lane.submitSynchronizationLabel(synchronization_label);
    }
    inline void submitTransferringTimespan(const std::string& synchronization_label, const Timespan& timespan) {
        transferring_lane.submitTimespan(synchronization_label, timespan);
    }

    void setSynchronizationEnabled(const std::string& _label) {
        enabled_synchronization_labels.insert(_label);
    }

    inline bool isStrongSynchronization() const { return strong_synchronization; }
    void setStrongSynchronization(bool _strong_synchronization) { strong_synchronization = _strong_synchronization; }

    void analyze() {
        analyzeSynchronization();
        generateTimepoint();
    }
};  // struct TimeModel

}   // namespace decisions
}   // namespace mori