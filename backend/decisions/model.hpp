#pragma once

#include <vector>
#include <map>
#include <unordered_map>

#include "includes/memory_status.hpp"
#include "includes/memory_info.hpp"
#include "includes/memory_layout.hpp"
#include "includes/memory_schedule_event.hpp"
#include "includes/exceptions/status_exceptions.hpp"

namespace mori {
namespace decisions {

struct Node final {
    Region &region;

    size_t lower_remaining_size = 0; // The remaining size analyzed with lower layer nodes.
    size_t upper_remaining_size = 0; // The remaining size analyzed with upper layer nodes.

    size_t lower_fragment_remaining_size = 0;
    size_t upper_fragment_remaining_size = 0;

    int cluster = 0;
    int lower_group = 0; // The group while this node analyzed with lower layer nodes.
    int upper_group = 0; // The group while this node analyzed with upper layer nodes.

    std::vector<Node*> posts;

    Node(Region &_r) : region(_r) {
        lower_remaining_size = _r.size;
        upper_remaining_size = _r.size;
    }

    void setFragment(size_t _size) {
        region.fragment_size = _size;
        lower_fragment_remaining_size = _size;
        upper_fragment_remaining_size = _size;
    }
}; //  struct Node

struct Model final {
private:


private:
    // std::unordered_map<std::string, Node*> tensors;
    MemoryMap memory_map;
    std::unordered_map<std::string, Node> nodes;

    std::map<int, size_t> clusters;

    int current_layer = 0;
    size_t layer_size = 0;

    // size_t smin = 1048576;  // 1 MB
    size_t smin = 16;

    size_t device_size = 0;

    bool sectioned  = true;
    bool fragmented = true;
    bool analyzed   = false;

protected:
    std::pair<int, int> getClusterSizeRatio(int c1, int c2) {
        size_t s1 = clusters.at(c1);
        size_t s2 = clusters.at(c2);

        if (s1 == s2) return std::make_pair(1, 1);

        size_t c = s1 < s2 ? s1 : s2;
        for (int i = c; i > 0; --i) {
            if (s1 % i == 0 && s2 % i == 0) {
                s1 /= i;
                s2 /= i;
            }
        }
        
        // TODO: calculate cluster size ratio.
        return std::make_pair(s2, s1);
    }

protected:
    void fillModel(status::MemoryStatus& status) {
        std::string entry = status.getEntry();
        for (auto &so : status.getExecutionOrder()) {
            status::OperatorPres op_pres = status.referenceOperator(so);
            for (auto &st : op_pres.getTensors()) {
                status::TensorPres tensor_pres = status.referenceTensor(st);
                // if (tensor_pres.isPersistant() || tensor_pres.isTransient()) continue;
                // Do submit here.
                Layer& l = memory_map.getCurrentLayer();
                if (l.requested_size + tensor_pres.getSize() > l.size) memory_map.createLayer();
                Region r(tensor_pres.getName(), tensor_pres.getSize());
                memory_map.submitMemoryRegion(r);
                nodes.emplace(r.name, memory_map.regions.at(r.name));
            }
        }
    }

    void clustering() {
        clusters[1] = 128;
        clusters[2] = 256;
        clusters[3] = 384;
        clusters[4] = 512;
        // TODO: Cluster the nodes with K-Means++
        for (auto &x : memory_map.layers) {
            for (auto &s : x.regions) {
                // clusters[y.size] = y.size;
                // y.cluster = y.size;
                Node& n = nodes.at(s);
                switch (n.region.size) {
                    case 124:
                        n.cluster = 1;
                        break;
                    case 256:
                    case 258:
                        n.cluster = 2;
                        break;
                    case 384:
                        n.cluster = 3;
                        break;
                    case 512:
                        n.cluster = 4;
                        break;
                    default:
                        assert(0);
                        break;
                }
            }
        }
    }

    void grouping() {
        int current_group = 1;

        auto pl = memory_map.layers.rbegin();
        ++pl;   // If grouping now, there must be two or more layers.
        while (pl != memory_map.layers.rend()) {
            auto pu = pl;
            --pu;

            auto& lowers = pl->regions;
            auto& uppers = pu->regions;

            auto ql = lowers.begin();
            auto qu = uppers.begin();

            while (ql != lowers.end() && qu != uppers.end()) {
                int lower_cluster = nodes.at(*ql).cluster;
                int upper_cluster = nodes.at(*qu).cluster;

                // Initial ratio.
                auto&& grouping_ratio = getClusterSizeRatio(lower_cluster, upper_cluster);

                int lower_count = grouping_ratio.first;
                int upper_count = grouping_ratio.second;

                int lower_count_current = 0;
                int upper_count_current = 0;

                while (ql != lowers.end() && lower_count_current < lower_count) {
                    Node& nl = nodes.at(*ql);
                    auto&& current_grouping_ratio = getClusterSizeRatio(lower_cluster, nl.cluster);
                    // Adjustment of grouping.
                    if (current_grouping_ratio.first != current_grouping_ratio.second) {
                        lower_count         = lower_count         * current_grouping_ratio.second / current_grouping_ratio.first;
                        lower_count_current = lower_count_current * current_grouping_ratio.second / current_grouping_ratio.first;
                        lower_cluster       = nl.cluster;
                    }
                    nl.upper_group = current_group;    // Analyzing with upper layer nodes, hence upper_group.
                    ++lower_count_current;
                    ++ql;
                }
                while (qu != uppers.end() && upper_count_current < upper_count) {
                    Node& nu = nodes.at(*qu);
                    auto&& current_grouping_ratio = getClusterSizeRatio(upper_cluster, nu.cluster);
                    // Adjustment of grouping.
                    if (current_grouping_ratio.first != current_grouping_ratio.second) {
                        upper_count         = upper_count         * current_grouping_ratio.second / current_grouping_ratio.first;
                        upper_count_current = upper_count_current * current_grouping_ratio.second / current_grouping_ratio.first;
                        upper_cluster       = nu.cluster;
                    }
                    nu.lower_group = current_group;    // Analyzing with lower layer nodes, hence lower_group.
                    ++upper_count_current;
                    ++qu;
                }
                ++current_group;
            }

            if (ql == lowers.end()) {
                while (qu != uppers.end()) nodes.at(*qu++).lower_group = current_group;
                ++current_group;
            } else if (qu == uppers.end()) {
                while (ql != lowers.end()) nodes.at(*ql++).upper_group = current_group;
                ++current_group;
            }

            ++pl;
        }
    }

    void generateFragments() {
        bool tensor_moved = true;
        while (tensor_moved) {
            tensor_moved = false;
            auto pl = memory_map.layers.rbegin();
            while (pl != memory_map.layers.rend()) {
                // Fragments exceed the memory capacity.
                if (!pl->isAccomodatable()) {
                    // Check the existance of the upper layer.
                    if (pl == memory_map.layers.rbegin()) memory_map.createLayer();
                    auto pu = pl;
                    --pu;
                    
                    auto& lowers = pl->regions;
                    auto& uppers = pu->regions;

                    auto ql = lowers.begin();
                    auto qu = uppers.begin();

                    // Remove all the fragments in lower and upper layer, since the fragments should be regenerated.
                    while (ql != lowers.end()) {
                        Node& nl = nodes.at(*ql);
                        if (nl.region.fragment_size != 0) {
                            pl->requested_size -= nl.region.fragment_size;
                            nl.setFragment(0);
                        }
                        nl.upper_group = 0;
                        ++ql;
                    }
                    while (qu != uppers.end()) {
                        Node& nu = nodes.at(*qu);
                        if (nu.region.fragment_size != 0) {
                            pu->requested_size -= nu.region.fragment_size;
                            nu.setFragment(0);
                        }
                        nu.lower_group = 0;
                        nu.upper_group = 0;
                        ++qu;
                    }

                    ql = lowers.begin();
                    qu = uppers.begin();

                    do {
                        assert(nodes.at(lowers.back()).region.fragment_size == 0);
                        qu = uppers.insert(qu, lowers.back());
                        ++qu;
                        pu->requested_size += nodes.at(lowers.back()).region.size;
                        pl->requested_size -= nodes.at(lowers.back()).region.size;
                        lowers.pop_back();
                    } while (!pl->isAccomodatable());
                    tensor_moved = true;

                    --pl;
                    continue;
                }

                if (tensor_moved) break;

                auto pu = pl++;
                if (pl == memory_map.layers.rend()) break;

                auto& lowers = pl->regions;
                auto& uppers = pu->regions;

                auto ql = lowers.begin();
                auto qu = uppers.begin();

                size_t size_tl = 0;
                size_t size_tu = 0;

                while (ql != lowers.end() && qu != uppers.end()) {  // If ql or qu reaches the end of layer, no need to generate a fragment.
                    Node& nl = nodes.at(*ql);
                    Node& nu = nodes.at(*qu);

                    size_t size_tl_target = size_tl + nl.region.size;
                    size_t size_tu_target = size_tu + nu.region.size + nu.region.fragment_size;

                    if (size_tl_target == size_tu_target) {
                        size_tl += nl.region.size;
                        size_tu += nu.region.size + nu.region.fragment_size;
                        ++ql;
                        ++qu;
                    } else if (size_tl_target > size_tu_target) {
                        size_tu += nu.region.size + nu.region.fragment_size;
                        ++qu;
                    } else {
                        size_tl += nl.region.size;
                        size_t size_frag = size_tu + nu.region.size + nu.region.fragment_size - size_tl;
                        if (size_frag < smin) {
                            // Generate fragment.
                            nl.setFragment(size_frag);
                            size_tl += size_frag;
                            pl->requested_size += size_frag;
                        }
                        ++ql;
                    }
                }


            }
        }
    }

    void generateTree() {
        auto pl = memory_map.layers.begin();
        auto pu = pl;
        ++pu;

        while (pu != memory_map.layers.end()) {
            auto& lowers = pl++->regions;
            auto& uppers = pu++->regions;

            auto ql = lowers.begin();
            auto qu = uppers.begin();

            while (ql != lowers.end() && qu != uppers.end()) {  // If ql or qu reaches the end of layer, no need to split the lower layer tensors.
                Node& nl = nodes.at(*ql);
                Node& nu = nodes.at(*qu);
                if (nl.upper_remaining_size > nu.lower_remaining_size) {
                    // Lower layer node along with the fragment larger than upper layer node.
                    size_t size_sect = (nl.upper_remaining_size - nu.lower_remaining_size) > smin ? nu.lower_remaining_size : nl.upper_remaining_size;
                    nl.region.sections.push_back(size_sect);
                    nl.upper_remaining_size -= size_sect;

                    // The nu should be fully covered by nl.
                    nu.lower_remaining_size = 0;

                    // Process of fragment.
                    size_t size_frag = nl.upper_remaining_size > nu.lower_fragment_remaining_size ? nu.lower_fragment_remaining_size : nl.upper_remaining_size;
                    nl.upper_remaining_size -= size_frag;
                    nu.lower_fragment_remaining_size -= size_frag;
                } else {
                    // The remaining lower tensor should be swapped out.
                    nl.region.sections.push_back(nl.upper_remaining_size);
                    nu.lower_remaining_size -= nl.upper_remaining_size;
                    nl.upper_remaining_size = 0;

                    // Process of fragment.
                    size_t size_frag = nl.upper_fragment_remaining_size > nu.lower_remaining_size ? nu.lower_remaining_size : nl.upper_fragment_remaining_size;
                    nl.upper_fragment_remaining_size -= size_frag;
                    nu.lower_remaining_size -= size_frag;
                }

                nl.posts.push_back(&nu);
                if (nl.upper_remaining_size == 0 && nl.upper_fragment_remaining_size == 0) ++ql;
                if (nu.lower_remaining_size == 0 && nu.lower_fragment_remaining_size == 0) ++qu;
            }

            if (qu == uppers.end()) {
                while (ql != lowers.end()) {
                    Node& n = nodes.at(*ql++);
                    n.region.sections.push_back(n.upper_remaining_size);
                    n.upper_remaining_size = 0;
                }
                for (auto &s : uppers) assert(nodes.at(s).lower_remaining_size == 0);
            }
            for (auto &s : lowers) assert(nodes.at(s).upper_remaining_size == 0);
        }

        // Section information for the top layer. No splition (only one section).
        for (auto &s : memory_map.layers.back()) {
            Region& r = memory_map.regions.at(s);
            r.sections.push_back(r.size);
        }
    }

public:
    Model() = default;
    Model(const Model&) = default;
    Model(Model&&) = default;

    Model& operator=(const Model&) = default;
    Model& operator=(Model&&) = default;

    void setMemoryInfo(const MemoryInfo& memory_info) {
        device_size = memory_info.device.total_size;
        memory_map.setMemorySize(device_size);
    }

    void analyze(status::MemoryStatus& status, bool fragmented = true) {
        if (analyzed) return;

        fillModel(status);
        for (auto &x : memory_map.layers) assert(x.isAccomodatable());
        // If only one layer, there'e no need to analyze.
        if (memory_map.layers.size() != 1) {
            generateFragments();
            for (auto &x : memory_map.layers) assert(x.isAccomodatable());
            generateTree();
        }
        analyzed = true;
    }

    int getLayerCount() const { return memory_map.layers.size(); }
    const std::vector<Layer>& getLayers() const { return memory_map.layers; }
    const Layer& getLayer(int _layer) const { return memory_map.layers[_layer]; }
    const Node getMemoryNode(const std::string& _node) const { return nodes.at(_node); }

    MemoryMap getMemoryMap() {
        if (!analyzed) throw status_exception("Memory map not analyzed.");
        return memory_map;
    }

    void clear() noexcept { 
        clusters.clear();
        memory_map.clear();
        memory_map.createLayer();
        analyzed = false;
    }

    ~Model() = default;
};  // struct Model

}   // namespace decisions
}   // namespace mori