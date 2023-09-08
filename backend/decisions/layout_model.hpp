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

struct LayoutModel final {
public:
    struct Node final {
        layout::Region &region;

        size_t lower_remaining_size = 0; // The remaining size analyzed with lower layer nodes.
        size_t upper_remaining_size = 0; // The remaining size analyzed with upper layer nodes.

        size_t lower_fragment_remaining_size = 0;
        size_t upper_fragment_remaining_size = 0;

        int cluster = 0;
        int lower_group = 0; // The group while this node analyzed with lower layer nodes.
        int upper_group = 0; // The group while this node analyzed with upper layer nodes.

        std::vector<Node*> posts;

        Node(layout::Region &_r) : region(_r) {
            lower_remaining_size = _r.size;
            upper_remaining_size = _r.size;
        }

        void setFragment(size_t _size) {
            region.fragment_size = _size;
            lower_fragment_remaining_size = _size;
            upper_fragment_remaining_size = _size;
        }
    }; //  inner struct Node

private:
    // std::unordered_map<std::string, Node*> tensors;
    layout::MemoryMapBuilder memory_map_builer;
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
    void fillModel(status::MemoryStatus& status) {
        std::string entry = status.getEntry();
        for (auto &so : status.getExecutionOrder()) {
            status::OperatorPres op_pres = status.referenceOperator(so);
            for (auto &st : op_pres.getTensors()) {
                status::TensorPres tensor_pres = status.referenceTensor(st);
                if (tensor_pres.isPersistent() || tensor_pres.isTransient()) continue;
                // Do submit here.
                layout::Layer& l = memory_map_builer.getCurrentLayer();
                size_t aligned_size = utils::get_memory_aligned_size(tensor_pres.getSize(), memory_map_builer.getMemoryInfo().device.align_size);
                if (l.requested_size + aligned_size > l.size) memory_map_builer.createLayer();
                layout::Region r(tensor_pres.getName(), aligned_size);
                memory_map_builer.submitMemoryRegion(r);
                nodes.emplace(r.name, memory_map_builer.regions.at(r.name));
            }
        }
    }
   
    void generateFragments() {
        bool tensor_moved = true;
        while (tensor_moved) {
            tensor_moved = false;
            auto pl = memory_map_builer.layers.rbegin();
            while (pl != memory_map_builer.layers.rend()) {
                // Fragments exceed the memory capacity.
                if (!pl->isAccomodatable()) {
                    // Check the existance of the upper layer.
                    if (pl == memory_map_builer.layers.rbegin()) memory_map_builer.createLayer();
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
                if (pl == memory_map_builer.layers.rend()) break;

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
        auto pl = memory_map_builer.layers.begin();
        auto pu = pl;
        ++pu;

        while (pu != memory_map_builer.layers.end()) {
            auto& lowers = pl++->regions;
            auto& uppers = pu++->regions;

            auto ql = lowers.begin();
            auto qu = uppers.begin();

            while (ql != lowers.end() && qu != uppers.end()) {  // If ql or qu reaches the end of layer, no need to split the lower layer tensors.
                Node& nl = nodes.at(*ql);
                Node& nu = nodes.at(*qu);

                size_t size_sectioned = nl.upper_remaining_size > nu.lower_remaining_size ? nu.lower_remaining_size : nl.upper_remaining_size;
                if (size_sectioned >= smin) nl.region.sections.push_back(size_sectioned);
                else nl.region.sections.back() += size_sectioned;
                nl.upper_remaining_size -= size_sectioned;
                nu.lower_remaining_size -= size_sectioned;

                // Process of fragment.
                if (nl.upper_remaining_size > 0) {
                    size_t size_frag = nl.upper_remaining_size > nu.lower_fragment_remaining_size ? nu.lower_fragment_remaining_size : nl.upper_remaining_size;
                    nl.upper_remaining_size          -= size_frag;
                    nu.lower_fragment_remaining_size -= size_frag;
                } else if (nu.lower_remaining_size > 0) {
                    size_t size_frag = nu.lower_remaining_size > nl.upper_fragment_remaining_size ? nl.upper_fragment_remaining_size : nu.lower_remaining_size;
                    nu.lower_remaining_size          -= size_frag;
                    nl.upper_fragment_remaining_size -= size_frag;
                } else {
                    size_t size_frag = nl.upper_fragment_remaining_size > nu.lower_fragment_remaining_size ? nu.lower_fragment_remaining_size : nl.upper_fragment_remaining_size;
                    nl.upper_fragment_remaining_size -= size_frag;
                    nu.lower_fragment_remaining_size -= size_frag;
                }

                nl.posts.push_back(&nu);
                if (nl.upper_remaining_size == 0 && nl.upper_fragment_remaining_size == 0) ++ql;
                if (nu.lower_remaining_size == 0 && nu.lower_fragment_remaining_size == 0) ++qu;
            }

            if (qu == uppers.end()) {
                while (ql != lowers.end()) {
                    Node& n = nodes.at(*ql++);
                    if (n.upper_remaining_size >= smin) n.region.sections.push_back(n.upper_remaining_size);
                    else n.region.sections.back() += n.upper_remaining_size;
                    n.upper_remaining_size = 0;
                }
                for (auto &s : uppers) assert(nodes.at(s).lower_remaining_size == 0);
            }
            for (auto &s : lowers) assert(nodes.at(s).upper_remaining_size == 0);
        }

        // Section information for the top layer. No splition (only one section).
        for (auto &s : memory_map_builer.layers.back()) {
            layout::Region& r = memory_map_builer.regions.at(s);
            r.sections.push_back(r.size);
        }
    }

public:
    LayoutModel() = default;
    LayoutModel(const LayoutModel&) = default;
    LayoutModel(LayoutModel&&) = default;

    LayoutModel& operator=(const LayoutModel&) = default;
    LayoutModel& operator=(LayoutModel&&) = default;

    void setMemoryInfo(const MemoryInfo& memory_info) {
        device_size = memory_info.device.common_block.size;
        memory_map_builer.setMemoryInfo(memory_info);
    }

    void analyze(status::MemoryStatus& status, bool fragmented = true) {
        if (analyzed) return;

        fillModel(status);
        for (auto &x : memory_map_builer.layers) assert(x.isAccomodatable());
        // If only one layer, there'e no need to analyze.
        if (memory_map_builer.layers.size() != 1) {
            generateFragments();
            for (auto &x : memory_map_builer.layers) assert(x.isAccomodatable());
            generateTree();
        }
        analyzed = true;
    }

    int getLayerCount() const { return memory_map_builer.layers.size(); }
    const std::vector<layout::Layer>& getLayers() const { return memory_map_builer.layers; }
    const layout::Layer& getLayer(int _layer) const { return memory_map_builer.layers[_layer]; }
    const Node getMemoryNode(const std::string& _node) const { return nodes.at(_node); }

    layout::MemoryMap getMemoryMap() {
        if (!analyzed) throw status_exception("Memory map not analyzed.");
        return memory_map_builer.build();
    }

    void clear() noexcept { 
        clusters.clear();
        memory_map_builer.clear();
        memory_map_builer.createLayer();
        analyzed = false;
    }

    ~LayoutModel() = default;
};  // struct Model

}   // namespace decisions
}   // namespace mori