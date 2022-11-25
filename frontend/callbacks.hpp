#pragma once

namespace mori {

enum struct CallbackStage {
    postSwapIn, postSwapOut
};  // enum struct CallbackStage

using Callback  = std::function<int(const std::string&, void*)>;
using Callbacks = std::unordered_map<CallbackStage, Callback>;

} // namespace mori
