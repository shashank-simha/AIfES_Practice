#include <cstdlib>

#include "ModelBase.h"

ModelBase::ModelBase(const ModelConfig& cfg)
    : config(cfg)
{
    // Ensure allocator function is valid
    if (!config.allocator_fn) {
        config.allocator_fn = [](size_t size) -> void* { return std::malloc(size); };
        LOG_WARN("ModelBase: No allocator provided, using std::malloc");
    }

    // Ensure free function is valid
    if (!config.free_fn) {
        config.free_fn = [](void* ptr) { std::free(ptr); };
        LOG_WARN("ModelBase: No free function provided, using std::free");
    }
}

ModelBase::~ModelBase() {}
