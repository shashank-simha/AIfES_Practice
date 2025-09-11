#include "ModelBase.h"
#include <cstdlib>     /**< malloc/free fallback */

/**
 * @brief Construct a Model with the given configuration.
 *
 * Ensures allocator_fn and free_fn are valid. If not provided by the user,
 * defaults (malloc/free) are installed and a warning is logged.
 *
 * @param cfg Model configuration.
 */
ModelBase::ModelBase(const ModelConfig& cfg)
    : config(cfg)
{
    // Ensure allocator function is valid
    if (!config.allocator_fn) {
        config.allocator_fn = [](size_t size) -> void* {
            return std::malloc(size);
        };
        LOG_WARN("ModelBase: No allocator provided, using std::malloc");
    }

    // Ensure free function is valid
    if (!config.free_fn) {
        config.free_fn = [](void* ptr) {
            std::free(ptr);
        };
        LOG_WARN("ModelBase: No free function provided, using std::free");
    }
}

/**
 * @brief Virtual destructor (no cleanup required in base).
 */
ModelBase::~ModelBase() {}
