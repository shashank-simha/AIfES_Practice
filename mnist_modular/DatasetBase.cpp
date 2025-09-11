#include "DatasetBase.h"
#include <cstdlib>     /**< malloc/free fallback */

/**
 * @brief Construct a DatasetBase with the given configuration.
 *
 * Ensures allocator_fn and free_fn are valid. If not provided by the user,
 * defaults (malloc/free) are installed and a warning is logged.
 *
 * @param cfg Dataset configuration.
 */
DatasetBase::DatasetBase(const DatasetConfig& cfg)
    : config(cfg), cursor(0), total_samples(0), transform_fn(nullptr)
{
    // Ensure allocator function is valid
    if (!config.allocator_fn) {
        config.allocator_fn = [](size_t size) -> void* {
            return std::malloc(size);
        };
        LOG_WARN("DatasetBase: No allocator provided, using std::malloc");
    }

    // Ensure free function is valid
    if (!config.free_fn) {
        config.free_fn = [](void* ptr) {
            std::free(ptr);
        };
        LOG_WARN("DatasetBase: No free function provided, using std::free");
    }

    // total_samples is typically computed by derived classes
}

/**
 * @brief Virtual destructor (no cleanup required in base).
 */
DatasetBase::~DatasetBase() {}

/**
 * @brief Reset the dataset cursor to the start.
 *
 * Default implementation simply sets @ref cursor = 0.
 * Derived classes may override if additional reset logic is required.
 */
void DatasetBase::reset() {
    cursor = 0;
}

/**
 * @brief Return the total number of samples in the dataset.
 *
 * Default implementation returns @ref total_samples.
 * Derived classes may override if the size is computed dynamically.
 *
 * @return Total number of samples.
 */
size_t DatasetBase::size() const {
    return total_samples;
}

/**
 * @brief Set an optional transformation hook for batches.
 *
 * The transform function is invoked after each batch is fetched.
 * Both input and label buffers are passed along with the batch size.
 *
 * @param fn Transformation function of signature:
 *           void(void* input, void* label, size_t batch_size)
 */
void DatasetBase::set_transform(std::function<void(void*, void*, size_t)> fn) {
    transform_fn = fn;
}
