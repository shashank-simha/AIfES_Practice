#include <cstdlib> // malloc/free fallback

#include "DatasetBase.h"

DatasetBase::DatasetBase(const DatasetConfig& cfg)
    : config(cfg),
      cursor(0),
      total_samples(0),
      transform_fn(nullptr)
{
    // Ensure allocator function is valid
    if (!config.allocator_fn) {
        config.allocator_fn = [](size_t size) -> void* { return std::malloc(size); };
        LOG_WARN("DatasetBase: No allocator provided, using std::malloc");
    }

    // Ensure free function is valid
    if (!config.free_fn) {
        config.free_fn = [](void* ptr) { std::free(ptr); };
        LOG_WARN("DatasetBase: No free function provided, using std::free");
    }

    // Default transform is a no-op to simplify derived implementations
    transform_fn = [](void* /*input*/, void* /*label*/, size_t /*batch_size*/) {
        /* no-op */
    };
}

DatasetBase::~DatasetBase() {
    // Nothing to do in base
}

void DatasetBase::reset() {
    cursor = 0;
}

size_t DatasetBase::size() const {
    return total_samples;
}

BatchStatus DatasetBase::fetch_batch(size_t batch_size,
                                     void* input_buffer,
                                     void* label_buffer)
{
    if (batch_size == 0) return BatchStatus::OK;

    BatchStatus st = next_batch_impl(batch_size, input_buffer, label_buffer);

    if (st == BatchStatus::OK) {
        // Always run transform (transform_fn is guaranteed non-null)
        transform_fn(input_buffer, label_buffer, batch_size);
    }

    return st;
}

void DatasetBase::set_transform(std::function<void(void*, void*, size_t)> fn) {
    if (fn) transform_fn = fn;
    else {
        // reset to default no-op
        transform_fn = [](void* /*input*/, void* /*label*/, size_t /*batch_size*/) {};
    }
}
