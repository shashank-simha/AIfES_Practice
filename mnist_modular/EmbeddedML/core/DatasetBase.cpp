#include <cstdlib> // malloc/free fallback
#include <cstring>  // for std::memcpy

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

    // set default transform
    set_transform(nullptr);
}

DatasetBase::~DatasetBase() {
    // Nothing to do in base
}

void DatasetBase::default_transform(const void* raw_input,
                                    const void* raw_label,
                                    void* dst_input,
                                    void* dst_label,
                                    size_t batch_size)
{
    // ---- Handle input ----
    if (raw_input && dst_input) {
        size_t num_elems = num_elements(config.input_shape);
        size_t bytes = num_elems * config.input_elem_size * batch_size;

        std::memcpy(dst_input, raw_input, bytes);
    }

    // ---- Handle label (optional) ----
    if (config.label_shape.size() > 0 && raw_label && dst_label) {
        size_t num_elems = num_elements(config.label_shape);
        size_t bytes = num_elems * config.label_elem_size * batch_size;

        std::memcpy(dst_label, raw_label, bytes);
    }
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

    // Input size
    size_t raw_input_elems  = num_elements(config.input_shape) * batch_size;
    size_t raw_input_bytes  = raw_input_elems * config.input_elem_size;

    // Label size (only if defined)
    size_t raw_label_elems  = config.label_shape.empty() ? 0 : num_elements(config.label_shape) * batch_size;
    size_t raw_label_bytes  = raw_label_elems * config.label_elem_size;

    // Allocate raw buffers
    void* raw_input = nullptr;
    void* raw_label = nullptr;

    if (raw_input_bytes > 0) {
        raw_input = config.allocator_fn(raw_input_bytes);
        if (!raw_input) return BatchStatus::Error;
    }

    if (raw_label_bytes > 0) {
        raw_label = config.allocator_fn(raw_label_bytes);
        if (!raw_label) {
            config.free_fn(raw_input);
            return BatchStatus::Error;
        }
    }

    // Fill from dataset
    BatchStatus st = next_batch_impl(batch_size, raw_input, raw_label);

    if (st == BatchStatus::OK) {
        // Run transform into user buffers
        transform_fn(raw_input, raw_label, input_buffer, label_buffer, batch_size);
    }

    // Free temporary raw buffers
    if (raw_input) config.free_fn(raw_input);
    if (raw_label) config.free_fn(raw_label);

    return st;
}


void DatasetBase::set_transform(std::function<void(const void*, const void*, void*, void*, size_t)> fn) {
    if (fn) {
        transform_fn = fn;
        LOG_INFO("DatasetBase: Custom transform function set. "
                "Make sure it correctly handles your dataset’s input/label types "
                "(raw input element size = %zu bytes, raw label element size = %zu bytes).",
                config.input_elem_size,
                config.label_shape.empty() ? 0 : config.label_elem_size);
    }
    else {
        transform_fn = [this](const void* raw_in, const void* raw_lbl,
                              void* dst_in, void* dst_lbl, size_t bs)
        {
            this->default_transform(raw_in, raw_lbl, dst_in, dst_lbl, bs);
        };
        // Loud warning about limitations of default transform
        LOG_WARN("DatasetBase: Default transform set (identity memcpy). "
        "This only works when source and destination element types are identical "
        "(input element size = %zu bytes, label element size = %zu bytes). "
        "Provide a custom transform if type conversion (e.g., uint8 → float) is required.",
        config.input_elem_size,
        config.label_shape.empty() ? 0 : config.label_elem_size);
    }
}
