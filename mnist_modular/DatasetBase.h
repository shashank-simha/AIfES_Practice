#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <cstdlib>
#include "logger.h"

/**
 * @brief Allocation strategy for dataset buffers.
 */
enum class AllocationStrategy {
    Lazy,   /**< Allocate buffers only on first use or resize if needed */
    Fixed   /**< Allocate buffers for largest chunk upfront */
};

/**
 * @brief Policy for handling end-of-dataset when requesting a batch.
 */
enum class BatchEndPolicy {
    DropLast,      /**< Drop partial batch if dataset does not have enough samples */
    WrapAround     /**< Wrap around to the beginning */
};

/**
 * @brief Status of a batch request.
 */
enum class BatchStatus {
    OK,     /**< Batch fetched successfully */
    End,    /**< End of dataset reached (depends on end_policy) */
    Error   /**< Fatal error (I/O failure, allocation failure, etc.) */
};

/**
 * @brief Configuration for a dataset instance.
 *
 * Defines shapes, allocation policy, and custom allocators.
 * Allocator functions are optional; if not provided, a default malloc/free pair is installed.
 */
struct DatasetConfig {
    std::vector<uint32_t> input_shape;   /**< Shape of each input sample (e.g., {1,28,28}) */
    std::vector<uint32_t> label_shape;   /**< Shape of each label (e.g., {1} or {10}) */

    AllocationStrategy alloc_strategy = AllocationStrategy::Lazy; /**< Buffer allocation strategy */
    BatchEndPolicy end_policy = BatchEndPolicy::DropLast;         /**< Policy at dataset end */

    /** User-provided allocator function (optional) */
    std::function<void*(size_t)> allocator_fn = nullptr;

    /** User-provided free function (optional) */
    std::function<void(void*)> free_fn = nullptr;
};

/**
 * @brief Compute number of elements from a shape vector.
 *
 * @param shape Vector of dimensions (e.g., {1, 28, 28})
 * @return Total number of elements (product of dimensions)
 */
static inline uint32_t num_elements(const std::vector<uint32_t>& shape) {
    uint32_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

/**
 * @brief Abstract base class for datasets.
 *
 * Provides configuration, cursor handling, and transform hooks.
 * Derived classes must implement next_batch_impl().
 *
 * Public usage:
 *   - Use fetch_batch(...) to obtain a batch. The base class will call the
 *     dataset-specific next_batch_impl(...) and apply the transform automatically.
 */
class DatasetBase {
public:
    /**
     * @brief Construct a DatasetBase with the given configuration.
     *
     * Ensures allocator_fn and free_fn are set (falls back to malloc/free).
     *
     * @param cfg Dataset configuration.
     */
    explicit DatasetBase(const DatasetConfig& cfg);

    /** Virtual destructor */
    virtual ~DatasetBase();

    /**
     * @brief Reset dataset cursor to the start.
     * Default implementation resets @ref cursor to 0.
     * Derived classes may override if needed.
     */
    virtual void reset();

    /**
     * @brief Return the total number of samples in the dataset.
     * @return Total samples.
     */
    virtual size_t size() const;

    /**
     * @brief Fetch the next batch of samples (with optional labels).
     *
     * This is the method that users/models should call. It delegates to the
     * dataset-specific next_batch_impl() and applies the transform hook (in-place)
     * if the fetch was successful.
     *
     * @param batch_size Number of samples to fetch.
     * @param input_buffer Pointer to user-allocated input buffer.
     * @param label_buffer Optional pointer to user-allocated label buffer (nullptr if not used).
     * @return BatchStatus indicating success, end of dataset, or error.
     */
    BatchStatus fetch_batch(size_t batch_size,
                            void* input_buffer,
                            void* label_buffer = nullptr);

    /**
     * @brief Set an optional transformation hook.
     *
     * Called in-place on each batch after fetching.
     *
     * Signature: void(void* input, void* label, size_t batch_size)
     *
     * @param fn Function of signature void(void* input, void* label, size_t batch_size)
     */
    void set_transform(std::function<void(void*, void*, size_t)> fn);

    /** Access the normalized dataset configuration */
    const DatasetConfig& get_config() const { return config; }

    /** Return input sample shape */
    const std::vector<uint32_t>& get_input_shape() const { return config.input_shape; }

    /** Return label sample shape */
    const std::vector<uint32_t>& get_label_shape() const { return config.label_shape; }

protected:
    /**
     * @brief Dataset-specific batch fetch implementation.
     *
     * Derived classes implement this to actually fill the provided buffers
     * with raw (untransformed) data. The base class will call this and then
     * apply the transform function automatically.
     *
     * @param batch_size Number of samples requested.
     * @param input_buffer Buffer to fill (must be large enough).
     * @param label_buffer Optional label buffer to fill (may be nullptr).
     */
    virtual BatchStatus next_batch_impl(size_t batch_size,
                                        void* input_buffer,
                                        void* label_buffer) = 0;

    DatasetConfig config;     /**< Normalized configuration (allocator_fn/free_fn guaranteed valid) */
    size_t cursor = 0;            /**< Current sample index (advances with each batch) */
    size_t total_samples = 0;     /**< Total number of samples available in dataset */

    /** Transform function — default is identity (no-op). Called after next_batch_impl() when fetch_batch() returns OK. */
    std::function<void(void*, void*, size_t)> transform_fn;
};
