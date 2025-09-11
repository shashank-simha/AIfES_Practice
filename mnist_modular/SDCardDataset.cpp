#include "SDCardDataset.h"
#include <cstring>   // for memcpy

/**
 * @brief Construct an SDCardDataset with sample/label files and configuration.
 *
 * The dataset loads data in chunks from disk into preallocated buffers.
 * The buffers are allocated using the allocator function from DatasetConfig.
 *
 * @param cfg          Dataset configuration (shapes, allocation strategy, allocator, etc.)
 * @param sample_files Vector of file paths for sample chunks
 * @param label_files  Optional vector of file paths for label chunks (empty if not used)
 * @param adapter      Pointer to a FileAdapter implementation (platform-agnostic)
 */
template<typename SampleT, typename LabelT>
SDCardDataset<SampleT, LabelT>::SDCardDataset(
    const DatasetConfig& cfg,
    const std::vector<std::string>& sample_files,
    const std::vector<std::string>& label_files,
    FileAdapter* adapter)
    : DatasetBase(cfg),
      sample_files(sample_files),
      label_files(label_files),
      has_labels(!label_files.empty()),
      adapter(adapter),
      sample_chunk_buffer(nullptr),
      label_chunk_buffer(nullptr),
      chunk_total(0),
      current_chunk(0)
{
    if (!adapter) {
        LOG_ERROR("SDCardDataset: FileAdapter pointer is null");
        return;
    }

    if (sample_files.empty()) {
        LOG_ERROR("SDCardDataset: No sample files provided");
        return;
    }

    // Precompute total number of samples across all chunks
    for (const auto& file : sample_files) {
        if (!adapter->open(file, FileAdapter::OpenMode::READ)) {
            LOG_ERROR("SDCardDataset: Failed to open sample file: %s", file.c_str());
            continue;
        }

        size_t file_size = adapter->size();
        adapter->close();

        size_t sample_size = 1;
        for (auto dim : cfg.input_shape) sample_size *= dim;

        if (file_size % (sample_size * sizeof(SampleT)) != 0) {
            LOG_WARN("SDCardDataset: Sample file size not a multiple of sample dimensions: %s", file.c_str());
        }

        total_samples += file_size / (sample_size * sizeof(SampleT));
    }

    // Allocate buffers based on max chunk size if FIXED strategy is used
    // For LAZY, buffers will be allocated on first load
    if (cfg.alloc_strategy == AllocationStrategy::FIXED) {
        // Determine largest chunk size
        size_t max_chunk_bytes = 0;
        for (const auto& file : sample_files) {
            if (!adapter->open(file, FileAdapter::OpenMode::READ)) continue;
            size_t file_size = adapter->size();
            adapter->close();
            if (file_size > max_chunk_bytes) max_chunk_bytes = file_size;
        }

        sample_chunk_buffer = static_cast<SampleT*>(cfg.allocator_fn(max_chunk_bytes));
        if (!sample_chunk_buffer) {
            LOG_ERROR("SDCardDataset: Failed to allocate sample buffer");
        }

        if (has_labels) {
            // Assume each label is LabelT size
            size_t max_label_bytes = max_chunk_bytes / sizeof(SampleT) * sizeof(LabelT);
            label_chunk_buffer = static_cast<LabelT*>(cfg.allocator_fn(max_label_bytes));
            if (!label_chunk_buffer) {
                LOG_ERROR("SDCardDataset: Failed to allocate label buffer");
            }
        }
    }
}

/**
 * @brief Destructor for SDCardDataset.
 *
 * Frees any allocated chunk buffers using DatasetConfig's free function.
 */
template<typename SampleT, typename LabelT>
SDCardDataset<SampleT, LabelT>::~SDCardDataset() {
    if (sample_chunk_buffer) {
        config.free_fn(sample_chunk_buffer);
        sample_chunk_buffer = nullptr;
    }

    if (label_chunk_buffer) {
        config.free_fn(label_chunk_buffer);
        label_chunk_buffer = nullptr;
    }
}

/**
 * @brief Fetch the next batch of samples from the dataset.
 *
 * Handles batch-level logic, including cursor updates and
 * dataset end policies (DROP_LAST or WRAP_AROUND). The actual reading
 * of samples from storage is delegated to load_chunk().
 *
 * @param batch_size    Number of samples to fetch.
 * @param sample_buffer Preallocated buffer to store input samples.
 * @param label_buffer  Preallocated buffer to store labels (nullptr if unused).
 * @return BatchStatus::OK if batch successfully fetched,
 *         BatchStatus::EPOCH_END if dataset end reached,
 *         BatchStatus::ERROR on failure.
 */
template<typename SampleT, typename LabelT>
BatchStatus SDCardDataset<SampleT, LabelT>::next_batch(
    size_t batch_size,
    void* sample_buffer,
    void* label_buffer)
{
    if (!adapter) {
        LOG_ERROR("SDCardDataset: FileAdapter pointer is null");
        return BatchStatus::ERROR;
    }

    if (!has_labels && label_buffer) {
        LOG_WARN("SDCardDataset: Label buffer provided but dataset has no labels");
    }

    if (chunk_total == 0) {
        if (!load_chunk(0)) {
            LOG_ERROR("SDCardDataset: Failed to load initial chunk");
            return BatchStatus::ERROR;
        }
    }

    size_t fetched = 0;
    SampleT* samples_ptr = static_cast<SampleT*>(sample_buffer);
    LabelT* labels_ptr = static_cast<LabelT*>(label_buffer);

    while (fetched < batch_size) {
        // Remaining samples in current chunk
        size_t remaining_in_chunk = chunk_total - cursor;
        size_t take = std::min(batch_size - fetched, remaining_in_chunk);

        // Copy samples
        if (sample_buffer) {
            std::memcpy(samples_ptr + fetched * num_elements(config.input_shape),
                        sample_chunk_buffer + cursor * num_elements(config.input_shape),
                        take * num_elements(config.input_shape) * sizeof(SampleT));
        }

        // Copy labels if present
        if (has_labels && label_buffer) {
            std::memcpy(labels_ptr + fetched * num_elements(config.label_shape),
                        label_chunk_buffer + cursor * num_elements(config.label_shape),
                        take * num_elements(config.label_shape) * sizeof(LabelT));
        }

        cursor += take;
        fetched += take;

        // If current chunk exhausted, load next chunk
        if (cursor >= chunk_total && fetched < batch_size) {
            size_t next_chunk = current_chunk + 1;

            if (next_chunk >= sample_files.size()) {
                // Handle end-of-dataset policy
                if (config.end_policy == BatchEndPolicy::WRAP_AROUND) {
                    next_chunk = 0;
                } else if (config.end_policy == BatchEndPolicy::DROP_LAST) {
                    // Drop partial batch
                    return BatchStatus::END;
                }
            }

            if (!load_chunk(next_chunk)) {
                LOG_ERROR("SDCardDataset: Failed to load chunk %s", std::to_string(next_chunk).c_str());
                return BatchStatus::ERROR;
            }
        }
    }

    // Apply optional transform
    if (transform_fn) {
        transform_fn(sample_buffer, label_buffer, batch_size);
    }

    return BatchStatus::OK;
}

/**
 * @brief Reset dataset cursor and reload the first chunk.
 *
 * Behavior:
 * - Resets global cursor to the beginning of the dataset.
 * - Loads the first chunk from storage into memory.
 * - If the dataset is empty, no chunk is loaded.
 */
template<typename SampleT, typename LabelT>
void SDCardDataset<SampleT, LabelT>::reset()
{
    if (!sample_files.empty()) {
        if (!load_chunk(0)) {
            LOG_ERROR("SDCardDataset: Failed to load first chunk during reset");
        }
    } else {
        LOG_WARN("SDCardDataset: Reset called on empty dataset");
    }
}

/**
 * @brief Load one dataset chunk (samples + optional labels) into memory.
 *
 * Behavior depends on allocation strategy:
 * - FIXED: Buffers are allocated in constructor and reused here.
 * - LAZY : Buffers are allocated on-demand for each chunk and freed when replaced.
 *
 * @param chunk_index Index of the chunk to load (aligned with sample_files / label_files).
 * @return true if load succeeded, false otherwise.
 */
template<typename SampleT, typename LabelT>
bool SDCardDataset<SampleT, LabelT>::load_chunk(size_t chunk_index)
{
    if (!adapter) {
        LOG_ERROR("SDCardDataset: FileAdapter pointer is null");
        return false;
    }

    if (chunk_index >= sample_files.size()) {
        LOG_ERROR("SDCardDataset: Invalid chunk index %s", std::to_string(chunk_index).c_str());
        return false;
    }

    // ---------------------------------------------------------------------
    // Load sample file
    // ---------------------------------------------------------------------
    if (!adapter->open(sample_files[chunk_index], FileAdapter::OpenMode::READ)) {
        LOG_ERROR("SDCardDataset: Failed to open sample file %s", sample_files[chunk_index].c_str());
        return false;
    }

    size_t sample_size = num_elements(config.input_shape) * sizeof(SampleT);
    size_t file_size   = adapter->size();
    adapter->close();

    if (file_size % sample_size != 0) {
        LOG_WARN("SDCardDataset: File size not aligned with sample size, truncating");
    }

    chunk_total = file_size / sample_size;

    // Allocate buffer if using LAZY strategy
    if (config.alloc_strategy == AllocationStrategy::LAZY) {
        if (sample_chunk_buffer) {
            config.free_fn(sample_chunk_buffer);
            sample_chunk_buffer = nullptr;
        }
        sample_chunk_buffer = static_cast<SampleT*>(
            config.allocator_fn(chunk_total * sample_size)
        );
        if (!sample_chunk_buffer) {
            LOG_ERROR("SDCardDataset: Failed to allocate sample buffer");
            return false;
        }
    }

    // Actually read data into buffer
    if (!adapter->open(sample_files[chunk_index], FileAdapter::OpenMode::READ)) return false;
    size_t read_bytes = adapter->read(reinterpret_cast<uint8_t*>(sample_chunk_buffer),
                                      chunk_total * sample_size);
    adapter->close();

    if (read_bytes < chunk_total * sample_size) {
        LOG_ERROR("SDCardDataset: Failed to read full sample file %s", sample_files[chunk_index].c_str());
        return false;
    }

    // ---------------------------------------------------------------------
    // Load label file (if present)
    // ---------------------------------------------------------------------
    if (has_labels) {
        if (chunk_index >= label_files.size()) {
            LOG_ERROR("SDCardDataset: Missing label file for chunk %s", std::to_string(chunk_index).c_str());
            return false;
        }

        if (!adapter->open(label_files[chunk_index], FileAdapter::OpenMode::READ)) {
            LOG_ERROR("SDCardDataset: Failed to open label file %s", label_files[chunk_index].c_str());
            return false;
        }

        size_t label_size      = num_elements(config.label_shape) * sizeof(LabelT);
        size_t label_file_size = adapter->size();
        adapter->close();

        size_t label_count = label_file_size / label_size;
        if (label_count < chunk_total) {
            LOG_WARN("SDCardDataset: Label file smaller than samples, truncating");
            chunk_total = label_count;
        }

        if (config.alloc_strategy == AllocationStrategy::LAZY) {
            if (label_chunk_buffer) {
                config.free_fn(label_chunk_buffer);
                label_chunk_buffer = nullptr;
            }
            label_chunk_buffer = static_cast<LabelT*>(
                config.allocator_fn(chunk_total * label_size)
            );
            if (!label_chunk_buffer) {
                LOG_ERROR("SDCardDataset: Failed to allocate label buffer");
                return false;
            }
        }

        if (!adapter->open(label_files[chunk_index], FileAdapter::OpenMode::READ)) return false;
        size_t label_read = adapter->read(reinterpret_cast<uint8_t*>(label_chunk_buffer),
                                          chunk_total * label_size);
        adapter->close();

        if (label_read < chunk_total * label_size) {
            LOG_ERROR("SDCardDataset: Failed to read full label file %s", label_files[chunk_index].c_str());
            return false;
        }
    }

    // ---------------------------------------------------------------------
    // Update dataset state
    // ---------------------------------------------------------------------
    current_chunk = chunk_index;
    cursor = 0;

    LOG_INFO("SDCardDataset: Loaded chunk %s with %s samples",
             std::to_string(chunk_index).c_str(), std::to_string(chunk_total).c_str());

    return true;
}

template class SDCardDataset<uint8_t, uint8_t>;
