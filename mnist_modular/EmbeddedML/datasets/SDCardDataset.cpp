#include <cstring>

#include "SDCardDataset.h"

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

    // Override element sizes to match template types
    if (config.input_elem_size != sizeof(SampleT)) {
        LOG_WARN("sample_elem_size mismatch, overriding to sizeof(SampleT)");
        config.input_elem_size = sizeof(SampleT);
    }

    if (config.label_elem_size != sizeof(LabelT)) {
        LOG_WARN("label_elem_size mismatch, overriding to sizeof(LabelT)");
        config.label_elem_size = sizeof(LabelT);
    }

    // Precompute total number of samples across all chunks
    for (const auto& file : sample_files) {
        if (!adapter->open(file, FileAdapter::OpenMode::READ)) {
            LOG_ERROR("SDCardDataset: Failed to open sample file: %s", file.c_str());
            continue;
        }

        size_t file_size = adapter->size();
        adapter->close();

        size_t num_sample_elems = num_elements(config.input_shape);
        size_t sample_size_bytes = num_sample_elems * config.input_elem_size;

        if (file_size % sample_size_bytes != 0) {
            LOG_WARN("SDCardDataset: Sample file size not a multiple of sample size: %s", file.c_str());
        }

        total_samples += file_size / sample_size_bytes;
    }

    // Allocate buffers for FIXED strategy
    if (cfg.alloc_strategy == AllocationStrategy::Fixed) {
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
            size_t max_label_bytes = max_chunk_bytes / sizeof(SampleT) * sizeof(LabelT);
            label_chunk_buffer = static_cast<LabelT*>(cfg.allocator_fn(max_label_bytes));
            if (!label_chunk_buffer) {
                LOG_ERROR("SDCardDataset: Failed to allocate label buffer");
            }
        }
    }
}

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

template<typename SampleT, typename LabelT>
BatchStatus SDCardDataset<SampleT, LabelT>::next_batch_impl(
    size_t batch_size,
    void* sample_buffer,
    void* label_buffer)
{
    if (!adapter) {
        LOG_ERROR("SDCardDataset: FileAdapter pointer is null");
        return BatchStatus::Error;
    }

    if (!has_labels && label_buffer) {
        LOG_WARN("SDCardDataset: Label buffer provided but dataset has no labels");
    }

    if (chunk_total == 0) {
        if (!load_chunk(0)) {
            LOG_ERROR("SDCardDataset: Failed to load initial chunk");
            return BatchStatus::Error;
        }
    }

    size_t fetched = 0;
    SampleT* samples_ptr = static_cast<SampleT*>(sample_buffer);
    LabelT* labels_ptr = static_cast<LabelT*>(label_buffer);

    while (fetched < batch_size) {
        size_t remaining_in_chunk = chunk_total - cursor;
        size_t take = std::min(batch_size - fetched, remaining_in_chunk);

        if (sample_buffer) {
            std::memcpy(samples_ptr + fetched * num_elements(config.input_shape),
                        sample_chunk_buffer + cursor * num_elements(config.input_shape),
                        take * num_elements(config.input_shape) * sizeof(SampleT));
        }

        if (has_labels && label_buffer) {
            std::memcpy(labels_ptr + fetched * num_elements(config.label_shape),
                        label_chunk_buffer + cursor * num_elements(config.label_shape),
                        take * num_elements(config.label_shape) * sizeof(LabelT));
        }

        cursor += take;
        fetched += take;

        if (cursor >= chunk_total && fetched < batch_size) {
            size_t next_chunk = current_chunk + 1;

            if (next_chunk >= sample_files.size()) {
                if (config.end_policy == BatchEndPolicy::WrapAround) {
                    next_chunk = 0;
                } else if (config.end_policy == BatchEndPolicy::DropLast) {
                    return BatchStatus::End;
                }
            }

            if (!load_chunk(next_chunk)) {
                LOG_ERROR("SDCardDataset: Failed to load chunk %zu", next_chunk);
                return BatchStatus::Error;
            }
        }
    }

    return BatchStatus::OK;
}

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

template<typename SampleT, typename LabelT>
bool SDCardDataset<SampleT, LabelT>::load_chunk(size_t chunk_index)
{
    if (!adapter) {
        LOG_ERROR("SDCardDataset: FileAdapter pointer is null");
        return false;
    }

    if (chunk_index >= sample_files.size()) {
        LOG_ERROR("SDCardDataset: Invalid chunk index %zu", chunk_index);
        return false;
    }

    // Load sample file
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

    // Allocate buffer for LAZY strategy
    if (config.alloc_strategy == AllocationStrategy::Lazy) {
        if (sample_chunk_buffer) {
            config.free_fn(sample_chunk_buffer);
            sample_chunk_buffer = nullptr;
        }
        sample_chunk_buffer = static_cast<SampleT*>(config.allocator_fn(chunk_total * sample_size));
        if (!sample_chunk_buffer) {
            LOG_ERROR("SDCardDataset: Failed to allocate sample buffer");
            return false;
        }
    }

    if (!adapter->open(sample_files[chunk_index], FileAdapter::OpenMode::READ)) return false;
    size_t read_bytes = adapter->read(reinterpret_cast<uint8_t*>(sample_chunk_buffer),
                                      chunk_total * sample_size);
    adapter->close();

    if (read_bytes < chunk_total * sample_size) {
        LOG_ERROR("SDCardDataset: Failed to read full sample file %s", sample_files[chunk_index].c_str());
        return false;
    }

    // Load label file if present
    if (has_labels) {
        if (chunk_index >= label_files.size()) {
            LOG_ERROR("SDCardDataset: Missing label file for chunk %zu", chunk_index);
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

        if (config.alloc_strategy == AllocationStrategy::Lazy) {
            if (label_chunk_buffer) {
                config.free_fn(label_chunk_buffer);
                label_chunk_buffer = nullptr;
            }
            label_chunk_buffer = static_cast<LabelT*>(config.allocator_fn(chunk_total * label_size));
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

    current_chunk = chunk_index;
    cursor = 0;

    LOG_INFO("SDCardDataset: Loaded chunk %zu with %zu samples", chunk_index, chunk_total);

    return true;
}

// Explicit instantiation
template class SDCardDataset<uint8_t, uint8_t>;
