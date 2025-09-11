#pragma once

#include "DatasetBase.h"
#include "FileAdapter.h"
#include "logger.h"
#include <vector>
#include <string>
#include <cstddef>

/**
 * @brief Dataset implementation that streams samples from SD card (or any file system).
 *
 * This class does not assume data type or shape — both are provided via DatasetConfig.
 * Data is loaded sample-by-sample or chunk-by-chunk from disk using a FileAdapter.
 *
 * @tparam SampleT Datatype for samples (e.g., uint8_t, float).
 * @tparam LabelT  Datatype for labels (e.g., uint8_t, float). Can be void if unused.
 */
template<typename SampleT, typename LabelT = void>
class SDCardDataset : public DatasetBase {
public:
    /**
     * @brief Construct SDCardDataset with file paths and configuration.
     *
     * @param cfg          Dataset configuration (shapes, allocation strategy, etc.)
     * @param sample_files Vector of file paths for input samples
     * @param label_files  Optional vector of file paths for labels (empty if not used)
     * @param adapter      File adapter implementation (abstracted for portability)
     */
    SDCardDataset(const DatasetConfig& cfg,
                  const std::vector<std::string>& sample_files,
                  const std::vector<std::string>& label_files,
                  FileAdapter* adapter);

    /**
     * @brief Virtual destructor.
     */
    ~SDCardDataset() override;

    /**
     * @brief Fetch the next batch of samples (and optional labels).
     *
     * @param batch_size   Number of samples to fetch
     * @param input_buffer Preallocated input buffer
     * @param label_buffer Preallocated label buffer (nullptr if not used)
     * @return BatchStatus::OK if batch successfully fetched,
     *         BatchStatus::EPOCH_END if dataset end reached,
     *         BatchStatus::ERROR on failure.
     */
    BatchStatus next_batch_impl(size_t batch_size,
                    void* input_buffer,
                    void* label_buffer = nullptr) override;

    /**
     * @brief Reset dataset cursor to beginning.
     *
     * Default implementation sets @ref cursor = 0.
     */
    void reset() override;

private:
    std::vector<std::string> sample_files; ///< Paths for input sample files
    std::vector<std::string> label_files;  ///< Paths for label files (if any)
    bool has_labels;                       ///< True if labels are present

    FileAdapter* adapter; ///< File adapter used to read data from storage

    SampleT* sample_chunk_buffer; ///< Temporary buffer holding one chunk of samples
    LabelT* label_chunk_buffer;   ///< Temporary buffer holding one chunk of labels
    size_t chunk_total;         ///< Number of samples in the current chunk
    size_t current_chunk;       ///< Index of currently loaded chunk

    /**
     * @brief Load one chunk of samples (and optional labels) into internal buffers.
     *
     * @param chunk_index Index of the chunk to load
     * @return true if load successful, false otherwise
     */
    bool load_chunk(size_t chunk_index);
};
