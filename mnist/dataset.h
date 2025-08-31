#include <stdint.h>
#pragma once
#include <Arduino.h>

// MNIST input image size
#define INPUT_CHANNELS 1
#define INPUT_HEIGHT 28
#define INPUT_WIDTH 28

// ==================================
// Dataset abstraction for MNIST data
// Supports train/test (with labels) or inference (no labels)
// Data is loaded in chunks from SD into PSRAM
// ==================================
class Dataset {
public:
    // Constructor for inference (no labels)
    Dataset(const char* image_files[], uint32_t num_chunks);

    // Constructor for train/test (with labels)
    Dataset(const char* image_files[], const char* label_files[], uint32_t num_chunks);

    // Reset cursor to start of dataset
    void reset();

    // Load next batch of images (inference mode, no labels)
    // Images are returned as uint8_t values [0..255]
    void next_batch(uint32_t batch_size, uint8_t* input_buffer);

    // Load next batch of images + labels (train/test mode)
    // Images and labels are returned as uint8_t
    void next_batch(uint32_t batch_size, uint8_t* input_buffer, uint8_t* target_buffer);

private:
    // Load a specific chunk from SD into PSRAM
    bool load_chunk(uint32_t chunk_idx);

    // PSRAM buffers (allocated to hold the largest chunk)
    uint8_t* input_data_psram;
    uint8_t* target_data_psram;

    // Metadata for current chunk
    uint32_t dataset_size;   // number of samples in current chunk
    uint32_t cursor;         // position in current chunk
    uint32_t current_chunk;  // index of currently loaded chunk

    // Chunk info
    const char** image_files;
    const char** label_files;
    uint32_t total_chunks;
    uint32_t max_image_chunk_size;
    uint32_t max_label_chunk_size;
};
