#pragma once
#include <Arduino.h>

// MNIST input image size
#define INPUT_CHANNELS 1
#define INPUT_HEIGHT 28
#define INPUT_WIDTH 28

// ==================================
// Dataset abstraction for MNIST data
// Supports train/test (with labels) or inference (no labels)
// ==================================
class Dataset {
public:
    // Constructor for inference (no labels)
    Dataset(const uint8_t input[][INPUT_CHANNELS][INPUT_HEIGHT][INPUT_WIDTH],
            uint32_t size);

    // Constructor for train/test (with labels)
    Dataset(const uint8_t input[][INPUT_CHANNELS][INPUT_HEIGHT][INPUT_WIDTH],
            const uint8_t* target,
            uint32_t size);

    // Reset cursor
    void reset();

    // Load next batch of images (inference mode, no labels)
    void next_batch(uint32_t batch_size, float* input_buffer);

    // Load next batch of images + labels (train/test mode)
    void next_batch(uint32_t batch_size, float* input_buffer, uint8_t* target_buffer);

    // Getters
    uint32_t size() const { return dataset_size; }
    uint32_t current_index() const { return cursor; }

private:
    const uint8_t (*input_data)[INPUT_CHANNELS][INPUT_HEIGHT][INPUT_WIDTH];
    const uint8_t* target_data; // optional (nullptr if no labels)
    uint32_t dataset_size;
    uint32_t cursor;
};
