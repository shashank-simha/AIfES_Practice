#pragma once
#include <Arduino.h>
#include "mnist_data.h"

// ==================================
// Dataset abstraction for MNIST data
// ==================================
class Dataset {
public:
    Dataset(uint32_t dataset_size);

    // Reset cursor
    void reset();

    // Load next batch of images + labels into provided buffers
    void next_batch(uint32_t batch_size, float* input_buffer, uint8_t* target_buffer);

    // Getters
    uint32_t size() const { return dataset_size; }
    uint32_t current_index() const { return cursor; }

private:
    uint32_t dataset_size;
    uint32_t cursor;
};
