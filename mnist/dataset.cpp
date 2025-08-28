#include "dataset.h"

// test_input_data: uint8_t [N][1][28][28] stored in PROGMEM
// test_target_data: uint8_t [N] stored in PROGMEM

Dataset::Dataset(uint32_t size) : dataset_size(size), cursor(0) {}

void Dataset::reset() {
    cursor = 0;
}

void Dataset::next_batch(uint32_t batch_size, float* input_buffer, uint8_t* target_buffer) {
    uint32_t input_size = 1 * 28 * 28;

    for (uint32_t i = 0; i < batch_size; i++) {
        uint32_t idx = (cursor + i) % dataset_size;

        // Load image from PROGMEM -> buffer
        for (uint32_t j = 0; j < input_size; j++) {
            uint8_t val = pgm_read_byte(&(test_input_data[idx][0][0][j]));
            input_buffer[i * input_size + j] = static_cast<float>(val) / 255.0f; // normalize to [0,1]
        }

        // Load label
        uint8_t label = pgm_read_byte(&(test_target_data[idx]));
        target_buffer[i] = label;
    }

    cursor = (cursor + batch_size) % dataset_size;
}
