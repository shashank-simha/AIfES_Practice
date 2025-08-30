#include "dataset.h"
#include <pgmspace.h>

// ----------------------------
// Constructor for inference (no labels)
// ----------------------------
Dataset::Dataset(const uint8_t input[][INPUT_CHANNELS][INPUT_HEIGHT][INPUT_WIDTH],
                 uint32_t size)
    : input_data(input), target_data(nullptr), dataset_size(size), cursor(0) {}

// ----------------------------
// Constructor for train/test (with labels)
// ----------------------------
Dataset::Dataset(const uint8_t input[][INPUT_CHANNELS][INPUT_HEIGHT][INPUT_WIDTH],
                 const uint8_t* target,
                 uint32_t size)
    : input_data(input), target_data(target), dataset_size(size), cursor(0) {}

void Dataset::reset() {
    cursor = 0;
}

// ----------------------------
// Inference mode (no labels)
// ----------------------------
void Dataset::next_batch(uint32_t batch_size, float* input_buffer) {
    uint32_t input_size = INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH;

    for (uint32_t i = 0; i < batch_size; i++) {
        uint32_t idx = (cursor + i) % dataset_size;

        // Load image from PROGMEM -> buffer
        for (uint32_t c = 0; c < INPUT_CHANNELS; c++) {
            for (uint32_t h = 0; h < INPUT_HEIGHT; h++) {
                for (uint32_t w = 0; w < INPUT_WIDTH; w++) {
                    uint8_t val = pgm_read_byte(&(input_data[idx][c][h][w]));
                    input_buffer[i * input_size + c * INPUT_HEIGHT * INPUT_WIDTH + h * INPUT_WIDTH + w] =
                        static_cast<float>(val) / 255.0f;
                }
            }
        }
    }

    cursor = (cursor + batch_size) % dataset_size;
}

// ----------------------------
// Train/Test mode (with labels)
// ----------------------------
void Dataset::next_batch(uint32_t batch_size, float* input_buffer, uint8_t* target_buffer) {
    uint32_t input_size = INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH;

    for (uint32_t i = 0; i < batch_size; i++) {
        uint32_t idx = (cursor + i) % dataset_size;

        // Load image from PROGMEM -> buffer
        for (uint32_t c = 0; c < INPUT_CHANNELS; c++) {
            for (uint32_t h = 0; h < INPUT_HEIGHT; h++) {
                for (uint32_t w = 0; w < INPUT_WIDTH; w++) {
                    uint8_t val = pgm_read_byte(&(input_data[idx][c][h][w]));
                    input_buffer[i * input_size + c * INPUT_HEIGHT * INPUT_WIDTH + h * INPUT_WIDTH + w] =
                        static_cast<float>(val) / 255.0f;
                }
            }
        }

        // Load label if available
        if (target_data) {
            uint8_t label = pgm_read_byte(&(target_data[idx]));
            target_buffer[i] = label;
        }
    }

    cursor = (cursor + batch_size) % dataset_size;
}
