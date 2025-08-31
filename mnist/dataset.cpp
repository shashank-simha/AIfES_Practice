#include "dataset.h"
#include <SD_MMC.h>
#include <FS.h>
#include <Arduino.h>
#include <stdlib.h>
#include <string.h>

// ----------------------------
// Constructor for inference (no labels)
// ----------------------------
Dataset::Dataset(const char* image_files[], uint32_t num_chunks)
    : image_files(image_files), label_files(nullptr),
      total_chunks(num_chunks), current_chunk(0), cursor(0),
      dataset_size(0), input_data_psram(nullptr), target_data_psram(nullptr),
      max_image_chunk_size(0), max_label_chunk_size(0) {

    if (total_chunks == 0) {
        Serial.println("Error: No image chunks provided.");
        while (1) {}
    }

    // -------- Pre-scan chunks --------
    for (uint32_t i = 0; i < total_chunks; i++) {
        File f = SD_MMC.open(image_files[i], FILE_READ);
        if (!f) {
            Serial.printf("Error: Failed to open image chunk %s\n", image_files[i]);
            while (1) {}
        }
        size_t img_size = f.size();
        f.close();

        if (img_size % (INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH) != 0) {
            Serial.printf("Error: Image chunk %s size not multiple of image dimension\n", image_files[i]);
            while (1) {}
        }

        if (img_size > max_image_chunk_size) {
            max_image_chunk_size = img_size;
        }
    }

    // Allocate PSRAM for input once (largest possible chunk)
    input_data_psram = (uint8_t*)ps_malloc(max_image_chunk_size);
    if (!input_data_psram) {
        Serial.println("Error: Failed to allocate PSRAM for input.");
        while (1) {}
    }

    // Load first chunk
    if (!load_chunk(current_chunk)) {
        Serial.println("Error: Failed to load initial chunk.");
        while (1) {}
    }
}

// ----------------------------
// Constructor for train/test (with labels)
// ----------------------------
Dataset::Dataset(const char* image_files[], const char* label_files[], uint32_t num_chunks)
    : image_files(image_files), label_files(label_files),
      total_chunks(num_chunks), current_chunk(0), cursor(0),
      dataset_size(0), input_data_psram(nullptr), target_data_psram(nullptr),
      max_image_chunk_size(0), max_label_chunk_size(0) {

    if (total_chunks == 0) {
        Serial.println("Error: No image chunks provided.");
        while (1) {}
    }

    if (!label_files) {
        Serial.println("Error: Label files array is null.");
        while (1) {}
    }

    // -------- Pre-scan chunks --------
    for (uint32_t i = 0; i < total_chunks; i++) {
        // Scan image chunk
        File f = SD_MMC.open(image_files[i], FILE_READ);
        if (!f) {
            Serial.printf("Error: Failed to open image chunk %s\n", image_files[i]);
            while (1) {}
        }
        size_t img_size = f.size();
        f.close();

        if (img_size % (INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH) != 0) {
            Serial.printf("Error: Image chunk %s size not multiple of image dimension\n", image_files[i]);
            while (1) {}
        }
        if (img_size > max_image_chunk_size) {
            max_image_chunk_size = img_size;
        }

        uint32_t num_images = img_size / (INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH);

        // Scan label chunk
        File lf = SD_MMC.open(label_files[i], FILE_READ);
        if (!lf) {
            Serial.printf("Error: Failed to open label chunk %s\n", label_files[i]);
            while (1) {}
        }
        size_t lbl_size = lf.size();
        lf.close();

        if (lbl_size != num_images) {
            Serial.printf("Error: Label count %u does not match image count %u in chunk %u\n",
                          lbl_size, num_images, i);
            while (1) {}
        }
        if (lbl_size > max_label_chunk_size) {
            max_label_chunk_size = lbl_size;
        }
    }

    // Allocate PSRAM for inputs and labels once (largest possible chunks)
    input_data_psram = (uint8_t*)ps_malloc(max_image_chunk_size);
    target_data_psram = (uint8_t*)ps_malloc(max_label_chunk_size);
    if (!input_data_psram || !target_data_psram) {
        Serial.println("Error: Failed to allocate PSRAM for input/target.");
        while (1) {}
    }

    // Load first chunk
    if (!load_chunk(current_chunk)) {
        Serial.println("Error: Failed to load initial chunk.");
        while (1) {}
    }
}

// ----------------------------
// Reset cursor to start
// ----------------------------
void Dataset::reset() {
    cursor = 0;
    current_chunk = 0;
    load_chunk(current_chunk);
}

// ----------------------------
// Load a chunk from SD into PSRAM
// ----------------------------
bool Dataset::load_chunk(uint32_t chunk_idx) {
    if (chunk_idx >= total_chunks) {
        Serial.printf("Error: Requested chunk index %u exceeds total_chunks %u\n", chunk_idx, total_chunks);
        return false;
    }

    // Read input file
    File f = SD_MMC.open(image_files[chunk_idx], FILE_READ);
    if (!f) {
        Serial.printf("Error: Failed to open image chunk %s\n", image_files[chunk_idx]);
        return false;
    }
    size_t img_size = f.size();
    f.read(input_data_psram, img_size);
    f.close();

    dataset_size = img_size / (INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH);
    Serial.printf("Loaded image chunk %u with %u samples\n", chunk_idx, dataset_size);

    // Read label file if present
    if (label_files) {
        File lf = SD_MMC.open(label_files[chunk_idx], FILE_READ);
        if (!lf) {
            Serial.printf("Error: Failed to open label chunk %s\n", label_files[chunk_idx]);
            return false;
        }
        size_t lbl_size = lf.size();
        lf.read(target_data_psram, lbl_size);
        lf.close();
        Serial.printf("Loaded label chunk %u with %u labels\n", chunk_idx, dataset_size);
    }

    cursor = 0;
    current_chunk = chunk_idx;
    return true;
}

// ----------------------------
// Next batch (inference mode)
// ----------------------------
void Dataset::next_batch(uint32_t batch_size, uint8_t* input_buffer) {
    uint32_t input_size = INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH;
    uint32_t remaining = batch_size;
    uint32_t out_idx = 0;

    while (remaining > 0) {
        uint32_t available = dataset_size - cursor;
        uint32_t take = min(remaining, available);

        memcpy(input_buffer + out_idx * input_size,
               input_data_psram + cursor * input_size,
               take * input_size);

        cursor += take;
        out_idx += take;
        remaining -= take;

        if (cursor >= dataset_size && remaining > 0) {
            if (!load_chunk((current_chunk + 1) % total_chunks)) {
                Serial.println("Fatal: Failed to load next chunk during batch read.");
                while (1) {}
            }
        }
    }
}

// ----------------------------
// Next batch (train/test mode)
// ----------------------------
void Dataset::next_batch(uint32_t batch_size, uint8_t* input_buffer, uint8_t* target_buffer) {
    uint32_t input_size = INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH;
    uint32_t remaining = batch_size;
    uint32_t out_idx = 0;

    while (remaining > 0) {
        uint32_t available = dataset_size - cursor;
        uint32_t take = min(remaining, available);

        memcpy(input_buffer + out_idx * input_size,
               input_data_psram + cursor * input_size,
               take * input_size);

        memcpy(target_buffer + out_idx,
               target_data_psram + cursor,
               take);

        cursor += take;
        out_idx += take;
        remaining -= take;

        if (cursor >= dataset_size && remaining > 0) {
            if (!load_chunk((current_chunk + 1) % total_chunks)) {
                Serial.println("Fatal: Failed to load next chunk during batch read.");
                while (1) {}
            }
        }
    }
}
