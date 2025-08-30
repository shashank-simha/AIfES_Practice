#pragma once
#include <Arduino.h>
#include <aifes.h>
#include <FS.h>
#include <SPIFFS.h>
#include "dataset.h"

// Model constants
#define CONV1_FILTERS 8
#define CONV2_FILTERS 16
#define KERNEL_SIZE {3, 3}
#define STRIDE {1, 1}
#define PADDING {1, 1}
#define DILATION {1, 1}
#define POOL_SIZE {2, 2}
#define POOL_STRIDE {2, 2}
#define POOL_PADDING {0, 0}
#define DENSE1_SIZE 64
#define OUTPUT_SIZE 10

// ==========================
// MNIST Model Wrapper Class
// ==========================

class MNISTModel {
public:
    MNISTModel();
    ~MNISTModel();

    // Initialize model (build layers, allocate memory)
    bool init();

    // Run inference for a single input
    uint32_t infer(float* input_buffer);

    // Test dataset end-to-end
    void test(Dataset& ds, uint32_t num_samples);

    // Train the model
    void train(Dataset& ds, uint32_t num_samples, uint32_t batch_size, uint32_t num_epoch, bool retrain);

private:
    // Model definition
    aimodel_t model;
    void* parameter_memory;
    void* training_memory;
    void* inference_memory;
    const char* params_file_path;

    // Internal helpers
    bool build_model();
    bool load_model_parameters();
    bool store_model_parameters();
    bool allocate_parameter_memory();
    void free_parameter_memory();
    bool allocate_training_memory(aiopti_t *optimizer);
    void free_training_memory();
    bool allocate_inference_memory();
    void free_inference_memory();
};
