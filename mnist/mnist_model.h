#pragma once
#include <Arduino.h>
#include <aifes.h>
#include "mnist_weights.h"
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
#define LAYER_COUNT 11        // Input → conv1 → relu1 → pool1 → conv2 → relu2 → pool2 → flatten → dense1 → relu3 → softmax

#define PRETRAINED_WEIGHTS true

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
    void train(Dataset& ds, uint32_t num_samples, uint32_t batch_size, uint32_t num_epoch);

private:
    // Model definition
    aimodel_t model;
    ailayer_t* layers[LAYER_COUNT];
    void* parameter_memory;
    void* training_memory;
    void* inference_memory;

    // Internal helpers
    bool build_model();
    bool load_weights();
    bool allocate_parameter_memory();
    void free_parameter_memory();
    bool allocate_training_memory(aiopti_t *optimizer);
    void free_training_memory();
    bool allocate_inference_memory();
    void free_inference_memory();
};
