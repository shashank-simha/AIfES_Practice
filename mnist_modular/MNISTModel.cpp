#include "FileAdapter.h"
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "MNISTModel.h"
#include "logger.h"

// ===== Constructor / Destructor =====

/**
 * @brief Construct MNISTModel with given configuration and optional parameter file path.
 *
 * The constructor forwards the configuration to ModelBase which ensures allocator
 * and free functions are valid. The constructor does not perform heavy I/O.
 *
 * @param cfg             Model configuration (shapes, allocators, ...).
 * @param param_file_path Optional path for model parameters. If nullptr, parameters
 *                        will not be auto-loaded in init().
 * @param adapter      Pointer to a FileAdapter implementation (platform-agnostic)
 */
MNISTModel::MNISTModel(const ModelConfig& cfg,
                       const char* param_file_path,
                       FileAdapter* adapter)
    : ModelBase(cfg),
      parameter_memory(nullptr),
      training_memory(nullptr),
      inference_memory(nullptr),
      params_file_path(param_file_path),
      adapter(adapter)
{
}

/**
 * @brief Destructor: free allocated model memory.
 */
MNISTModel::~MNISTModel()
{
    free_parameter_memory();
    free_training_memory();
    free_inference_memory();
}

/**
 * @brief Set or update the path to the parameters file.
 *
 * If nullptr is passed, no parameter file will be used on init().
 *
 * @param path File path string or nullptr
 */
void MNISTModel::set_param_path(const char* path)
{
    params_file_path = path;
}

// ===== Build model =====

/**
 * @brief Build the AIfES model layer graph for MNIST.
 *
 * @return true on success, false on failure.
 */
bool MNISTModel::build_model() {
    // ===== Layers =====
    static uint16_t input_shape[] = {1, config.input_shape[0], config.input_shape[1], config.input_shape[2]};
    static ailayer_input_f32_t input_layer = AILAYER_INPUT_F32_A(4, input_shape);
    static ailayer_conv2d_f32_t conv1_layer = AILAYER_CONV2D_F32_A(CONV1_FILTERS, KERNEL_SIZE, STRIDE, DILATION, PADDING);
    static ailayer_relu_f32_t relu1_layer = AILAYER_RELU_F32_A();
    static ailayer_maxpool2d_f32_t pool1_layer = AILAYER_MAXPOOL2D_F32_A(POOL_SIZE, POOL_STRIDE, POOL_PADDING);
    static ailayer_conv2d_f32_t conv2_layer = AILAYER_CONV2D_F32_A(CONV2_FILTERS, KERNEL_SIZE, STRIDE, DILATION, PADDING);
    static ailayer_relu_f32_t relu2_layer = AILAYER_RELU_F32_A();
    static ailayer_maxpool2d_f32_t pool2_layer = AILAYER_MAXPOOL2D_F32_A(POOL_SIZE, POOL_STRIDE, POOL_PADDING);
    static ailayer_flatten_f32_t flatten_layer = AILAYER_FLATTEN_F32_A();
    static ailayer_dense_f32_t dense1_layer = AILAYER_DENSE_F32_A(DENSE1_SIZE);
    static ailayer_relu_f32_t relu3_layer = AILAYER_RELU_F32_A();
    static ailayer_dense_f32_t dense2_layer = AILAYER_DENSE_F32_A(OUTPUT_SIZE);
    static ailayer_softmax_f32_t softmax_layer = AILAYER_SOFTMAX_F32_A();

    // Layer pointer to perform the connection
    ailayer_t *x = nullptr;

    model.input_layer = ailayer_input_f32_default(&input_layer);
    if (!model.input_layer) {
        LOG_ERROR("MNISTModel: failed to create input layer");
        return false;
    }

    x = ailayer_conv2d_chw_f32_default(&conv1_layer, model.input_layer);
    if (!x) { LOG_ERROR("MNISTModel: conv1 init failed"); return false; }
    x = ailayer_relu_f32_default(&relu1_layer, x);
    if (!x) { LOG_ERROR("MNISTModel: relu1 init failed"); return false; }
    x = ailayer_maxpool2d_chw_f32_default(&pool1_layer, x);
    if (!x) { LOG_ERROR("MNISTModel: pool1 init failed"); return false; }

    x = ailayer_conv2d_chw_f32_default(&conv2_layer, x);
    if (!x) { LOG_ERROR("MNISTModel: conv2 init failed"); return false; }
    x = ailayer_relu_f32_default(&relu2_layer, x);
    if (!x) { LOG_ERROR("MNISTModel: relu2 init failed"); return false; }
    x = ailayer_maxpool2d_chw_f32_default(&pool2_layer, x);
    if (!x) { LOG_ERROR("MNISTModel: pool2 init failed"); return false; }

    x = ailayer_flatten_f32_default(&flatten_layer, x);
    if (!x) { LOG_ERROR("MNISTModel: flatten init failed"); return false; }
    x = ailayer_dense_f32_default(&dense1_layer, x);
    if (!x) { LOG_ERROR("MNISTModel: dense1 init failed"); return false; }
    x = ailayer_relu_f32_default(&relu3_layer, x);
    if (!x) { LOG_ERROR("MNISTModel: relu3 init failed"); return false; }
    x = ailayer_dense_f32_default(&dense2_layer, x);
    if (!x) { LOG_ERROR("MNISTModel: dense2 init failed"); return false; }

    model.output_layer = ailayer_softmax_f32_default(&softmax_layer, x);
    if (!model.output_layer) {
        LOG_ERROR("MNISTModel: softmax init failed");
        return false;
    }

    return true;
}

// ===== Load pretrained weights and biases (chunked read) =====

 /**
 * @brief Load model parameters from persistent storage via FileAdapter.
 *
 * Uses chunked reading for reliability on embedded filesystems.
 * The adapter is provided by the user, so this method remains
 * platform-agnostic.
 *
 * @return true on success (or if no parameter file present), false on I/O error.
 */
bool MNISTModel::load_model_parameters() {
    if (!adapter) {
        LOG_WARN("MNISTModel: No FileAdapter provided, skipping parameter load");
        return true;
    }

    if (!params_file_path) {
        LOG_WARN("MNISTModel: No parameter file path provided, skipping parameter load");
        return true;
    }

    LOG_INFO("MNISTModel: Loading model parameters from %s...", params_file_path);

    if (!adapter->open(params_file_path, FileAdapter::OpenMode::READ)) {
        LOG_INFO("MNISTModel: No saved parameters found, continuing with defaults");
        return true;
    }

    size_t expected_bytes = aialgo_sizeof_parameter_memory(&model);
    size_t actual_bytes   = adapter->size();

    if (actual_bytes != expected_bytes) {
        LOG_WARN("MNISTModel: Parameter file size mismatch (expected=%u, actual=%u)",
                 (unsigned)expected_bytes, (unsigned)actual_bytes);
        adapter->close();
        return true;
    }

    uint8_t* dst = reinterpret_cast<uint8_t*>(parameter_memory);
    const size_t CHUNK_SIZE = 1024;  // safe, can tune

    size_t total_read = 0;
    while (total_read < expected_bytes) {
        size_t to_read = std::min(CHUNK_SIZE, expected_bytes - total_read);
        size_t n = adapter->read(dst + total_read, to_read);
        if (n != to_read) {
            LOG_ERROR("MNISTModel: Read error (got %u, expected %u)",
                      (unsigned)n, (unsigned)to_read);
            adapter->close();
            return false;
        }
        total_read += n;
    }

    adapter->close();
    LOG_INFO("MNISTModel: Parameters loaded successfully (%u bytes)", (unsigned)total_read);
    return true;
}


// ===== Store current weights and biases (chunked write + temp file + rename) =====

/**
 * @brief Store model parameters to persistent storage via FileAdapter.
 *
 * Writes to a temporary file first, then renames atomically.
 *
 * @return true on success, false on failure.
 */
bool MNISTModel::store_model_parameters() {
    if (!adapter || !params_file_path) {
        LOG_WARN("MNISTModel: No adapter or file path provided, skipping store");
        return false;
    }

    LOG_INFO("MNISTModel: Storing model parameters...");

    // Temporary file path (same dir, .tmp suffix)
    char tmp_path[256];
    std::snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", params_file_path);

    // Open temporary file for writing
    if (!adapter->open(tmp_path, FileAdapter::OpenMode::WRITE)) {
        LOG_ERROR("MNISTModel: Failed to open temp file for writing");
        return false;
    }

    ailayer_t* current = model.input_layer;
    size_t total_bytes_written = 0;

    while (current) {
        for (int p = 0; p < current->trainable_params_count; p++) {
            aitensor_t* tensor = current->trainable_params[p];
            if (!tensor || !tensor->data) continue;

            uint32_t total_elements = 1;
            for (uint8_t d = 0; d < tensor->dim; d++) {
                total_elements *= tensor->shape[d];
            }

            size_t bytes = static_cast<size_t>(total_elements) * sizeof(float);
            size_t written = 0;
            const uint8_t* data_ptr = reinterpret_cast<const uint8_t*>(tensor->data);
            const size_t CHUNK_SIZE = 1024;

            while (written < bytes) {
                size_t to_write = std::min(CHUNK_SIZE, bytes - written);
                size_t ret = adapter->write(data_ptr + written, to_write);
                if (ret != to_write) {
                    LOG_ERROR("MNISTModel: Write error (got %u, expected %u)",
                              (unsigned)ret, (unsigned)to_write);
                    adapter->close();
                    adapter->remove(tmp_path); // cleanup temp
                    return false;
                }
                written += ret;
                total_bytes_written += ret;
            }
        }

        if (current == model.output_layer) break;
        current = current->output_layer;
    }

    adapter->close();

    // Remove existing file if present
    adapter->remove(params_file_path);

    // Rename tmp → final
    if (!adapter->rename(tmp_path, params_file_path)) {
        LOG_ERROR("MNISTModel: Failed to rename temp file to final");
        adapter->remove(tmp_path);
        return false;
    }

    LOG_INFO("MNISTModel: Parameters saved to %s (%u bytes)",
             params_file_path, (unsigned)total_bytes_written);
    return true;
}

// ===== Memory handling =====

/**
 * @brief Allocate parameter memory using configured allocator.
 *
 * @return true on success, false on failure.
 */
bool MNISTModel::allocate_parameter_memory() {
    size_t size = aialgo_sizeof_parameter_memory(&model);
    parameter_memory = config.allocator_fn(size);
    if (!parameter_memory) {
        LOG_ERROR("MNISTModel: Failed to allocate parameter memory (%u bytes)", (unsigned)size);
        return false;
    }
    aialgo_distribute_parameter_memory(&model, parameter_memory, static_cast<uint32_t>(size));
    LOG_INFO("MNISTModel: Parameter memory allocated: %u bytes", (unsigned)size);
    return true;
}

/**
 * @brief Free parameter memory using configured free function.
 */
void MNISTModel::free_parameter_memory() {
    if (parameter_memory) {
        config.free_fn(parameter_memory);
        parameter_memory = nullptr;
        LOG_INFO("MNISTModel: Parameter memory freed");
    }
}

/**
 * @brief Allocate training memory using configured allocator.
 *
 * @param optimizer Pointer to optimizer instance (used to compute size).
 * @return true on success, false otherwise.
 */
bool MNISTModel::allocate_training_memory(aiopti_t *optimizer) {
    size_t size = aialgo_sizeof_training_memory(&model, optimizer);
    training_memory = config.allocator_fn(size);
    if (!training_memory) {
        LOG_ERROR("MNISTModel: Failed to allocate training memory (%u bytes)", (unsigned)size);
        return false;
    }
    aialgo_schedule_training_memory(&model, optimizer, training_memory, static_cast<uint32_t>(size));
    aialgo_init_model_for_training(&model, optimizer);
    LOG_INFO("MNISTModel: Training memory allocated: %u bytes", (unsigned)size);
    return true;
}

/**
 * @brief Free training memory.
 */
void MNISTModel::free_training_memory() {
    if (training_memory) {
        config.free_fn(training_memory);
        training_memory = nullptr;
        LOG_INFO("MNISTModel: Training memory freed");
    }
}

/**
 * @brief Allocate inference memory using configured allocator.
 *
 * @return true on success, false otherwise.
 */
bool MNISTModel::allocate_inference_memory() {
    size_t size = aialgo_sizeof_inference_memory(&model);
    inference_memory = config.allocator_fn(size);
    if (!inference_memory) {
        LOG_ERROR("MNISTModel: Failed to allocate inference memory (%u bytes)", (unsigned)size);
        return false;
    }
    aialgo_schedule_inference_memory(&model, inference_memory, static_cast<uint32_t>(size));
    LOG_INFO("MNISTModel: Inference memory allocated: %u bytes", (unsigned)size);
    return true;
}

/**
 * @brief Free inference memory.
 */
void MNISTModel::free_inference_memory() {
    if (inference_memory) {
        config.free_fn(inference_memory);
        inference_memory = nullptr;
        LOG_INFO("MNISTModel: Inference memory freed");
    }
}

// ===== Public API =====

/**
 * @brief Initialize the MNIST model (build, compile, allocate memory, load params).
 *
 * @return true on success, false on failure.
 */
bool MNISTModel::init() {
    LOG_INFO("MNISTModel: Initializing...");
    if (!build_model()) {
        LOG_ERROR("MNISTModel: Layer init failed");
        return false;
    }

    aialgo_compile_model(&model);
    if (!model.output_layer) {
        LOG_ERROR("MNISTModel: Model compilation failed");
        return false;
    }

    if (!allocate_parameter_memory()) return false;

    if (params_file_path && adapter) {
        if (!load_model_parameters()) return false;
    } else {
        LOG_INFO("MNISTModel: No parameter file path or adapter provided, skipping load");
    }

    // Allocate inference memory eagerly to avoid fragmentation later
    if (!allocate_inference_memory()) return false;

    LOG_INFO("MNISTModel: Initialization complete");
    return true;
}

/**
 * @brief Run inference for one sample.
 *
 * Builds an input tensor from `config.input_shape`, runs a forward pass,
 * performs argmax on the output tensor and returns the predicted class index.
 *
 * @param input_buffer Pointer to normalized float input (C-order flattened).
 * @return Predicted class index on success, or MNIST_INVALID_CLASS on failure.
 */
uint32_t MNISTModel::infer(float* input_buffer) {
    if (!input_buffer) {
        LOG_ERROR("MNISTModel: null input_buffer");
        return MNIST_INVALID_CLASS;
    }

    // === Derive shapes from config ===
    if (config.input_shape.size() != 3 || config.output_shape.size() != 1) {
        LOG_ERROR("MNISTModel: Invalid input/output shape (in=%zu, out=%zu)",
                  config.input_shape.size(), config.output_shape.size());
        free_training_memory();
        return MNIST_INVALID_CLASS;
    }

    const uint32_t batch_size     = 1;
    const uint32_t input_channels = config.input_shape[0];
    const uint32_t input_height   = config.input_shape[1];
    const uint32_t input_width    = config.input_shape[2];

    // Create tensors
    const uint16_t input_shape[]  = { static_cast<uint16_t>(batch_size),
                                        static_cast<uint16_t>(input_channels),
                                        static_cast<uint16_t>(input_height),
                                        static_cast<uint16_t>(input_width) };

    aitensor_t input_tensor  = AITENSOR_4D_F32(input_shape, const_cast<float*>(input_buffer));

    // Forward pass
    aitensor_t* out_tensor = aialgo_forward_model(&model, &input_tensor);
    if (!out_tensor) {
        LOG_ERROR("MNISTModel: aialgo_forward_model returned null");
        return MNIST_INVALID_CLASS;
    }
    if (!out_tensor->data || out_tensor->dim == 0) {
        LOG_ERROR("MNISTModel: invalid output tensor");
        return MNIST_INVALID_CLASS;
    }

    // Compute output size
    uint32_t out_size = 1;
    for (uint8_t d = 0; d < out_tensor->dim; d++) {
        out_size *= out_tensor->shape[d];
    }
    if (out_size == 0) {
        LOG_ERROR("MNISTModel: output tensor size is zero");
        return MNIST_INVALID_CLASS;
    }

    // Argmax
    float* out = reinterpret_cast<float*>(out_tensor->data);
    uint32_t pred_class = 0;
    float max_val = out[0];
    for (uint32_t i = 1; i < out_size; ++i) {
        if (out[i] > max_val) {
            max_val = out[i];
            pred_class = i;
        }
    }

    LOG_DEBUG("MNISTModel: pred=%u, confidence=%.6f", pred_class, max_val);
    return pred_class;
}

/**
 * @brief Test the model on a dataset.
 *
 * Fetches batches one-by-one, performs normalization, runs inference,
 * and reports overall accuracy.
 *
 * @param ds DatasetBase-derived dataset
 * @param num_samples Number of samples to evaluate
 */
void MNISTModel::test(DatasetBase& ds, uint32_t num_samples) {
    LOG_INFO("MNISTModel: Testing %u images...", num_samples);

    // Derive dimensions from config instead of hardcoding
    if (config.input_shape.size() != 3) {
        LOG_ERROR("MNISTModel: Invalid input shape (expected 3 dims, got %zu)",
                  config.input_shape.size());
        return;
    }

    const uint32_t input_channels = config.input_shape[0];
    const uint32_t input_height   = config.input_shape[1];
    const uint32_t input_width    = config.input_shape[2];
    const uint32_t total_elements = input_channels * input_height * input_width;

    const size_t input_bytes_u8  = sizeof(uint8_t) * total_elements;
    const size_t input_bytes_f32 = sizeof(float) * total_elements;

    // allocate buffers via config allocator
    uint8_t* input_buffer = static_cast<uint8_t*>(config.allocator_fn(input_bytes_u8));
    uint8_t* target_buffer = static_cast<uint8_t*>(config.allocator_fn(sizeof(uint8_t)));
    float* input_normalized_buffer = static_cast<float*>(config.allocator_fn(input_bytes_f32));

    if (!input_buffer || !target_buffer || !input_normalized_buffer) {
        LOG_ERROR("MNISTModel: Buffer allocation failed (u8=%zu, f32=%zu)",
                  input_bytes_u8, input_bytes_f32);
        if (input_buffer) config.free_fn(input_buffer);
        if (target_buffer) config.free_fn(target_buffer);
        if (input_normalized_buffer) config.free_fn(input_normalized_buffer);
        return;
    }
    LOG_DEBUG("MNISTModel: Buffers allocated (u8=%zu, f32=%zu)", input_bytes_u8, input_bytes_f32);

    uint32_t correct = 0;
    for (uint32_t idx = 0; idx < num_samples; ++idx) {
        BatchStatus st = ds.fetch_batch(1, input_buffer, target_buffer);
        if (st == BatchStatus::Error) {
            LOG_ERROR("MNISTModel: Dataset fetch_batch returned error at idx=%u", idx);
            break;
        } else if (st == BatchStatus::End) {
            LOG_INFO("MNISTModel: Reached end of dataset at idx=%u", idx);
            break;
        }

        // === Normalize input ===
        // TODO: Move normalization into Dataset transform hook.
        for (uint32_t i = 0; i < total_elements; ++i) {
            float v = static_cast<float>(input_buffer[i]) / 255.0f;
            input_normalized_buffer[i] = (v - 0.1307f) / 0.3081f;
        }

        uint32_t pred = infer(input_normalized_buffer);
        uint32_t actual = static_cast<uint32_t>(target_buffer[0]);
        if (pred == actual) ++correct;

        // Progress log
        if (((idx + 1) * 100) / num_samples > (idx * 100) / num_samples) {
            LOG_PROGRESS(idx + 1, num_samples, "Images");
        }
    }

    float acc = (num_samples > 0) ? (100.0f * static_cast<float>(correct) / num_samples) : 0.0f;
    LOG_INFO("MNISTModel: Accuracy: %u/%u (%.2f%%)", correct, num_samples, acc);

    // cleanup
    config.free_fn(input_buffer);
    config.free_fn(target_buffer);
    config.free_fn(input_normalized_buffer);
    LOG_DEBUG("MNISTModel: Buffers freed after testing");
}

/**
 * @brief Train the model on a dataset.
 *
 * Performs training over multiple epochs using cross-entropy loss and SGD optimizer.
 * Handles buffer allocation, normalization, one-hot encoding, and progress logging.
 *
 * @param ds Dataset to train on (DatasetBase)
 * @param num_samples Number of samples to train on
 * @param batch_size Batch size
 * @param num_epoch Number of epochs
 * @param retrain If true, reinitialize parameters
 * @param early_stopping If true, break when target loss achieved
 * @param early_stopping_target_loss Loss threshold for early stopping
 */
void MNISTModel::train(DatasetBase& ds,
                       uint32_t num_samples,
                       uint32_t batch_size,
                       uint32_t num_epoch,
                       bool retrain,
                       bool early_stopping,
                       float early_stopping_target_loss)
{
    LOG_INFO("MNISTModel: Training on %u images, batch_size=%u, epochs=%u",
             num_samples, batch_size, num_epoch);

    // === Configure loss ===
    ailoss_crossentropy_f32_t crossentropy_loss;
    this->model.loss = ailoss_crossentropy_f32_default(&crossentropy_loss, this->model.output_layer);
    if (!this->model.loss) {
        LOG_ERROR("MNISTModel: Loss initialization failed");
        return;
    }

    // === Configure optimizer (SGD) ===
    aiopti_sgd_f32_t sgd_opti = { .learning_rate = 0.001f };
    aiopti_t* optimizer = aiopti_sgd_f32_default(&sgd_opti);
    if (!optimizer) {
        LOG_ERROR("MNISTModel: Optimizer initialization failed");
        return;
    }

    // Retrain (reinitialize params)
    if (retrain) {
        LOG_INFO("MNISTModel: Retraining from scratch (random init)");
        aialgo_initialize_parameters_model(&this->model);
    }

    // Allocate training memory
    if (!allocate_training_memory(optimizer)) {
        LOG_ERROR("MNISTModel: Failed to allocate training memory");
        return;
    }

    // === Derive shapes from config ===
    if (config.input_shape.size() != 3 || config.output_shape.size() != 1) {
        LOG_ERROR("MNISTModel: Invalid input/output shape (in=%zu, out=%zu)",
                  config.input_shape.size(), config.output_shape.size());
        free_training_memory();
        return;
    }

    const uint32_t input_channels = config.input_shape[0];
    const uint32_t input_height   = config.input_shape[1];
    const uint32_t input_width    = config.input_shape[2];
    const uint32_t output_size    = config.output_shape[0];
    const uint32_t total_elements = input_channels * input_height * input_width;

    // === Allocate buffers for one batch ===
    size_t input_bytes_u8       = static_cast<size_t>(batch_size) * total_elements * sizeof(uint8_t);
    size_t input_bytes_f32      = static_cast<size_t>(batch_size) * total_elements * sizeof(float);
    size_t target_bytes         = static_cast<size_t>(batch_size) * sizeof(uint8_t);
    size_t target_onehot_bytes  = static_cast<size_t>(batch_size) * output_size * sizeof(float);

    uint8_t* input_buffer = static_cast<uint8_t*>(config.allocator_fn(input_bytes_u8));
    uint8_t* target_buffer = static_cast<uint8_t*>(config.allocator_fn(target_bytes));
    float* input_normalized_buffer = static_cast<float*>(config.allocator_fn(input_bytes_f32));
    float* target_onehot_buffer = static_cast<float*>(config.allocator_fn(target_onehot_bytes));

    if (!input_buffer || !target_buffer || !input_normalized_buffer || !target_onehot_buffer) {
        LOG_ERROR("MNISTModel: Buffer allocation failed (u8=%zu, f32=%zu, onehot=%zu)",
                  input_bytes_u8, input_bytes_f32, target_onehot_bytes);
        if (input_buffer) config.free_fn(input_buffer);
        if (target_buffer) config.free_fn(target_buffer);
        if (input_normalized_buffer) config.free_fn(input_normalized_buffer);
        if (target_onehot_buffer) config.free_fn(target_onehot_buffer);
        free_training_memory();
        return;
    }
    LOG_DEBUG("MNISTModel: Buffers allocated (u8=%zu, f32=%zu, onehot=%zu)",
              input_bytes_u8, input_bytes_f32, target_onehot_bytes);

    // === Training loop ===
    for (uint32_t epoch = 0; epoch < num_epoch; ++epoch) {
        uint32_t steps = (batch_size > 0) ? (num_samples / batch_size) : 0;
        float epoch_loss = 0.0f;

        for (uint32_t step = 0; step < steps; ++step) {
            BatchStatus st = ds.fetch_batch(batch_size, input_buffer, target_buffer);
            if (st == BatchStatus::Error) {
                LOG_ERROR("MNISTModel: Dataset fetch_batch returned error at step=%u", step);
                goto training_cleanup;
            } else if (st == BatchStatus::End) {
                LOG_INFO("MNISTModel: Dataset reached end during training at step=%u", step);
                goto training_cleanup;
            }

            // === Normalize inputs ===
            // TODO: Move normalization into Dataset transform hook.
            uint32_t total_batch_elements = batch_size * total_elements;
            for (uint32_t i = 0; i < total_batch_elements; ++i) {
                input_normalized_buffer[i] = static_cast<float>(input_buffer[i]) / 255.0f;
            }

            // === One-hot encode labels ===
            // TODO: Move one-hot encoding into Dataset transform hook.
            for (uint32_t i = 0; i < batch_size; ++i) {
                for (uint32_t c = 0; c < output_size; ++c) {
                    target_onehot_buffer[i * output_size + c] =
                        (target_buffer[i] == c) ? 1.0f : 0.0f;
                }
            }

            // Create tensors
            const uint16_t input_shape[]  = { static_cast<uint16_t>(batch_size),
                                              static_cast<uint16_t>(input_channels),
                                              static_cast<uint16_t>(input_height),
                                              static_cast<uint16_t>(input_width) };
            const uint16_t target_shape[] = { static_cast<uint16_t>(batch_size),
                                              static_cast<uint16_t>(output_size) };
            aitensor_t input_tensor  = AITENSOR_4D_F32(input_shape, input_normalized_buffer);
            aitensor_t target_tensor = AITENSOR_2D_F32(target_shape, target_onehot_buffer);

            // Train step
            aialgo_train_model(&this->model, &input_tensor, &target_tensor, optimizer, batch_size);

            // Track batch loss
            float batch_loss = 0.0f;
            aialgo_calc_loss_model_f32(&this->model, &input_tensor, &target_tensor, &batch_loss);
            epoch_loss += batch_loss;

            // Progress logging
            if (((step + 1) * 100) / (steps > 0 ? steps : 1) >
                ((step) * 100) / (steps > 0 ? steps : 1)) {
                char metric[64];
                std::snprintf(metric, sizeof(metric), "Epoch %u/%u", epoch + 1, num_epoch);
                LOG_PROGRESS(step + 1, steps, metric);
            }
        }

        // End of epoch
        if (steps > 0) epoch_loss /= static_cast<float>(steps);
        LOG_INFO("MNISTModel: Epoch %u/%u - Loss: %.4f", epoch + 1, num_epoch, epoch_loss);
        store_model_parameters();

        if (early_stopping && (epoch_loss < early_stopping_target_loss)) {
            LOG_INFO("MNISTModel: Early stopping triggered: epoch_loss %.4f < target %.4f",
                     epoch_loss, early_stopping_target_loss);
            break;
        }
    }

training_cleanup:
    LOG_INFO("MNISTModel: Finished training");

    // Cleanup
    config.free_fn(input_buffer);
    config.free_fn(target_buffer);
    config.free_fn(input_normalized_buffer);
    config.free_fn(target_onehot_buffer);
    free_training_memory();
    LOG_DEBUG("MNISTModel: Buffers freed after training");
}
