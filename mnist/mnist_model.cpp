#include "mnist_model.h"

// ===== Constructor / Destructor =====
MNISTModel::MNISTModel() : parameter_memory(nullptr), training_memory(nullptr), inference_memory(nullptr)
{
    this->params_file_path = "/params.bin";
}

MNISTModel::~MNISTModel()
{ 
    free_parameter_memory();
    free_training_memory();
    free_inference_memory();
}

// ===== Build model =====
bool MNISTModel::build_model() {
    // ===== Layers =====
    static uint16_t input_shape[] = {1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH};
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
    ailayer_t *x;    

    model.input_layer = ailayer_input_f32_default(&input_layer);
    if (!model.input_layer) return false;

    x = ailayer_conv2d_chw_f32_default(&conv1_layer, model.input_layer);
    if (!x) return false;
    x = ailayer_relu_f32_default(&relu1_layer, x);
    if (!x) return false;
    x = ailayer_maxpool2d_chw_f32_default(&pool1_layer, x);
    if (!x) return false;

    x = ailayer_conv2d_chw_f32_default(&conv2_layer, x);
    if (!x) return false;
    x = ailayer_relu_f32_default(&relu2_layer, x);
    if (!x) return false;
    x = ailayer_maxpool2d_chw_f32_default(&pool2_layer, x);
    if (!x) return false;

    x = ailayer_flatten_f32_default(&flatten_layer, x);
    if (!x) return false;
    x = ailayer_dense_f32_default(&dense1_layer, x);
    if (!x) return false;
    x = ailayer_relu_f32_default(&relu3_layer, x);
    if (!x) return false;
    x = ailayer_dense_f32_default(&dense2_layer, x);
    if (!x) return false;

    model.output_layer = ailayer_softmax_f32_default(&softmax_layer, x);
    if (!model.output_layer) return false;

    return true;
}

// ===== Load pretrained weights and biases (chunked read + file size check) =====
bool MNISTModel::load_model_parameters() {
    Serial.println("Loading model parameters.......");

    if (!SPIFFS.begin(false)) { // don't format on mount failure
        Serial.println("SPIFFS Mount Failed, continuing with current parameters");
        return true;
    }

    File file = SPIFFS.open(this->params_file_path, FILE_READ);
    if (!file) {
        Serial.println("No saved parameters found, continuing with current parameters");
        return true;
    }

    size_t expected_bytes = aialgo_sizeof_parameter_memory(&model);
    size_t actual_bytes = file.size();

    if (actual_bytes != expected_bytes) {
        Serial.printf("Warning: file size mismatch! expected=%u, actual=%u\n", (unsigned)expected_bytes, (unsigned)actual_bytes);
        Serial.println("Continuing with current parameters");
        return true;
    }

    ailayer_t* current = model.input_layer;
    while (current) {
        for (int p = 0; p < current->trainable_params_count; p++) {
            aitensor_t* tensor = current->trainable_params[p];
            if (!tensor || !tensor->data) continue;

            uint32_t total_elements = 1;
            for (uint8_t d = 0; d < tensor->dim; d++) {
                total_elements *= tensor->shape[d];
            }

            size_t bytes = total_elements * sizeof(float);
            if (file.available() < bytes) {
                Serial.println("Fatal: file too small, corrupted parameters?");
                file.close();
                return false;
            }

            size_t read_bytes = 0;
            uint8_t* data_ptr = (uint8_t*)tensor->data;
            const size_t CHUNK_SIZE = 1024;

            while (read_bytes < bytes) {
                size_t to_read = min(CHUNK_SIZE, bytes - read_bytes);
                size_t ret = file.read(data_ptr + read_bytes, to_read);
                if (ret != to_read) {
                    Serial.printf("Fatal: read %u bytes, expected %u\n", (unsigned)ret, (unsigned)to_read);
                    file.close();
                    return false;
                }
                read_bytes += ret;
            }
        }

        if (current == model.output_layer) break;
        current = current->output_layer;
    }

    file.close();
    Serial.printf("Parameters loaded from %s\n", this->params_file_path);
    return true;
}

// ===== Store current weights and biases (chunked write + temp file + rename) =====
bool MNISTModel::store_model_parameters() {
    Serial.println("Storing model parameters.......");

    if (!SPIFFS.begin(false)) { // don't format on mount failure
        Serial.println("SPIFFS Mount Failed, skipping storing parameters");
        return false;
    }

    char tmp_path[64];
    sprintf(tmp_path, "%s.tmp", this->params_file_path);
    File file = SPIFFS.open(tmp_path, FILE_WRITE);
    if (!file) {
        Serial.println("Failed to open temp file for writing, skipping storing parameters");
        return true;
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

            size_t bytes = total_elements * sizeof(float);
            size_t written = 0;
            uint8_t* data_ptr = (uint8_t*)tensor->data;
            const size_t CHUNK_SIZE = 1024;

            while (written < bytes) {
                size_t to_write = min(CHUNK_SIZE, bytes - written);
                size_t ret = file.write(data_ptr + written, to_write);
                if (ret != to_write) {
                    Serial.printf("Warning: wrote %u bytes, expected %u\n", (unsigned)ret, (unsigned)to_write);
                    Serial.println("Skipping storing parameters");
                    return true;
                }
                written += ret;
                total_bytes_written += ret;
            }
        }

        if (current == model.output_layer) break;
        current = current->output_layer;
    }

    file.close();

    // Delete existing final file if it exists
    if (SPIFFS.exists(this->params_file_path)) {
        if (!SPIFFS.remove(this->params_file_path)) {
            Serial.println("Failed to remove existing final file!, Skipping storing parameters");
            return true;
        }
    }

    // Rename temp file -> final file
    if (!SPIFFS.rename(tmp_path, this->params_file_path)) {
        Serial.println("Fatal: Failed to rename temp file to final file!");
        return false;
    }

    Serial.printf("Parameters saved to %s, total bytes written=%u\n", this->params_file_path, (unsigned)total_bytes_written);
    return true;
}

// ===== Memory handling =====
bool MNISTModel::allocate_parameter_memory() {
    uint32_t size = aialgo_sizeof_parameter_memory(&model);
    parameter_memory = ps_malloc(size);
    if (!parameter_memory) return false;
    aialgo_distribute_parameter_memory(&model, parameter_memory, size);
    Serial.printf("Parameter memory allocated: %u bytes, Free PSRAM: %u bytes\n",
                size, ESP.getFreePsram());
    return true;
}

void MNISTModel::free_parameter_memory() {
    if (parameter_memory) {
        free(parameter_memory);
        parameter_memory = nullptr;
        Serial.printf("Parameter memory freed, Free PSRAM: %u bytes\n", ESP.getFreePsram());
    }
}

bool MNISTModel::allocate_training_memory(aiopti_t *optimizer) {
    uint32_t size = aialgo_sizeof_training_memory(&model, optimizer);
    training_memory = ps_malloc(size);
    if (!training_memory) return false;
    aialgo_schedule_training_memory(&model, optimizer, training_memory, size);
    aialgo_init_model_for_training(&model, optimizer);
    Serial.printf("Training memory allocated: %u bytes, Free PSRAM: %u bytes\n",
                size, ESP.getFreePsram());
    return true;
}

void MNISTModel::free_training_memory() {
    if (training_memory) {
        free(training_memory);
        training_memory = nullptr;
        Serial.printf("Training memory freed, Free PSRAM: %u bytes\n", ESP.getFreePsram());
    }
}

bool MNISTModel::allocate_inference_memory() {
    uint32_t size = aialgo_sizeof_inference_memory(&model);
    inference_memory = ps_malloc(size);
    if (!inference_memory) return false;
    aialgo_schedule_inference_memory(&model, inference_memory, size);
    Serial.printf("Inference memory allocated: %u bytes, Free PSRAM: %u bytes\n",
                size, ESP.getFreePsram());
    return true;
}

void MNISTModel::free_inference_memory() {
    if (inference_memory) {
        free(inference_memory);
        inference_memory = nullptr;
        Serial.printf("Inference memory freed, Free PSRAM: %u bytes\n", ESP.getFreePsram());
    }
}

// ===== Public API =====
bool MNISTModel::init() {
    Serial.println(F("Initializing MNISTModel..."));
    if (!build_model()) {
        Serial.println(F("Layer init failed"));
        return false;
    }
    aialgo_compile_model(&model);
    if (!model.output_layer) {
        Serial.println(F("Model compilation failed"));
        return false;
    }

    if(!allocate_parameter_memory())
        return false;

    if(!load_model_parameters())
        return false;

    /* Allocate inference memory during model rather than during inference/test to avoid fragmentation */
    return allocate_inference_memory();
}

// Run inference for one sample
uint32_t MNISTModel::infer(float* input_buffer) {
    const uint16_t shape[] = {1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH};
    aitensor_t input_tensor = AITENSOR_4D_F32(shape, input_buffer);

    aitensor_t* out_tensor = aialgo_forward_model(&model, &input_tensor);
    if (!out_tensor) return UINT32_MAX;

    float* out = (float*)out_tensor->data;
    uint32_t out_size = 1;
    for (uint8_t d = 0; d < out_tensor->dim; d++) out_size *= out_tensor->shape[d];

    uint32_t pred_class = 0;
    float max_val = out[0];
    for (uint32_t i = 1; i < out_size; i++) {
        if (out[i] > max_val) {
            max_val = out[i];
            pred_class = i;
        }
    }
    return pred_class;
}

// // Test dataset
void MNISTModel::test(Dataset& ds, uint32_t num_samples) {
    Serial.printf("Testing %u images...\n", num_samples);

    float* input_buffer = (float*)ps_malloc(INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH * sizeof(float));
    uint8_t* target_buffer = (uint8_t*)ps_malloc(sizeof(uint8_t));
    if (!input_buffer || !target_buffer) {
        Serial.println(F("Buffer allocation failed"));
        return;
    }

    uint32_t correct = 0;
    for (uint32_t i = 0; i < num_samples; i++) {
        ds.next_batch(1, input_buffer, target_buffer);
        uint32_t pred = infer(input_buffer);
        uint32_t actual = target_buffer[0];
        if (pred == actual) correct++;
        Serial.printf("Image %d: Predicted %u, Actual %u, %s\n",
                      i, pred, actual, pred == actual ? "Correct" : "Wrong");
    }

    float acc = 100.0f * correct / num_samples;
    Serial.printf("Accuracy: %u/%u (%.2f%%)\n", correct, num_samples, acc);

    free(input_buffer);
    free(target_buffer);
}


// Train dataset
void MNISTModel::train(Dataset& ds, uint32_t num_samples, uint32_t batch_size, uint32_t num_epoch) {
    Serial.printf("Training on %u images...\n", num_samples);

    // Configure cross-entropy loss
    ailoss_crossentropy_f32_t crossentropy_loss;
    this->model.loss = ailoss_crossentropy_f32_default(&crossentropy_loss, this->model.output_layer);
    if (!this->model.loss) {
        Serial.println(F("Loss initialization failed"));
        return;
    }

    // Configure SGD optimizer
    aiopti_sgd_f32_t sgd_opti = { .learning_rate = 0.001f };
    aiopti_t* optimizer = aiopti_sgd_f32_default(&sgd_opti);
    if (!optimizer) {
        Serial.println(F("Optimizer initialization failed"));
        return;
    }

    // Initialize model parameters + training memory
    // aialgo_initialize_parameters_model(&this->model);
    if(!allocate_training_memory(optimizer))
        return;

    // Allocate input/target buffers for one batch
    float* input_buffer = (float*)ps_malloc(batch_size * INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH * sizeof(float));
    uint8_t* target_buffer = (uint8_t*)ps_malloc(batch_size * sizeof(uint8_t));
    float* target_onehot_buffer = (float*)ps_malloc(batch_size * OUTPUT_SIZE * sizeof(float));
    if (!input_buffer || !target_buffer || !target_onehot_buffer) {
        Serial.println(F("Buffer allocation failed"));
        if (input_buffer) free(input_buffer);
        if (target_buffer) free(target_buffer);
        if (target_onehot_buffer) free(target_onehot_buffer);
        free_training_memory();
        return;
    }

    // Training loop
    for (uint32_t epoch = 0; epoch < num_epoch; epoch++) {
        uint32_t steps = num_samples / batch_size;
        float epoch_loss = 0.0f;

        for (uint32_t step = 0; step < steps; step++) {
            // Get batch
            ds.next_batch(batch_size, input_buffer, target_buffer);

            // Convert labels → one-hot float32
            for (uint32_t i = 0; i < batch_size; i++) {
                for (uint32_t c = 0; c < OUTPUT_SIZE; c++) {
                    target_onehot_buffer[i * OUTPUT_SIZE + c] =
                        (target_buffer[i] == c) ? 1.0f : 0.0f;
                }
            }

            // Create tensors
            const uint16_t input_shape[] = { batch_size, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH };
            const uint16_t target_shape[] = { batch_size, OUTPUT_SIZE };
            aitensor_t input_tensor  = AITENSOR_4D_F32(input_shape, input_buffer);
            aitensor_t target_tensor = AITENSOR_2D_F32(target_shape, target_onehot_buffer);

            // Train step
            aialgo_train_model(&this->model, &input_tensor, &target_tensor, optimizer, batch_size);

            // Track batch loss
            float batch_loss;
            aialgo_calc_loss_model_f32(&this->model, &input_tensor, &target_tensor, &batch_loss);
            epoch_loss += batch_loss;
        }

        // Print per-epoch loss
        epoch_loss /= steps;
        Serial.printf("Epoch %u/%u - Loss: %.4f\n", epoch + 1, num_epoch, epoch_loss);
        store_model_parameters();
    }

    aiprint("Finished training\n");

    // Cleanup
    free(input_buffer);
    free(target_buffer);
    free(target_onehot_buffer);
    free_training_memory();
}