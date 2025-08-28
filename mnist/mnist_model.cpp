#include "mnist_model.h"

// ===== Layers (weights already included via mnist_weights.h) =====
static uint16_t input_shape[] = {1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH};

static ailayer_input_f32_t input_layer = AILAYER_INPUT_F32_M(4, input_shape);

static ailayer_conv2d_f32_t conv1_layer = AILAYER_CONV2D_F32_M(
    CONV1_FILTERS, KERNEL_SIZE, STRIDE, DILATION, PADDING, conv1_weights, conv1_bias);
static ailayer_relu_f32_t relu1_layer = AILAYER_RELU_F32_M();
static ailayer_maxpool2d_f32_t pool1_layer = AILAYER_MAXPOOL2D_F32_M(POOL_SIZE, POOL_STRIDE, POOL_PADDING);

static ailayer_conv2d_f32_t conv2_layer = AILAYER_CONV2D_F32_M(
    CONV2_FILTERS, KERNEL_SIZE, STRIDE, DILATION, PADDING, conv2_weights, conv2_bias);
static ailayer_relu_f32_t relu2_layer = AILAYER_RELU_F32_M();
static ailayer_maxpool2d_f32_t pool2_layer = AILAYER_MAXPOOL2D_F32_M(POOL_SIZE, POOL_STRIDE, POOL_PADDING);

static ailayer_flatten_f32_t flatten_layer = AILAYER_FLATTEN_F32_M();
static ailayer_dense_f32_t dense1_layer = AILAYER_DENSE_F32_M(DENSE1_SIZE, fc1_weights, fc1_bias);
static ailayer_relu_f32_t relu3_layer = AILAYER_RELU_F32_M();
static ailayer_dense_f32_t dense2_layer = AILAYER_DENSE_F32_M(OUTPUT_SIZE, fc2_weights, fc2_bias);
static ailayer_softmax_f32_t softmax_layer = AILAYER_SOFTMAX_F32_M();

// ===== Constructor / Destructor =====
MNISTModel::MNISTModel() : inference_memory(nullptr) {}
MNISTModel::~MNISTModel() { free_inference_memory(); }

// ===== Build model =====
bool MNISTModel::build_model() {
    layers[0] = model.input_layer = ailayer_input_f32_default(&input_layer);
    if (!model.input_layer) return false;

    conv1_layer.channel_axis = 1;
    layers[1] = ailayer_conv2d_f32_default(&conv1_layer, model.input_layer);
    layers[2] = ailayer_relu_f32_default(&relu1_layer, layers[1]);
    pool1_layer.channel_axis = 1;
    layers[3] = ailayer_maxpool2d_f32_default(&pool1_layer, layers[2]);

    conv2_layer.channel_axis = 1;
    layers[4] = ailayer_conv2d_f32_default(&conv2_layer, layers[3]);
    layers[5] = ailayer_relu_f32_default(&relu2_layer, layers[4]);
    pool2_layer.channel_axis = 1;
    layers[6] = ailayer_maxpool2d_f32_default(&pool2_layer, layers[5]);

    layers[7] = ailayer_flatten_f32_default(&flatten_layer, layers[6]);
    layers[8] = ailayer_dense_f32_default(&dense1_layer, layers[7]);
    layers[9] = ailayer_relu_f32_default(&relu3_layer, layers[8]);

    layers[10] = model.output_layer = ailayer_softmax_f32_default(
        &softmax_layer, ailayer_dense_f32_default(&dense2_layer, layers[9]));

    return model.output_layer != nullptr;
}

// ===== Memory handling =====
bool MNISTModel::allocate_inference_memory() {
    uint32_t size = aialgo_sizeof_inference_memory(&model);
    inference_memory = ps_malloc(size);
    if (!inference_memory) return false;
    aialgo_schedule_inference_memory(&model, inference_memory, size);
    Serial.printf("Inference memory allocated: %u bytes\n", size);
    return true;
}

void MNISTModel::free_inference_memory() {
    if (inference_memory) {
        free(inference_memory);
        inference_memory = nullptr;
        Serial.println(F("Inference memory freed"));
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


// // Train dataset
// void MNISTModel::train(Dataset& ds, uint32_t num_samples) {
//     Serial.printf("Training on %u images...\n", num_samples);

//     // Allocate input/target buffers for one batch
//     float* input_buffer = (float*)ps_malloc(BATCH_SIZE * INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH * sizeof(float));
//     uint8_t* target_buffer = (uint8_t*)ps_malloc(BATCH_SIZE * sizeof(uint8_t));
//     if (!input_buffer || !target_buffer) {
//         Serial.println(F("Buffer allocation failed"));
//         return;
//     }

//     // Configure cross-entropy loss
//     ailoss_crossentropy_f32_t crossentropy_loss;
//     this->loss = ailoss_crossentropy_f32_default(&crossentropy_loss, this->output_layer);
//     if (!this->loss) {
//         Serial.println(F("Loss initialization failed"));
//         return;
//     }

//     // Configure SGD optimizer
//     aiopti_sgd_f32_t sgd_opti = { .learning_rate = LEARNING_RATE };
//     aiopti_t* optimizer = aiopti_sgd_f32_default(&sgd_opti);
//     if (!optimizer) {
//         Serial.println(F("Optimizer initialization failed"));
//         return;
//     }

//     // Initialize model parameters + training memory
//     aialgo_initialize_parameters_model(&this->model);
//     uint32_t training_memory_size = aialgo_sizeof_training_memory(&this->model, optimizer);
//     void* training_memory = ps_malloc(training_memory_size);
//     if (!training_memory) {
//         Serial.println(F("Training memory allocation failed"));
//         return;
//     }
//     aialgo_schedule_training_memory(&this->model, optimizer, training_memory, training_memory_size);
//     aialgo_init_model_for_training(&this->model, optimizer);

//     // Training loop
//     aiprint("Start training\n");
//     for (uint32_t epoch = 0; epoch < EPOCHS; epoch++) {
//         uint32_t steps = num_samples / BATCH_SIZE;
//         float epoch_loss = 0.0f;

//         for (uint32_t step = 0; step < steps; step++) {
//             // Get batch
//             ds.next_batch(BATCH_SIZE, input_buffer, target_buffer);

//             // Create tensors
//             const uint16_t input_shape[] = { BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH };
//             const uint16_t target_shape[] = { BATCH_SIZE };
//             aitensor_t input_tensor  = AITENSOR_4D_F32(input_shape, input_buffer);
//             aitensor_t target_tensor = AITENSOR_1D_U8(target_shape, target_buffer);

//             // Train step
//             aialgo_train_model(&this->model, &input_tensor, &target_tensor, optimizer, BATCH_SIZE);

//             // Track batch loss
//             float batch_loss;
//             aialgo_calc_loss_model_f32(&this->model, &input_tensor, &target_tensor, &batch_loss);
//             epoch_loss += batch_loss;
//         }

//         // Print per-epoch loss
//         epoch_loss /= steps;
//         Serial.printf("Epoch %u/%u - Loss: %.4f\n", epoch + 1, EPOCHS, epoch_loss);
//     }

//     aiprint("Finished training\n");

//     // Cleanup
//     free(input_buffer);
//     free(target_buffer);
//     free(training_memory);
//     Serial.printf("Training memory freed, Free PSRAM: %u bytes\n", ESP.getFreePsram());
// }


// // Inside mnist_model.h or mnist_model.cpp

// void MNISTModel::train() {
//   Serial.println(F("Training..."));
// #if DEBUG
//   Serial.printf("Free SRAM before: %u bytes\n", ESP.getFreeHeap());
// #endif

//   // Configure cross-entropy loss
//   ailoss_crossentropy_f32_t crossentropy_loss;
//   this->loss = ailoss_crossentropy_f32_default(&crossentropy_loss, this->output_layer);
//   if (!this->loss) {
//     Serial.println(F("Loss initialization failed"));
//     while (1);
//   }
// #if DEBUG
//   aiprint("\nLoss specs:\n");
//   aialgo_print_loss_specs(this->loss);
//   aiprint("\n");
// #endif

//   // Configure SGD optimizer
//   aiopti_sgd_f32_t sgd_opti = { .learning_rate = LEARNING_RATE };
//   aiopti_t *optimizer = aiopti_sgd_f32_default(&sgd_opti);
//   if (!optimizer) {
//     Serial.println(F("Optimizer initialization failed"));
//     while (1);
//   }
// #if DEBUG
//   aiprint("Optimizer specs:\n");
//   aialgo_print_optimizer_specs(optimizer);
//   aiprint("\n");
// #endif

//   // Initialize model parameters
//   aialgo_initialize_parameters_model(&this->model);
//   Serial.println(F("Parameters initialized"));

//   // Allocate training memory in PSRAM
//   uint32_t training_memory_size = aialgo_sizeof_training_memory(&this->model, optimizer);
//   void *training_memory = ps_malloc(training_memory_size);
//   if (!training_memory) {
//     Serial.println(F("Training memory allocation failed"));
//     while (1);
//   }
//   aialgo_schedule_training_memory(&this->model, optimizer, training_memory, training_memory_size);
//   aialgo_init_model_for_training(&this->model, optimizer);
//   Serial.printf("Training memory allocated: %u bytes, Free PSRAM: %u bytes\n",
//                 training_memory_size, ESP.getFreePsram());

//   // Input/target tensors from PROGMEM
//   const uint16_t input_shape[]  = { TRAIN_DATASET, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH };
//   const uint16_t target_shape[] = { TRAIN_DATASET }; // labels are uint8 (class indices)
//   aitensor_t input_tensor  = AITENSOR_4D_U8(input_shape, (uint8_t *)train_input_data);
//   aitensor_t target_tensor = AITENSOR_1D_U8(target_shape, (uint8_t *)train_target_data);

// #if DEBUG
//   Serial.printf("Input tensor shape: [%u,%u,%u,%u]\n",
//                 input_tensor.shape[0], input_tensor.shape[1],
//                 input_tensor.shape[2], input_tensor.shape[3]);
//   Serial.printf("Target tensor shape: [%u]\n", target_tensor.shape[0]);
//   Serial.printf("Free SRAM after tensors: %u bytes\n", ESP.getFreeHeap());
// #endif

//   // Test forward pass
//   Serial.println(F("Testing forward pass"));
//   aitensor_t *output_tensor = aialgo_forward_model(&this->model, &input_tensor);
//   if (!output_tensor) {
//     Serial.println(F("Forward pass failed"));
//     while (1);
//   }
//   Serial.println(F("Forward pass completed"));

//   // Training loop
//   float loss;
//   aiprint("Start training\n");
//   for (int i = 0; i < EPOCHS; i++) {
//     aialgo_train_model(&this->model, &input_tensor, &target_tensor, optimizer, BATCH_SIZE);

//     if (i % PRINT_INTERVAL == 0) {
//       aialgo_calc_loss_model_f32(&this->model, &input_tensor, &target_tensor, &loss);
//       aiprint("Epoch ");
//       aiprint_int("%5d", i);
//       aiprint(": train loss: ");
//       aiprint_float("%f", loss);
//       aiprint("\n");
//     }
//   }
//   aiprint("Finished training\n");

//   // Free training memory
//   free(training_memory);
//   Serial.printf("Training memory freed, Free PSRAM: %u bytes\n", ESP.getFreePsram());
// }

