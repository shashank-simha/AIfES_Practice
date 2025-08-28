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

// Test dataset
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
