#include <aifes.h>
#include <esp_heap_caps.h>
#include "mnist_data.h"
#include "mnist_weights.h"

// Debug logs
#define DEBUG 1

// Model constants
#define INPUT_CHANNELS 1
#define INPUT_HEIGHT 28
#define INPUT_WIDTH 28
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
#define NUM_CLASSES 10
#define LAYER_COUNT 11   // Added dense2 layer
#define TEST_DATASET 1

// Global model variables
aimodel_t model;
ailayer_t *layers[LAYER_COUNT];

// Layer structures
uint16_t input_shape[] = {1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH};
ailayer_input_f32_t input_layer = AILAYER_INPUT_F32_M(4, input_shape);

ailayer_conv2d_f32_t conv1_layer = AILAYER_CONV2D_F32_M(CONV1_FILTERS, KERNEL_SIZE, STRIDE, DILATION, PADDING, conv1_weights, conv1_bias);
ailayer_relu_f32_t relu1_layer = AILAYER_RELU_F32_M();
ailayer_maxpool2d_f32_t pool1_layer = AILAYER_MAXPOOL2D_F32_M(POOL_SIZE, POOL_STRIDE, POOL_PADDING);

ailayer_conv2d_f32_t conv2_layer = AILAYER_CONV2D_F32_M(CONV2_FILTERS, KERNEL_SIZE, STRIDE, DILATION, PADDING, conv2_weights, conv2_bias);
ailayer_relu_f32_t relu2_layer = AILAYER_RELU_F32_M();
ailayer_maxpool2d_f32_t pool2_layer = AILAYER_MAXPOOL2D_F32_M(POOL_SIZE, POOL_STRIDE, POOL_PADDING);

ailayer_flatten_f32_t flatten_layer = AILAYER_FLATTEN_F32_M();
ailayer_dense_f32_t dense1_layer = AILAYER_DENSE_F32_M(DENSE1_SIZE, fc1_weights, fc1_bias);
ailayer_relu_f32_t relu3_layer = AILAYER_RELU_F32_M();

// New dense2 layer → final output
ailayer_dense_f32_t dense2_layer = AILAYER_DENSE_F32_M(NUM_CLASSES, fc2_weights, fc2_bias);

// Initialize layers
void init_layers() {
  layers[0] = model.input_layer = ailayer_input_f32_default(&input_layer);

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

  // New final dense layer
  layers[10] = model.output_layer = ailayer_dense_f32_default(&dense2_layer, layers[9]);

  if (!model.output_layer) {
    Serial.println(F("Layer init failed"));
    while (1);
  }
}

// Initialize model
void init_model() {
  Serial.println(F("Initializing model..."));
  init_layers();
  aialgo_compile_model(&model);
  if (!model.output_layer) {
    Serial.println(F("Model compilation failed"));
    while (1);
  }
  Serial.println(F("Model initialized"));
}

// Test on one image
void test() {
  Serial.println(F("Testing up to Dense2..."));

  // Allocate inference memory
  uint32_t inference_memory_size = aialgo_sizeof_inference_memory(&model);
  void *inference_memory = ps_malloc(inference_memory_size);
  if (!inference_memory) {
    Serial.println(F("Inference memory allocation failed"));
    return;
  }
  aialgo_schedule_inference_memory(&model, inference_memory, inference_memory_size);
  Serial.printf("Inference memory allocated: %u bytes\n", inference_memory_size);

  // Allocate input buffer dynamically
  uint32_t input_buffer_size = INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH;
  float *input_buffer = (float *)ps_malloc(sizeof(float) * input_buffer_size);
  if (!input_buffer) {
    Serial.println(F("Input buffer allocation failed"));
    free(inference_memory);
    return;
  }

  // Load test input image
  for (uint32_t c = 0; c < INPUT_CHANNELS; c++) {
    for (uint32_t h = 0; h < INPUT_HEIGHT; h++) {
      for (uint32_t w = 0; w < INPUT_WIDTH; w++) {
        input_buffer[c * INPUT_HEIGHT * INPUT_WIDTH + h * INPUT_WIDTH + w] =
            pgm_read_float(&test_input_data[0][c][h][w]);
      }
    }
  }
  aitensor_t input_tensor = AITENSOR_4D_F32(input_shape, input_buffer);

  // Run model
  aitensor_t *output_tensor_ptr = aialgo_forward_model(&model, &input_tensor);
  if (!output_tensor_ptr) {
    Serial.println(F("Forward pass failed"));
    free(input_buffer);
    free(inference_memory);
    return;
  }

  // Print shape and values
  float *output_data_ptr = (float *)output_tensor_ptr->data;
  uint32_t out_size = 1;
  Serial.print("Dense2 output shape: [");
  for (uint8_t d = 0; d < output_tensor_ptr->dim; d++) {
    Serial.printf("%u", output_tensor_ptr->shape[d]);
    out_size *= output_tensor_ptr->shape[d];
    if (d < output_tensor_ptr->dim - 1) Serial.print(", ");
  }
  Serial.println("]");

  Serial.print("Dense2 output: [");
  for (uint32_t i = 0; i < out_size; i++) {
    Serial.printf("%.6f", output_data_ptr[i]);
    if (i < out_size - 1) Serial.print(",");
  }
  Serial.println("]");

  // Free memory
  free(input_buffer);
  free(inference_memory);
  Serial.println(F("Input and inference memory freed"));
}

// Setup
void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!psramInit()) {
    Serial.println(F("PSRAM init failed"));
    while (1);
  }
  Serial.println(F("PSRAM initialized"));

  init_model();
  Serial.println(F("Type >t< to test"));
}

// Loop
void loop() {
  if (Serial.available() > 0) {
    String str = Serial.readString();
    if (str.indexOf("t") > -1) {
      test();
    } else {
      Serial.println(F("Unknown command"));
    }
  }
}
