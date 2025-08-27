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
#define KERNEL_SIZE {3, 3}
#define STRIDE {1, 1}
#define PADDING {1, 1}
#define DILATION {1, 1}
#define LAYER_COUNT 2  // Input, conv1
#define TEST_DATASET 1  // One image

// Global model variables
aimodel_t model;
ailayer_t *layers[LAYER_COUNT];
void *parameter_memory;

// Layer structures
uint16_t input_shape[] = {1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH};
ailayer_input_f32_t input_layer = AILAYER_INPUT_F32_M(4, input_shape);
ailayer_conv2d_f32_t conv1_layer = AILAYER_CONV2D_F32_M(CONV1_FILTERS, KERNEL_SIZE, STRIDE, DILATION, PADDING, conv1_weights, conv1_bias);

// Initialize layers
void init_layers() {
  layers[0] = model.input_layer = ailayer_input_f32_default(&input_layer);
  if (!model.input_layer) {
    Serial.println(F("Input layer init failed"));
    while (1);
  }
  conv1_layer.channel_axis = 1;  // NCHW
  layers[1] = model.output_layer = ailayer_conv2d_f32_default(&conv1_layer, model.input_layer);
  if (!model.output_layer) {
    Serial.println(F("Conv1 layer init failed"));
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
  Serial.println(F("Testing conv1..."));
  uint32_t inference_memory_size = aialgo_sizeof_inference_memory(&model);
  void *inference_memory = ps_malloc(inference_memory_size);
  if (!inference_memory) {
    Serial.println(F("Inference memory allocation failed"));
    while (1);
  }
  aialgo_schedule_inference_memory(&model, inference_memory, inference_memory_size);
  Serial.printf("Inference memory allocated: %u bytes\n", inference_memory_size);

  const uint16_t single_input_shape[] = {1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH};
  const uint16_t output_shape[] = {1, CONV1_FILTERS, INPUT_HEIGHT, INPUT_WIDTH};
  float input_buffer[INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH];
  float output_data[CONV1_FILTERS * INPUT_HEIGHT * INPUT_WIDTH];
  aitensor_t input_tensor = AITENSOR_4D_F32(single_input_shape, input_buffer);
  aitensor_t output_tensor = AITENSOR_4D_F32(output_shape, output_data);

  for (uint32_t c = 0; c < INPUT_CHANNELS; c++) {
    for (uint32_t h = 0; h < INPUT_HEIGHT; h++) {
      for (uint32_t w = 0; w < INPUT_WIDTH; w++) {
        input_buffer[c * INPUT_HEIGHT * INPUT_WIDTH + h * INPUT_WIDTH + w] = pgm_read_float(&test_input_data[0][c][h][w]);
      }
    }
  }

  aitensor_t *output_tensor_ptr = aialgo_forward_model(&model, &input_tensor);
  if (!output_tensor_ptr) {
    Serial.println(F("Forward pass failed"));
    return;
  }

  float *output_data_ptr = (float *)output_tensor_ptr->data;
  Serial.print("Conv1 output: [");
  for (uint32_t j = 0; j < CONV1_FILTERS * INPUT_HEIGHT * INPUT_WIDTH; j++) {
    Serial.printf("%.6f", output_data_ptr[j]);
    if (j < CONV1_FILTERS * INPUT_HEIGHT * INPUT_WIDTH - 1) Serial.print(", ");
  }
  Serial.println("]");

  free(inference_memory);
  Serial.printf("Inference memory freed\n");
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