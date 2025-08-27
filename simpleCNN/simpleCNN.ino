#include <aifes.h>
#include <esp_heap_caps.h>
#include "data.h"
#include "weights.h"

// Enable debug logs for tensor outputs
#define DEBUG 1

// Model constants
#define INPUT_CHANNELS 1  // Grayscale input
#define INPUT_HEIGHT 4    // Image height
#define INPUT_WIDTH 5     // Image width
#define CONV1_FILTERS 1   // Conv1 output channels
#define KERNEL_SIZE \
  { 2, 3 }  // Convolution kernel size
#define STRIDE \
  { 1, 1 }  // Convolution stride
#define PADDING \
  { 0, 0 }  // Convolution padding
#define DILATION \
  { 1, 1 }  // Convolution dilation
#define TEST_DATASET 4  // Number of test inputs
#define LAYER_COUNT 2   // Layers: input, conv1

// Global model variables
aimodel_t model;                // Neural network struct
ailayer_t *layers[LAYER_COUNT]; // Array of layer pointers
void *parameter_memory;         // PSRAM for weights/biases

// Layer structures
uint16_t input_shape[] = { 1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH };                           // Input: [1,1,4,5]
ailayer_input_f32_t input_layer = AILAYER_INPUT_F32_M(4, input_shape);                               // 4D input tensor
ailayer_conv2d_f32_t conv1_layer = AILAYER_CONV2D_F32_M(CONV1_FILTERS, KERNEL_SIZE, STRIDE, DILATION, PADDING, conv1_weight, conv1_bias); // Conv1: 1->1, 2x3

// Initialize CNN layers
void init_layers() {
  // Input layer
  layers[0] = model.input_layer = ailayer_input_f32_default(&input_layer);
  if (!model.input_layer) {
    Serial.println(F("Input layer initialization failed"));
    while (1);
  }

  // Conv1: 1->1 channel, 2x3 kernel, output [1,1,3,3]
  conv1_layer.channel_axis = 1;  // NCHW format
  layers[1] = model.output_layer = ailayer_conv2d_f32_default(&conv1_layer, model.input_layer);
  if (!model.output_layer) {
    Serial.println(F("Conv1 layer initialization failed"));
    while (1);
  }
}

// Initialize model
void init_model() {
  Serial.println(F("Initializing model..."));

  // Set up layers
  init_layers();

  // Compile model
  aialgo_compile_model(&model);
  if (!model.output_layer) {
    Serial.println(F("Model compilation failed"));
    while (1);
  }

  Serial.println(F("Model initialized"));
}

// Test the model
void test() {
  Serial.println(F("Testing..."));
  uint32_t inference_memory_size = aialgo_sizeof_inference_memory(&model);
  void *inference_memory = ps_malloc(inference_memory_size);
  if (!inference_memory) {
    Serial.println(F("Inference memory allocation failed"));
    while (1);
  }
  aialgo_schedule_inference_memory(&model, inference_memory, inference_memory_size);
  Serial.printf("Inference memory allocated: %u bytes, Free PSRAM: %u bytes\n",
                inference_memory_size, ESP.getFreePsram());

  const uint16_t single_input_shape[] = {1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH};
  const uint16_t output_shape[] = {1, CONV1_FILTERS, INPUT_HEIGHT-1, INPUT_WIDTH-2};
  float input_buffer[INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH];
  float output_data[CONV1_FILTERS * (INPUT_HEIGHT-1) * (INPUT_WIDTH-2)];
  aitensor_t input_tensor = AITENSOR_4D_F32(single_input_shape, input_buffer);
  aitensor_t output_tensor = AITENSOR_4D_F32(output_shape, output_data);

  for (uint32_t i = 0; i < TEST_DATASET; i++) {
    // Debug: Print raw input_data from data.h
    Serial.printf("Raw input_data[%u]: [", i);
    for (uint32_t c = 0; c < INPUT_CHANNELS; c++) {
      for (uint32_t h = 0; h < INPUT_HEIGHT; h++) {
        for (uint32_t w = 0; w < INPUT_WIDTH; w++) {
          Serial.printf("%.1f", input_data[i][c][h][w]);
          if (c * INPUT_HEIGHT * INPUT_WIDTH + h * INPUT_WIDTH + w < INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH - 1) Serial.print(", ");
        }
      }
    }
    Serial.println("]");

    // Load input data from data.h
    for (uint32_t c = 0; c < INPUT_CHANNELS; c++) {
      for (uint32_t h = 0; h < INPUT_HEIGHT; h++) {
        for (uint32_t w = 0; w < INPUT_WIDTH; w++) {
          input_buffer[c * INPUT_HEIGHT * INPUT_WIDTH + h * INPUT_WIDTH + w] = input_data[i][c][h][w];
        }
      }
    }

    // Forward pass
    aitensor_t *output_tensor_ptr = aialgo_forward_model(&model, &input_tensor);
    if (!output_tensor_ptr) {
      Serial.printf("Forward pass failed for input %u\n", i);
      continue;
    }

    // Copy output data
    float *output_data_ptr = (float *)output_tensor_ptr->data;
    for (uint32_t j = 0; j < CONV1_FILTERS * (INPUT_HEIGHT-1) * (INPUT_WIDTH-2); j++) {
      output_data[j] = output_data_ptr[j];
    }

    // Print input and output
    Serial.printf("Input %u shape: [1, %u, %u, %u]\n", i, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH);
    Serial.printf("Input %u: [", i);
    for (uint32_t j = 0; j < INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH; j++) {
      Serial.printf("%.1f", input_buffer[j]);
      if (j < INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH - 1) Serial.print(", ");
    }
    Serial.println("]");
    Serial.printf("Conv1 output shape: [1, %u, %u, %u]\n", CONV1_FILTERS, INPUT_HEIGHT-1, INPUT_WIDTH-2);
    Serial.printf("Conv1 output: [", i);
    for (uint32_t j = 0; j < CONV1_FILTERS * (INPUT_HEIGHT-1) * (INPUT_WIDTH-2); j++) {
      Serial.printf("%.1f", output_data[j]);
      if (j < CONV1_FILTERS * (INPUT_HEIGHT-1) * (INPUT_WIDTH-2) - 1) Serial.print(", ");
    }
    Serial.println("]");
  }

  free(inference_memory);
  Serial.printf("Inference memory freed, Free PSRAM: %u bytes\n", ESP.getFreePsram());
}

// Setup function
void setup() {
  Serial.begin(115200);
  while (!Serial);

  // Initialize PSRAM
  if (!psramInit()) {
    Serial.println(F("PSRAM initialization failed"));
    while (1);
  }
  Serial.println(F("PSRAM initialized"));

  // Initialize model
  init_model();
  Serial.println(F("Type >t< to test"));
}

// Main loop
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