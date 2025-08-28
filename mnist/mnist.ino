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
#define OUTPUT_SIZE 10        // For softmax
#define LAYER_COUNT 11        // Input → conv1 → relu1 → pool1 → conv2 → relu2 → pool2 → flatten → dense1 → relu3 → softmax
#define TEST_DATASET 20

// Global model variables
aimodel_t model;
ailayer_t *layers[LAYER_COUNT];
void *parameter_memory;

// =======================
// Dataset abstraction
// =======================

typedef struct {
  uint32_t size;       // number of samples in dataset
  uint32_t index;      // current position
} Dataset;

// Initialize dataset
Dataset dataset_init(uint32_t dataset_size) {
  Dataset ds;
  ds.size = dataset_size;
  ds.index = 0;
  return ds;
}

// Layer structures
uint16_t input_shape[] = {1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH};
ailayer_input_f32_t input_layer = AILAYER_INPUT_F32_M(4, input_shape);

// Convolution, ReLU, Pool layers
ailayer_conv2d_f32_t conv1_layer = AILAYER_CONV2D_F32_M(
    CONV1_FILTERS, KERNEL_SIZE, STRIDE, DILATION, PADDING, conv1_weights, conv1_bias);
ailayer_relu_f32_t relu1_layer = AILAYER_RELU_F32_M();
ailayer_maxpool2d_f32_t pool1_layer = AILAYER_MAXPOOL2D_F32_M(POOL_SIZE, POOL_STRIDE, POOL_PADDING);

ailayer_conv2d_f32_t conv2_layer = AILAYER_CONV2D_F32_M(
    CONV2_FILTERS, KERNEL_SIZE, STRIDE, DILATION, PADDING, conv2_weights, conv2_bias);
ailayer_relu_f32_t relu2_layer = AILAYER_RELU_F32_M();
ailayer_maxpool2d_f32_t pool2_layer = AILAYER_MAXPOOL2D_F32_M(POOL_SIZE, POOL_STRIDE, POOL_PADDING);

// Flatten and Dense layers
ailayer_flatten_f32_t flatten_layer = AILAYER_FLATTEN_F32_M();
ailayer_dense_f32_t dense1_layer = AILAYER_DENSE_F32_M(DENSE1_SIZE, fc1_weights, fc1_bias);
ailayer_relu_f32_t relu3_layer = AILAYER_RELU_F32_M();

// Softmax final layer
ailayer_dense_f32_t dense2_layer = AILAYER_DENSE_F32_M(OUTPUT_SIZE, fc2_weights, fc2_bias);
ailayer_softmax_f32_t softmax_layer = AILAYER_SOFTMAX_F32_M();

// =======================
// Data preparation helpers
// =======================

// Prepare a single input image from PROGMEM into float buffer
// - Normalizes pixel values to [0, 1]
// - Converts from uint8_t to float
void prepare_input(uint32_t img_idx, float *input_buffer) {
  for (uint32_t c = 0; c < INPUT_CHANNELS; c++) {
    for (uint32_t h = 0; h < INPUT_HEIGHT; h++) {
      for (uint32_t w = 0; w < INPUT_WIDTH; w++) {
        uint8_t pixel_val = pgm_read_byte(&test_input_data[img_idx][c][h][w]);
        input_buffer[c * INPUT_HEIGHT * INPUT_WIDTH + h * INPUT_WIDTH + w] =
            (float)pixel_val / 255.0f;  // normalize
      }
    }
  }
}

// Retrieve the target class (0–9) from PROGMEM
uint8_t get_target_class(uint32_t img_idx) {
  return pgm_read_byte(&test_target_data[img_idx]);
}

// Get next batch of data
// - Fills input buffer [batch_size, channels*height*width]
// - Fills target buffer [batch_size]
void dataset_next_batch(Dataset *ds, uint32_t batch_size,
                        float *input_batch, uint8_t *target_batch) {
  for (uint32_t b = 0; b < batch_size; b++) {
    uint32_t img_idx = ds->index;

    // Fill input
    prepare_input(img_idx, &input_batch[b * INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH]);

    // Fill target
    target_batch[b] = get_target_class(img_idx);

    // Advance index (wrap around dataset)
    ds->index = (ds->index + 1) % ds->size;
  }
}

// =======================
// Model initialization
// =======================

// Initialize layers
void init_layers() {
  // Input
  layers[0] = model.input_layer = ailayer_input_f32_default(&input_layer);
  if (!model.input_layer) {
    Serial.println(F("Input layer init failed"));
    while (1);
  }

  // Conv1
  conv1_layer.channel_axis = 1;
  layers[1] = ailayer_conv2d_f32_default(&conv1_layer, model.input_layer);

  // ReLU1
  layers[2] = ailayer_relu_f32_default(&relu1_layer, layers[1]);

  // Pool1
  pool1_layer.channel_axis = 1;
  layers[3] = ailayer_maxpool2d_f32_default(&pool1_layer, layers[2]);

  // Conv2
  conv2_layer.channel_axis = 1;
  layers[4] = ailayer_conv2d_f32_default(&conv2_layer, layers[3]);

  // ReLU2
  layers[5] = ailayer_relu_f32_default(&relu2_layer, layers[4]);

  // Pool2
  pool2_layer.channel_axis = 1;
  layers[6] = ailayer_maxpool2d_f32_default(&pool2_layer, layers[5]);

  // Flatten
  layers[7] = ailayer_flatten_f32_default(&flatten_layer, layers[6]);

  // Dense1
  layers[8] = ailayer_dense_f32_default(&dense1_layer, layers[7]);

  // ReLU3
  layers[9] = ailayer_relu_f32_default(&relu3_layer, layers[8]);

  // Dense2 + Softmax
  layers[10] = model.output_layer = ailayer_softmax_f32_default(&softmax_layer,
                          ailayer_dense_f32_default(&dense2_layer, layers[9]));

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

// =======================
// Inference / Test
// =======================

void test() {
  Serial.printf(F("Testing %d images...\n"), TEST_DATASET);

  uint32_t inference_memory_size = aialgo_sizeof_inference_memory(&model);
  void *inference_memory = ps_malloc(inference_memory_size);
  if (!inference_memory) {
    Serial.println(F("Inference memory allocation failed"));
    while (1);
  }
  aialgo_schedule_inference_memory(&model, inference_memory, inference_memory_size);
  Serial.printf("Inference memory allocated: %u bytes\n", inference_memory_size);

  // Allocate input + label buffers once
  float *input_buffer = (float *)ps_malloc(INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH * sizeof(float));
  uint8_t *target_buffer = (uint8_t *)ps_malloc(sizeof(uint8_t));
  if (!input_buffer || !target_buffer) {
    Serial.println(F("Buffer allocation failed"));
    free(inference_memory);
    while (1);
  }

  const uint16_t single_input_shape[] = {1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH};
  aitensor_t input_tensor = AITENSOR_4D_F32(single_input_shape, input_buffer);

  Dataset test_ds = dataset_init(TEST_DATASET);
  uint32_t correct_count = 0;

  for (uint32_t img_idx = 0; img_idx < TEST_DATASET; img_idx++) {
    // Get next sample (batch_size = 1)
    dataset_next_batch(&test_ds, 1, input_buffer, target_buffer);

    // Run model forward
    aitensor_t *output_tensor_ptr = aialgo_forward_model(&model, &input_tensor);
    if (!output_tensor_ptr) {
      Serial.printf(F("Forward pass failed for image %d\n"), img_idx);
      continue;
    }

    float *output_data_ptr = (float *)output_tensor_ptr->data;
    uint32_t out_size = 1;
    for (uint8_t d = 0; d < output_tensor_ptr->dim; d++) out_size *= output_tensor_ptr->shape[d];

    // Determine predicted class (argmax)
    uint32_t pred_class = 0;
    float max_val = output_data_ptr[0];
    for (uint32_t i = 1; i < out_size; i++) {
      if (output_data_ptr[i] > max_val) {
        max_val = output_data_ptr[i];
        pred_class = i;
      }
    }

    // Compare with actual label
    uint32_t target_class = target_buffer[0];
    bool matches = (pred_class == target_class);
    if (matches) correct_count++;

    Serial.printf("Image %d: Predicted: %u, Actual: %u, Correct: %s\n",
                  img_idx, pred_class, target_class, matches ? "YES" : "NO");
  }

  float accuracy = 100.0f * correct_count / TEST_DATASET;
  Serial.printf("Accuracy: %u/%u (%.2f%%)\n", correct_count, TEST_DATASET, accuracy);

  free(input_buffer);
  free(target_buffer);
  free(inference_memory);
  Serial.println(F("Memory freed after inference"));
}

// =======================
// Setup & Loop
// =======================

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
