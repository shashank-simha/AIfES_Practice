#include <aifes.h>
#include <esp_heap_caps.h>
#include "mnist_data.h"

SET_LOOP_TASK_STACK_SIZE(128 * 1024);  // 128KB

// MNIST and CNN constants
#define INPUT_CHANNELS 1  // Grayscale images
#define INPUT_HEIGHT 28   // Image height
#define INPUT_WIDTH 28    // Image width
#define OUTPUT_SIZE 10    // Number of classes (digits 0-9)
#define CONV1_FILTERS 4   // Conv1 output channels
#define CONV2_FILTERS 8   // Conv2 output channels
#define KERNEL_SIZE \
  { 3, 3 }  // Convolution kernel size
#define STRIDE \
  { 1, 1 }  // Convolution stride
#define PADDING \
  { 1, 1 }  // Convolution padding
#define DILATION \
  { 1, 1 }  // Convolution dilation
#define POOL_SIZE \
  { 2, 2 }  // Max pooling size
#define POOL_STRIDE \
  { 2, 2 }  // Max pooling stride
#define POOL_PADDING \
  { 0, 0 }              // Max pooling padding
#define DENSE1_SIZE 32  // Dense layer neurons
#define LAYER_COUNT 12  // Layers: input, conv1, relu1, pool1, conv2, relu2, pool2, flatten, dense1, relu3, dense2, softmax

// Training constants
#define TRAIN_DATASET 10  // Number of training samples
#define TEST_DATASET 5    // Number of test samples
#define BATCH_SIZE 1
#define EPOCHS 10
#define PRINT_INTERVAL 1
#define LEARNING_RATE 0.01f

// Normalize uint8 image (0-255) to float32 (mean=0.1307, std=0.3081)
void normalize_image(const uint8_t *raw, float *output, uint32_t size) {
  for (uint32_t i = 0; i < size; i++) {
    output[i] = ((float)raw[i] / 255.0f - 0.1307f) / 0.3081f;
  }
}

// Convert uint8 label (0-9) to one-hot float32
void one_hot_label(uint8_t label, float *output, uint32_t num_classes) {
  for (uint32_t i = 0; i < num_classes; i++) {
    output[i] = (i == label) ? 1.0f : 0.0f;
  }
}


// Global model variables
aimodel_t model;                 // Neural network struct
ailayer_t *layers[LAYER_COUNT];  // Array of layer pointers
void *parameter_memory;          // PSRAM for weights/biases

// Layer structures
uint16_t input_shape[] = { 1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH };                                       // Input: [1,1,28,28]
ailayer_input_f32_t input_layer = AILAYER_INPUT_F32_A(4, input_shape);                                           // 4D input tensor
ailayer_conv2d_f32_t conv1_layer = AILAYER_CONV2D_F32_A(CONV1_FILTERS, KERNEL_SIZE, STRIDE, DILATION, PADDING);  // Conv1: 1->4, 3x3, output [1,4,28,28]
ailayer_relu_f32_t relu1_layer = AILAYER_RELU_F32_A();                                                           // ReLU activation
ailayer_maxpool2d_f32_t pool1_layer = AILAYER_MAXPOOL2D_F32_A(POOL_SIZE, POOL_STRIDE, POOL_PADDING);             // MaxPool: 2x2, output [1,4,14,14]
ailayer_conv2d_f32_t conv2_layer = AILAYER_CONV2D_F32_A(CONV2_FILTERS, KERNEL_SIZE, STRIDE, DILATION, PADDING);  // Conv2: 4->8, 3x3, output [1,8,14,14]
ailayer_relu_f32_t relu2_layer = AILAYER_RELU_F32_A();                                                           // ReLU activation
ailayer_maxpool2d_f32_t pool2_layer = AILAYER_MAXPOOL2D_F32_A(POOL_SIZE, POOL_STRIDE, POOL_PADDING);             // MaxPool: 2x2, output [1,8,7,7]
ailayer_flatten_f32_t flatten_layer = AILAYER_FLATTEN_F32_A();                                                   // Flatten to [1,392]
ailayer_dense_f32_t dense1_layer = AILAYER_DENSE_F32_A(DENSE1_SIZE);                                             // Dense: 392->32
ailayer_relu_f32_t relu3_layer = AILAYER_RELU_F32_A();                                                           // ReLU activation
ailayer_dense_f32_t dense2_layer = AILAYER_DENSE_F32_A(OUTPUT_SIZE);                                             // Dense: 32->10
ailayer_softmax_f32_t softmax_layer = AILAYER_SOFTMAX_F32_A();                                                   // Softmax for 10-class output

// Initialize MNIST CNN model
void init_model() {
  Serial.println(F("Initializing model..."));

  // Connect layers and populate layers array
  ailayer_t *x;
  layers[0] = model.input_layer = ailayer_input_f32_default(&input_layer);  // Input: [1,1,28,28]
  if (!model.input_layer) {
    Serial.println(F("Input layer initialization failed"));
    while (1)
      ;
  }

  conv1_layer.channel_axis = 1;                                                 // NCHW
  layers[1] = x = ailayer_conv2d_f32_default(&conv1_layer, model.input_layer);  // Conv1: 1->4, 3x3
  if (!x) {
    Serial.println(F("Conv1 layer initialization failed"));
    while (1)
      ;
  }

  layers[2] = x = ailayer_relu_f32_default(&relu1_layer, x);  // ReLU
  if (!x) {
    Serial.println(F("ReLU1 layer initialization failed"));
    while (1)
      ;
  }

  pool1_layer.channel_axis = 1;                                    // NCHW
  layers[3] = x = ailayer_maxpool2d_f32_default(&pool1_layer, x);  // MaxPool: 2x2
  if (!x) {
    Serial.println(F("MaxPool1 layer initialization failed"));
    while (1)
      ;
  }

  conv2_layer.channel_axis = 1;                                 // NCHW
  layers[4] = x = ailayer_conv2d_f32_default(&conv2_layer, x);  // Conv2: 4->8, 3x3
  if (!x) {
    Serial.println(F("Conv2 layer initialization failed"));
    while (1)
      ;
  }

  layers[5] = x = ailayer_relu_f32_default(&relu2_layer, x);  // ReLU
  if (!x) {
    Serial.println(F("ReLU2 layer initialization failed"));
    while (1)
      ;
  }

  pool2_layer.channel_axis = 1;                                    // NCHW
  layers[6] = x = ailayer_maxpool2d_f32_default(&pool2_layer, x);  // MaxPool: 2x2
  if (!x) {
    Serial.println(F("MaxPool2 layer initialization failed"));
    while (1)
      ;
  }

  layers[7] = x = ailayer_flatten_f32_default(&flatten_layer, x);  // Flatten
  if (!x) {
    Serial.println(F("Flatten layer initialization failed"));
    while (1)
      ;
  }

  layers[8] = x = ailayer_dense_f32_default(&dense1_layer, x);  // Dense: 392->32
  if (!x) {
    Serial.println(F("Dense1 layer initialization failed"));
    while (1)
      ;
  }

  layers[9] = x = ailayer_relu_f32_default(&relu3_layer, x);  // ReLU
  if (!x) {
    Serial.println(F("ReLU3 layer initialization failed"));
    while (1)
      ;
  }

  layers[10] = x = ailayer_dense_f32_default(&dense2_layer, x);  // Dense: 32->10
  if (!x) {
    Serial.println(F("Dense2 layer initialization failed"));
    while (1)
      ;
  }

  layers[11] = model.output_layer = ailayer_softmax_f32_default(&softmax_layer, x);  // Softmax
  if (!model.output_layer) {
    Serial.println(F("Softmax layer initialization failed"));
    while (1)
      ;
  }

  // Compile model to verify connections
  aialgo_compile_model(&model);
  if (!model.output_layer) {
    Serial.println(F("Model compilation failed"));
    while (1)
      ;
  }

  // Allocate parameter memory in PSRAM
  uint32_t parameter_memory_size = aialgo_sizeof_parameter_memory(&model);
  parameter_memory = ps_malloc(parameter_memory_size);
  if (!parameter_memory) {
    Serial.println(F("Model memory allocation failed"));
    while (1)
      ;
  }
  aialgo_distribute_parameter_memory(&model, parameter_memory, parameter_memory_size);
  Serial.printf("Model memory allocated: %u bytes, Free PSRAM: %u bytes\n",
                parameter_memory_size, ESP.getFreePsram());

  // Print model structure
  aiprint("\n-------------- Model structure ---------------\n");
  aialgo_print_model_structure(&model);
  aiprint("----------------------------------------------\n\n");

  Serial.println(F("Model initialized"));
}

void train() {
  Serial.println("Training...");
  Serial.printf("Free SRAM before: %u bytes\n", ESP.getFreeHeap());

  // Configure loss
  ailoss_crossentropy_f32_t crossentropy_loss;
  model.loss = ailoss_crossentropy_f32_default(&crossentropy_loss, model.output_layer);
  if (!model.loss) {
    Serial.println(F("Loss initialization failed"));
    while (1)
      ;
  }
  aiprint("\nLoss specs:\n");
  aialgo_print_loss_specs(model.loss);
  aiprint("\n");

  // Configure SGD optimizer
  aiopti_sgd_f32_t sgd_opti = { .learning_rate = LEARNING_RATE };
  aiopti_t *optimizer = aiopti_sgd_f32_default(&sgd_opti);
  if (!optimizer) {
    Serial.println(F("Optimizer initialization failed"));
    while (1)
      ;
  }
  aiprint("Optimizer specs:\n");
  aialgo_print_optimizer_specs(optimizer);
  aiprint("\n");

  // Initialize parameters
  aialgo_initialize_parameters_model(&model);
  Serial.println(F("Parameters initialized"));

  // Allocate training memory in PSRAM
  uint32_t training_memory_size = aialgo_sizeof_training_memory(&model, optimizer);
  void *training_memory = ps_malloc(training_memory_size);
  if (!training_memory) {
    Serial.println(F("Training memory allocation failed"));
    while (1)
      ;
  }
  aialgo_schedule_training_memory(&model, optimizer, training_memory, training_memory_size);
  aialgo_init_model_for_training(&model, optimizer);
  Serial.printf("Training memory allocated: %u bytes, Free PSRAM: %u bytes\n",
                training_memory_size, ESP.getFreePsram());

  // Allocate SRAM buffers for tensor data
  float input_buffer[TRAIN_DATASET][1][28][28];  // ~3,136 bytes
  float target_buffer[TRAIN_DATASET][10];        // ~40 bytes

  // Copy pre-normalized data from PROGMEM to SRAM with debugging
  Serial.printf("Copying PROGMEM data to SRAM, Free SRAM: %u bytes\n", ESP.getFreeHeap());
  for (uint32_t i = 0; i < TRAIN_DATASET; i++) {
    Serial.printf("Copying image %u\n", i);
    for (uint32_t c = 0; c < 1; c++) {
      for (uint32_t h = 0; h < 28; h++) {
        for (uint32_t w = 0; w < 28; w++) {
          input_buffer[i][c][h][w] = pgm_read_float(&train_input_data[i][c][h][w]);
          if (w == 0 && h == 0 && c == 0) {
            Serial.printf("Image %u, first pixel: %.4f\n", i, input_buffer[i][c][h][w]);  // ~0.4245
          }
        }
      }
    }
    for (uint32_t j = 0; j < 10; j++) {
      target_buffer[i][j] = pgm_read_float(&train_target_data[i][j]);
      if (j == 0) {
        Serial.printf("Image %u, first target value: %.1f\n", i, target_buffer[i][j]);
      }
    }
  }
  Serial.printf("Data copy completed, Free SRAM: %u bytes\n", ESP.getFreeHeap());

  // Verify first pixel
  Serial.printf("First pixel (normalized): %.4f\n", input_buffer[0][0][0][0]);  // ~0.4245

  // Create tensors
  const uint16_t input_shape[] = { TRAIN_DATASET, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH };
  const uint16_t target_shape[] = { TRAIN_DATASET, OUTPUT_SIZE };
  aitensor_t input_tensor = AITENSOR_4D_F32(input_shape, (float *)input_buffer);
  aitensor_t target_tensor = AITENSOR_2D_F32(target_shape, (float *)target_buffer);

  // Debug tensor shapes and data
  Serial.printf("Input tensor shape: [%u,%u,%u,%u]\n",
                input_tensor.shape[0], input_tensor.shape[1],
                input_tensor.shape[2], input_tensor.shape[3]);
  Serial.printf("Target tensor shape: [%u,%u]\n",
                target_tensor.shape[0], target_tensor.shape[1]);
  Serial.printf("Free SRAM after tensors: %u bytes\n", ESP.getFreeHeap());

  // Test forward pass
  Serial.println(F("Testing forward pass"));
  aitensor_t *output_tensor = aialgo_forward_model(&model, &input_tensor);
  if (!output_tensor) {
    Serial.println(F("Forward pass failed"));
    while (1)
      ;
  }
  Serial.println(F("Forward pass completed"));

  // Training loop
  float loss;
  aiprint("\nStart training\n");
  for (int i = 0; i < EPOCHS; i++) {
    Serial.println(F("Before train_model"));
    aialgo_train_model(&model, &input_tensor, &target_tensor, optimizer, BATCH_SIZE);
    Serial.println(F("After train_model"));
    if (i % PRINT_INTERVAL == 0) {
      aialgo_calc_loss_model_f32(&model, &input_tensor, &target_tensor, &loss);
      aiprint("Epoch ");
      aiprint_int("%5d", i);
      aiprint(": test loss: ");
      aiprint_float("%f", loss);
      aiprint("\n");
    }
  }
  aiprint("Finished training\n\n");

  // Free training memory
  free(training_memory);
  Serial.printf("Training memory freed, Free PSRAM: %u bytes\n", ESP.getFreePsram());
}

void infer() {
  Serial.println("Inferring...");
}

void test() {
  Serial.println(F("Testing..."));
  Serial.printf("Free SRAM before: %u bytes\n", ESP.getFreeHeap());

  // Allocate SRAM buffers for test data
  float input_buffer[TEST_DATASET][1][28][28];  // ~3,136 bytes
  float target_buffer[TEST_DATASET][10];        // ~40 bytes

  // Copy pre-normalized test data from PROGMEM to SRAM
  Serial.printf("Copying PROGMEM test data to SRAM, Free SRAM: %u bytes\n", ESP.getFreeHeap());
  for (uint32_t i = 0; i < TEST_DATASET; i++) {
    Serial.printf("Copying test image %u\n", i);
    for (uint32_t h = 0; h < 28; h++) {
      for (uint32_t w = 0; w < 28; w++) {
        input_buffer[i][0][h][w] = pgm_read_float(&test_input_data[i][0][h][w]);
        if (h == 0 && w == 0) {
          Serial.printf("Test image %u, first pixel: %.4f\n", i, input_buffer[i][0][h][w]);  // ~-0.4242
        }
      }
      if (h % 7 == 0) {
        Serial.printf("Test image %u, row %u, SRAM: %u bytes\n", i, h, ESP.getFreeHeap());
      }
    }
    for (uint32_t j = 0; j < 10; j++) {
      target_buffer[i][j] = pgm_read_float(&test_target_data[i][j]);
      if (j == 0) {
        Serial.printf("Test image %u, first target value: %.1f\n", i, target_buffer[i][j]);
      }
    }
  }
  Serial.printf("Test data copy completed, Free SRAM: %u bytes\n", ESP.getFreeHeap());

  // Verify first pixel
  Serial.printf("First pixel (normalized): %.4f\n", input_buffer[0][0][0][0]);  // ~-0.4242

  // Create tensors
  const uint16_t input_shape[] = { TEST_DATASET, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH };
  const uint16_t target_shape[] = { TEST_DATASET, OUTPUT_SIZE };
  aitensor_t input_tensor = AITENSOR_4D_F32(input_shape, (float *)input_buffer);
  aitensor_t target_tensor = AITENSOR_2D_F32(target_shape, (float *)target_buffer);

  // Debug tensor shapes and data
  Serial.printf("Input tensor shape: [%u,%u,%u,%u]\n",
                input_tensor.shape[0], input_tensor.shape[1],
                input_tensor.shape[2], input_tensor.shape[3]);
  Serial.printf("Target tensor shape: [%u,%u]\n",
                target_tensor.shape[0], target_tensor.shape[1]);
  Serial.printf("Free SRAM after tensors: %u bytes\n", ESP.getFreeHeap());

  // Test inference
  Serial.println(F("Testing forward pass"));
  aitensor_t *output_tensor = aialgo_forward_model(&model, &input_tensor);
  if (!output_tensor) {
    Serial.println(F("Forward pass failed"));
    while (1)
      ;
  }
  Serial.println(F("Forward pass completed"));

  // Calculate accuracy
  uint32_t correct = 0;
  for (uint32_t i = 0; i < TEST_DATASET; i++) {
    uint32_t predicted = 0, true_label = 0;
    float max_prob = ((float *)output_tensor->data)[i * 10];
    for (uint32_t j = 1; j < 10; j++) {
      float prob = ((float *)output_tensor->data)[i * 10 + j];
      if (prob > max_prob) {
        max_prob = prob;
        predicted = j;
      }
      if (target_buffer[i][j] == 1.0f) {
        true_label = j;
      }
    }
    if (predicted == true_label) {
      correct++;
    }
    Serial.printf("Test image %u: Predicted=%u, True=%u, Correct=%s\n",
                  i, predicted, true_label, predicted == true_label ? "Yes" : "No");
  }

  // Print accuracy
  float accuracy = (float)correct / TEST_DATASET * 100.0f;
  Serial.printf("Testing completed, Accuracy: %.2f%% (%u/%u)\n",
                accuracy, correct, TEST_DATASET);
  Serial.printf("Free SRAM after: %u bytes\n", ESP.getFreeHeap());
}

// Setup function with PSRAM check
void setup() {
  Serial.begin(115200);
  while (!Serial)
    ;

  Serial.println(F("\n##################################"));
  Serial.println(F("ESP32 Information:"));
  Serial.printf("Internal Total Heap %d, Internal Used Heap %d, Internal Free Heap %d\n", ESP.getHeapSize(), ESP.getHeapSize() - ESP.getFreeHeap(), ESP.getFreeHeap());
  Serial.printf("Sketch Size %d, Free Sketch Space %d\n", ESP.getSketchSize(), ESP.getFreeSketchSpace());
  Serial.printf("SPIRam Total heap %d, SPIRam Free Heap %d\n", ESP.getPsramSize(), ESP.getFreePsram());
  Serial.printf("Chip Model %s, ChipRevision %d, Cpu Freq %d, SDK Version %s\n", ESP.getChipModel(), ESP.getChipRevision(), ESP.getCpuFreqMHz(), ESP.getSdkVersion());
  Serial.printf("Flash Size %d, Flash Speed %d\n", ESP.getFlashChipSize(), ESP.getFlashChipSpeed());
  Serial.println(F("##################################\n\n"));

  // Check PSRAM initialization
  if (!psramInit()) {
    Serial.println(F("PSRAM initialization failed"));
    while (1)
      ;
  }
  Serial.println(F("PSRAM initialized"));

  srand(analogRead(A5));
  init_model();
  Serial.println(F("Type >train< or >infer<"));
}

// Main loop for command input
void loop() {
  if (Serial.available() > 0) {
    String str = Serial.readString();
    // if (str.indexOf("train") > -1) {
    if (str.indexOf("t") > -1) {
      train();
      test();
    } else if (str.indexOf("infer") > -1) {
      infer();
    } else {
      Serial.println(F("Unknown command"));
    }
  }
}