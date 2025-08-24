#include <aifes.h>
#include <esp_heap_caps.h>
#include "mnist_data.h"
#include "mnist_weights.h"

// Enable debug logs (set to 0 to disable verbose logs like tensor shapes, per-image copy)
#define DEBUG 1

// Set stack size for loopTask to handle large buffers and AIfES internals
SET_LOOP_TASK_STACK_SIZE(270 * 1024);  // 256KB

// Model and training constants
#define INPUT_CHANNELS 1  // Grayscale images
#define INPUT_HEIGHT 28   // Image height
#define INPUT_WIDTH 28    // Image width
#define OUTPUT_SIZE 10    // Number of classes (digits 0-9)
#define CONV1_FILTERS 8   // Conv1 output channels
#define CONV2_FILTERS 16  // Conv2 output channels
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
  { 0, 0 }                    // Max pooling padding
#define DENSE1_SIZE 64        // Dense layer neurons
#define LAYER_COUNT 12        // Layers: input, conv1, relu1, pool1, conv2, relu2, pool2, flatten, dense1, relu3, dense2, softmax
#define TRAIN_DATASET 200     // Number of training samples
#define TEST_DATASET 20       // Number of test samples
#define BATCH_SIZE 4          // Batch size for training
#define EPOCHS 1              // Number of training epochs
#define PRINT_INTERVAL 1      // Print loss every epoch
#define LEARNING_RATE 0.001f  // SGD learning rate

// Global model variables
aimodel_t model;                 // Neural network struct
ailayer_t *layers[LAYER_COUNT];  // Array of layer pointers
void *parameter_memory;          // PSRAM for weights/biases

// Layer structures
uint16_t input_shape[] = { 1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH };                                                                  // Input: [1,1,28,28]
ailayer_input_f32_t input_layer = AILAYER_INPUT_F32_M(4, input_shape);                                                                      // 4D input tensor
ailayer_conv2d_f32_t conv1_layer = AILAYER_CONV2D_F32_M(CONV1_FILTERS, KERNEL_SIZE, STRIDE, DILATION, PADDING, conv1_weights, conv1_bias);  // Conv1: 1->4, 3x3
ailayer_relu_f32_t relu1_layer = AILAYER_RELU_F32_M();                                                                                      // ReLU activation
ailayer_maxpool2d_f32_t pool1_layer = AILAYER_MAXPOOL2D_F32_M(POOL_SIZE, POOL_STRIDE, POOL_PADDING);                                        // MaxPool: 2x2
ailayer_conv2d_f32_t conv2_layer = AILAYER_CONV2D_F32_M(CONV2_FILTERS, KERNEL_SIZE, STRIDE, DILATION, PADDING, conv2_weights, conv2_bias);  // Conv2: 4->8, 3x3
ailayer_relu_f32_t relu2_layer = AILAYER_RELU_F32_M();                                                                                      // ReLU activation
ailayer_maxpool2d_f32_t pool2_layer = AILAYER_MAXPOOL2D_F32_M(POOL_SIZE, POOL_STRIDE, POOL_PADDING);                                        // MaxPool: 2x2
ailayer_flatten_f32_t flatten_layer = AILAYER_FLATTEN_F32_M();                                                                              // Flatten to vector
ailayer_dense_f32_t dense1_layer = AILAYER_DENSE_F32_M(DENSE1_SIZE, fc1_weights, fc1_bias);                                                 // Dense: 392->32
ailayer_relu_f32_t relu3_layer = AILAYER_RELU_F32_M();                                                                                      // ReLU activation
ailayer_dense_f32_t dense2_layer = AILAYER_DENSE_F32_M(OUTPUT_SIZE, fc2_weights, fc2_bias);                                                 // Dense: 32->10
ailayer_softmax_f32_t softmax_layer = AILAYER_SOFTMAX_F32_M();                                                                              // Softmax for probabilities

// Initialize CNN layers
void init_layers() {
  // Define input shape: [1,1,28,28]
  uint16_t input_shape[] = { 1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH };
  ailayer_t *x;

  // Input layer
  layers[0] = model.input_layer = ailayer_input_f32_default(&input_layer);
  if (!model.input_layer) {
    Serial.println(F("Input layer initialization failed"));
    while (1)
      ;
  }

  // Conv1: 1->4 channels, 3x3 kernel, output [1,4,28,28]
  conv1_layer.channel_axis = 1;  // NCHW format
  layers[1] = x = ailayer_conv2d_f32_default(&conv1_layer, model.input_layer);
  if (!x) {
    Serial.println(F("Conv1 layer initialization failed"));
    while (1)
      ;
  }

  // ReLU activation
  layers[2] = x = ailayer_relu_f32_default(&relu1_layer, x);
  if (!x) {
    Serial.println(F("ReLU1 layer initialization failed"));
    while (1)
      ;
  }

  // MaxPool: 2x2, output [1,4,14,14]
  pool1_layer.channel_axis = 1;
  layers[3] = x = ailayer_maxpool2d_f32_default(&pool1_layer, x);
  if (!x) {
    Serial.println(F("MaxPool1 layer initialization failed"));
    while (1)
      ;
  }

  // Conv2: 4->8 channels, 3x3 kernel, output [1,8,14,14]
  conv2_layer.channel_axis = 1;
  layers[4] = x = ailayer_conv2d_f32_default(&conv2_layer, x);
  if (!x) {
    Serial.println(F("Conv2 layer initialization failed"));
    while (1)
      ;
  }

  // ReLU activation
  layers[5] = x = ailayer_relu_f32_default(&relu2_layer, x);
  if (!x) {
    Serial.println(F("ReLU2 layer initialization failed"));
    while (1)
      ;
  }

  // MaxPool: 2x2, output [1,8,7,7]
  pool2_layer.channel_axis = 1;
  layers[6] = x = ailayer_maxpool2d_f32_default(&pool2_layer, x);
  if (!x) {
    Serial.println(F("MaxPool2 layer initialization failed"));
    while (1)
      ;
  }

  // Flatten to [1,392]
  layers[7] = x = ailayer_flatten_f32_default(&flatten_layer, x);
  if (!x) {
    Serial.println(F("Flatten layer initialization failed"));
    while (1)
      ;
  }

  // Dense: 392->32
  layers[8] = x = ailayer_dense_f32_default(&dense1_layer, x);
  if (!x) {
    Serial.println(F("Dense1 layer initialization failed"));
    while (1)
      ;
  }

  // ReLU activation
  layers[9] = x = ailayer_relu_f32_default(&relu3_layer, x);
  if (!x) {
    Serial.println(F("ReLU3 layer initialization failed"));
    while (1)
      ;
  }

  // Dense: 32->10
  layers[10] = x = ailayer_dense_f32_default(&dense2_layer, x);
  if (!x) {
    Serial.println(F("Dense2 layer initialization failed"));
    while (1)
      ;
  }

  // Softmax for 10-class probabilities
  layers[11] = model.output_layer = ailayer_softmax_f32_default(&softmax_layer, x);
  if (!x) {
    Serial.println(F("Softmax layer initialization failed"));
    while (1)
      ;
  }
}

// Initialize MNIST CNN model
void init_model() {
  // Start model initialization
  Serial.println(F("Initializing model..."));

  // Set up layers
  init_layers();

  // Compile model to verify connections
  aialgo_compile_model(&model);
  if (!model.output_layer) {
    Serial.println(F("Model compilation failed"));
    while (1)
      ;
  }

  // Allocate parameter memory in PSRAM (~100 KB)
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

// Print model structure if DEBUG enabled
#if DEBUG
  aiprint("\n-------------- Model structure ---------------\n");
  aialgo_print_model_structure(&model);
  aiprint("----------------------------------------------\n\n");
#endif

  Serial.println(F("Model initialized"));

#if DEBUG
  Serial.printf("conv1 weights shape: [%u, %u, %u, %u]\n", conv1_layer.weights_shape[0], conv1_layer.weights_shape[1], conv1_layer.weights_shape[2], conv1_layer.weights_shape[3]);
  Serial.printf("conv1 bias shape: %u\n", *(conv1_layer.bias_shape));
  Serial.printf("conv2 weights shape: [%u, %u, %u, %u]\n", conv2_layer.weights_shape[0], conv2_layer.weights_shape[1], conv2_layer.weights_shape[2], conv2_layer.weights_shape[3]);
  Serial.printf("conv2 bias shape: %u\n", *(conv2_layer.bias_shape));
  Serial.printf("dense1 weights shape: [%u, %u]\n", dense1_layer.weights_shape[0], dense1_layer.weights_shape[1]);
  Serial.printf("dense1 bias shape: %u\n", *(dense1_layer.bias_shape));
  Serial.printf("dense2 weights shape: [%u, %u]\n", dense2_layer.weights_shape[0], dense2_layer.weights_shape[1]);
  Serial.printf("dense2 bias shape: %u\n", *(dense2_layer.bias_shape));
#endif
}

// Train the model
void train() {
  // Start training
  Serial.println(F("Training..."));
#if DEBUG
  Serial.printf("Free SRAM before: %u bytes\n", ESP.getFreeHeap());
#endif

  // Configure cross-entropy loss
  ailoss_crossentropy_f32_t crossentropy_loss;
  model.loss = ailoss_crossentropy_f32_default(&crossentropy_loss, model.output_layer);
  if (!model.loss) {
    Serial.println(F("Loss initialization failed"));
    while (1)
      ;
  }
#if DEBUG
  aiprint("\nLoss specs:\n");
  aialgo_print_loss_specs(model.loss);
  aiprint("\n");
#endif

  // Configure SGD optimizer
  aiopti_sgd_f32_t sgd_opti = { .learning_rate = LEARNING_RATE };
  aiopti_t *optimizer = aiopti_sgd_f32_default(&sgd_opti);
  if (!optimizer) {
    Serial.println(F("Optimizer initialization failed"));
    while (1)
      ;
  }
#if DEBUG
  aiprint("Optimizer specs:\n");
  aialgo_print_optimizer_specs(optimizer);
  aiprint("\n");
#endif

  // Initialize model parameters
  aialgo_initialize_parameters_model(&model);
  Serial.println(F("Parameters initialized"));

  // Allocate training memory in PSRAM (~153 KB)
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

#if DEBUG
  // Verify first pixel directly from PROGMEM
  Serial.printf("First train pixel: %.4f\n", pgm_read_float(&train_input_data[0]));  // ~-0.4242
#endif

  // Create tensors directly from PROGMEM
  const uint16_t input_shape[] = { TRAIN_DATASET, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH };
  const uint16_t target_shape[] = { TRAIN_DATASET, OUTPUT_SIZE };
  aitensor_t input_tensor = AITENSOR_4D_F32(input_shape, (float *)train_input_data);
  aitensor_t target_tensor = AITENSOR_2D_F32(target_shape, (float *)train_target_data);

// Log tensor shapes if DEBUG enabled
#if DEBUG
  Serial.printf("Input tensor shape: [%u,%u,%u,%u]\n",
                input_tensor.shape[0], input_tensor.shape[1],
                input_tensor.shape[2], input_tensor.shape[3]);
  Serial.printf("Target tensor shape: [%u,%u]\n",
                target_tensor.shape[0], target_tensor.shape[1]);
  Serial.printf("Free SRAM after tensors: %u bytes\n", ESP.getFreeHeap());
#endif

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
  aiprint("Start training\n");
  for (int i = 0; i < EPOCHS; i++) {
#if DEBUG
    Serial.println(F("Before train_model"));
#endif
    aialgo_train_model(&model, &input_tensor, &target_tensor, optimizer, BATCH_SIZE);
#if DEBUG
    Serial.println(F("After train_model"));
#endif
    if (i % PRINT_INTERVAL == 0) {
      aialgo_calc_loss_model_f32(&model, &input_tensor, &target_tensor, &loss);
      aiprint("Epoch ");
      aiprint_int("%5d", i);
      aiprint(": test loss: ");
      aiprint_float("%f", loss);
      aiprint("\n");
    }
  }
  aiprint("Finished training\n");

  // Free training memory
  free(training_memory);
  Serial.printf("Training memory freed, Free PSRAM: %u bytes\n", ESP.getFreePsram());
}

// Test the model
void test() {
  // Start testing
  Serial.println(F("Testing..."));

  // Allocate inference memory in PSRAM
  uint32_t inference_memory_size = aialgo_sizeof_inference_memory(&model);
  void *inference_memory = ps_malloc(inference_memory_size);
  if (!inference_memory) {
    Serial.println(F("Inference memory allocation failed"));
    while (1)
      ;
  }
  aialgo_schedule_inference_memory(&model, inference_memory, inference_memory_size);
  Serial.printf("Inference memory allocated: %u bytes, Free PSRAM: %u bytes\n",
                inference_memory_size, ESP.getFreePsram());
#if DEBUG
  // Verify first pixel directly from PROGMEM
  Serial.printf("First test pixel: %.4f\n", pgm_read_float(&test_input_data[0]));  // ~-0.4242
#endif

  // Create tensors directly from PROGMEM
  const uint16_t input_shape[] = { TEST_DATASET, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH };
  const uint16_t target_shape[] = { TEST_DATASET, OUTPUT_SIZE };
  aitensor_t input_tensor = AITENSOR_4D_F32(input_shape, (float *)test_input_data);
  aitensor_t target_tensor = AITENSOR_2D_F32(target_shape, (float *)test_target_data);

// Log tensor shapes if DEBUG enabled
#if DEBUG
  Serial.printf("Input tensor shape: [%u,%u,%u,%u]\n",
                input_tensor.shape[0], input_tensor.shape[1],
                input_tensor.shape[2], input_tensor.shape[3]);
  Serial.printf("Target tensor shape: [%u,%u]\n",
                target_tensor.shape[0], target_tensor.shape[1]);
  Serial.printf("Free SRAM after tensors: %u bytes\n", ESP.getFreeHeap());
#endif

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
      for (uint32_t j = 0; j < 10; j++) {  // Changed from j=1 to j=0
        if (pgm_read_float(&test_target_data[i][j]) == 1.0f) {
          true_label = j;
          break;  // Exit loop once true label is found
        }
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

  // Free training memory
  free(inference_memory);
  Serial.printf("Inference memory freed, Free PSRAM: %u bytes\n", ESP.getFreePsram());
}

// Infer function (stub)
void infer() {
  // Placeholder for single-image inference
  Serial.println(F("Inferring..."));
}

// Setup function
void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  while (!Serial)
    ;

  // Print ESP32 system information
  Serial.println(F("\n##################################"));
  Serial.println(F("ESP32 Information:"));
  Serial.printf("Internal Total Heap %d, Internal Used Heap %d, Internal Free Heap %d\n",
                ESP.getHeapSize(), ESP.getHeapSize() - ESP.getFreeHeap(), ESP.getFreeHeap());
  Serial.printf("Sketch Size %d, Free Sketch Space %d\n",
                ESP.getSketchSize(), ESP.getFreeSketchSpace());
  Serial.printf("SPIRam Total heap %d, SPIRam Free Heap %d\n",
                ESP.getPsramSize(), ESP.getFreePsram());
  Serial.printf("Chip Model %s, ChipRevision %d, Cpu Freq %d, SDK Version %s\n",
                ESP.getChipModel(), ESP.getChipRevision(), ESP.getCpuFreqMHz(), ESP.getSdkVersion());
  Serial.printf("Flash Size %d, Flash Speed %d\n",
                ESP.getFlashChipSize(), ESP.getFlashChipSpeed());
  Serial.println(F("##################################\n\n"));

  // Initialize PSRAM
  if (!psramInit()) {
    Serial.println(F("PSRAM initialization failed"));
    while (1)
      ;
  }
  Serial.println(F("PSRAM initialized"));

  // Seed random number generator
  srand(analogRead(A5));

  // Initialize model
  init_model();
  Serial.println(F("Type >t< or >infer<"));
}

// Main loop
void loop() {
  // Handle serial commands
  if (Serial.available() > 0) {
    String str = Serial.readString();
    if (str.indexOf("t") > -1) {
      // train();
      test();
    } else if (str.indexOf("infer") > -1) {
      infer();
    } else {
      Serial.println(F("Unknown command"));
    }
  }
}