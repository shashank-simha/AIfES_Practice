#include <aifes.h>
#include <esp_heap_caps.h>

// MNIST and CNN constants
#define DATASETS 100      // Number of MNIST training/test samples
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
  { 0, 0 }              // Max pooling padding
#define DENSE1_SIZE 64  // Dense layer neurons
#define LAYER_COUNT 11  // Layers: input, conv1, relu1, pool1, conv2, relu2, pool2, flatten, dense1, relu3, dense2, softmax

// Global model variables
aimodel_t model;                 // Neural network struct
ailayer_t *layers[LAYER_COUNT];  // Array of layer pointers
void *parameter_memory;          // PSRAM for weights/biases

// Layer structures
uint16_t input_shape[] = { DATASETS, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH };                                // Input: [100,1,28,28]
ailayer_input_f32_t input_layer = AILAYER_INPUT_F32_A(4, input_shape);                                           // 4D input tensor
ailayer_conv2d_f32_t conv1_layer = AILAYER_CONV2D_F32_A(CONV1_FILTERS, KERNEL_SIZE, STRIDE, DILATION, PADDING);  // Conv1: 1->8, 3x3, output [100,8,28,28]
ailayer_relu_f32_t relu1_layer = AILAYER_RELU_F32_A();                                                           // ReLU activation
ailayer_maxpool2d_f32_t pool1_layer = AILAYER_MAXPOOL2D_F32_A(POOL_SIZE, POOL_STRIDE, POOL_PADDING);             // MaxPool: 2x2, output [100,8,14,14]
ailayer_conv2d_f32_t conv2_layer = AILAYER_CONV2D_F32_A(CONV2_FILTERS, KERNEL_SIZE, STRIDE, DILATION, PADDING);  // Conv2: 8->16, 3x3, output [100,16,14,14]
ailayer_relu_f32_t relu2_layer = AILAYER_RELU_F32_A();                                                           // ReLU activation
ailayer_maxpool2d_f32_t pool2_layer = AILAYER_MAXPOOL2D_F32_A(POOL_SIZE, POOL_STRIDE, POOL_PADDING);             // MaxPool: 2x2, output [100,16,7,7]
ailayer_flatten_f32_t flatten_layer = AILAYER_FLATTEN_F32_A();                                                   // Flatten to [100,784]
ailayer_dense_f32_t dense1_layer = AILAYER_DENSE_F32_A(DENSE1_SIZE);                                             // Dense: 784->64
ailayer_relu_f32_t relu3_layer = AILAYER_RELU_F32_A();                                                           // ReLU activation
ailayer_dense_f32_t dense2_layer = AILAYER_DENSE_F32_A(OUTPUT_SIZE);                                             // Dense: 64->10
ailayer_softmax_f32_t softmax_layer = AILAYER_SOFTMAX_F32_A();                                                   // Softmax for 10-class output

// Error handling function prototypes
void error_handling_training(int8_t error_nr);
void error_handling_inference(int8_t error_nr);

// Initialize CNN model
void init_model() {
  Serial.println(F("Initializing model..."));

  // Connect layers and populate layers array
  ailayer_t *x;
  layers[0] = model.input_layer = ailayer_input_f32_default(&input_layer);  // Input: [100,1,28,28]
  if (!model.input_layer) {
    Serial.println(F("Input layer initialization failed"));
    while (1)
      ;
  }

  conv1_layer.channel_axis = 1;                                                 // NCHW
  layers[1] = x = ailayer_conv2d_f32_default(&conv1_layer, model.input_layer);  // Conv1: 1->8, 3x3, output [100,8,28,28]
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
  layers[3] = x = ailayer_maxpool2d_f32_default(&pool1_layer, x);  // MaxPool: 2x2, output [100,8,14,14]
  if (!x) {
    Serial.println(F("MaxPool1 layer initialization failed"));
    while (1)
      ;
  }

  conv2_layer.channel_axis = 1;                                 // NCHW
  layers[4] = x = ailayer_conv2d_f32_default(&conv2_layer, x);  // Conv2: 8->16, 3x3, output [100,16,14,14]
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
  layers[6] = x = ailayer_maxpool2d_f32_default(&pool2_layer, x);  // MaxPool: 2x2, output [100,16,7,7]
  if (!x) {
    Serial.println(F("MaxPool2 layer initialization failed"));
    while (1)
      ;
  }

  layers[7] = x = ailayer_flatten_f32_default(&flatten_layer, x);  // Flatten to [100,784]
  if (!x) {
    Serial.println(F("Flatten layer initialization failed"));
    while (1)
      ;
  }

  layers[8] = x = ailayer_dense_f32_default(&dense1_layer, x);  // Dense: 784->64
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

  layers[10] = x = ailayer_dense_f32_default(&dense2_layer, x);  // Dense: 64->10
  if (!x) {
    Serial.println(F("Dense2 layer initialization failed"));
    while (1)
      ;
  }

  model.output_layer = ailayer_softmax_f32_default(&softmax_layer, x);  // Softmax: 10-class output
  if (!model.output_layer) {
    Serial.println(F("Softmax layer initialization failed"));
    while (1)
      ;
  }

  // Compile model
  aialgo_compile_model(&model);

  // Allocate parameter memory in PSRAM
  Serial.printf("Free PSRAM before allocation: %u bytes\n", ESP.getFreePsram());
  uint32_t parameter_memory_size = aialgo_sizeof_parameter_memory(&model);
  parameter_memory = ps_malloc(parameter_memory_size);
  if (!parameter_memory) {
    Serial.println(F("Parameter memory allocation failed"));
    while (1)
      ;
  }
  aialgo_distribute_parameter_memory(&model, parameter_memory, parameter_memory_size);

  // Initialize weights (Glorot uniform) and biases (zeros)
  srand(micros());
  aimath_f32_default_init_glorot_uniform(&conv1_layer.weights);
  aimath_f32_default_init_zeros(&conv1_layer.bias);
  aimath_f32_default_init_glorot_uniform(&conv2_layer.weights);
  aimath_f32_default_init_zeros(&conv2_layer.bias);
  aimath_f32_default_init_glorot_uniform(&dense1_layer.weights);
  aimath_f32_default_init_zeros(&dense1_layer.bias);
  aimath_f32_default_init_glorot_uniform(&dense2_layer.weights);
  aimath_f32_default_init_zeros(&dense2_layer.bias);
  aialgo_initialize_parameters_model(&model);

  // Debug: Print model structure and tensor shapes
  Serial.println(F("\n-------------- Model structure ---------------"));
  aialgo_print_model_structure(&model);
  Serial.printf("Parameter memory: %u bytes\n", parameter_memory_size);
  Serial.printf("Free PSRAM after allocation: %u bytes\n", ESP.getFreePsram());
  // Verify tensor shapes
  Serial.println(F("Tensor shapes:"));
  Serial.printf("Input: [%u,%u,%u,%u]\n", layers[0]->result.shape[0], layers[0]->result.shape[1],
                layers[0]->result.shape[2], layers[0]->result.shape[3]);
  Serial.printf("Conv1: [%u,%u,%u,%u]\n", layers[1]->result.shape[0], layers[1]->result.shape[1],
                layers[1]->result.shape[2], layers[1]->result.shape[3]);
  Serial.printf("Pool1: [%u,%u,%u,%u]\n", layers[3]->result.shape[0], layers[3]->result.shape[1],
                layers[3]->result.shape[2], layers[3]->result.shape[3]);
  Serial.printf("Conv2: [%u,%u,%u,%u]\n", layers[4]->result.shape[0], layers[4]->result.shape[1],
                layers[4]->result.shape[2], layers[4]->result.shape[3]);
  Serial.printf("Pool2: [%u,%u,%u,%u]\n", layers[6]->result.shape[0], layers[6]->result.shape[1],
                layers[6]->result.shape[2], layers[6]->result.shape[3]);
  Serial.printf("Dense1: [%u,%u]\n", layers[8]->result.shape[0], layers[8]->result.shape[1]);
  Serial.printf("Dense2: [%u,%u]\n", layers[10]->result.shape[0], layers[10]->result.shape[1]);
  Serial.println(F("----------------------------------------------\n"));
  Serial.println(F("Model initialized"));
}

// Train the CNN
void train() {
  Serial.println(F("Training..."));
  // TODO: Load MNIST data, create tensors, train with SGD and cross-entropy
}

// Run inference on a single image
void infer() {
  Serial.println(F("Inferring..."));
  // TODO: Create single image tensor, run inference, output predicted digit
}

// Test model accuracy
void test() {
  Serial.println(F("Testing..."));
  // TODO: Load test data, run inference, compute accuracy
}

// Setup with PSRAM check and diagnostics
void setup() {
  Serial.begin(115200);
  while (!Serial)
    ;

  // Print ESP32 diagnostics
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
    if (str.indexOf("train") > -1) {
      train();
    } else if (str.indexOf("infer") > -1) {
      infer();
    } else {
      Serial.println(F("Unknown command"));
    }
  }
}

// Handle training errors
void error_handling_training(int8_t error_nr) {
  switch (error_nr) {
    case 0: break;
    case -1: Serial.println(F("ERROR! Tensor dtype")); break;
    case -2: Serial.println(F("ERROR! Tensor shape: Data Number")); break;
    case -3: Serial.println(F("ERROR! Input tensor shape does not correspond to ANN inputs")); break;
    case -4: Serial.println(F("ERROR! Output tensor shape does not correspond to ANN outputs")); break;
    case -5: Serial.println(F("ERROR! Use crossentropy as loss for softmax")); break;
    case -6: Serial.println(F("ERROR! learn_rate or sgd_momentum negative")); break;
    case -7: Serial.println(F("ERROR! Init uniform weights min - max wrong")); break;
    case -8: Serial.println(F("ERROR! batch_size: min = 1 / max = Number of training data")); break;
    case -9: Serial.println(F("ERROR! Unknown activation function")); break;
    case -10: Serial.println(F("ERROR! Unknown loss function")); break;
    case -11: Serial.println(F("ERROR! Unknown init weights method")); break;
    case -12: Serial.println(F("ERROR! Unknown optimizer")); break;
    case -13: Serial.println(F("ERROR! Not enough memory")); break;
    default: Serial.println(F("Unknown error"));
  }
}

// Handle inference errors
void error_handling_inference(int8_t error_nr) {
  switch (error_nr) {
    case 0: break;
    case -1: Serial.println(F("ERROR! Tensor dtype")); break;
    case -2: Serial.println(F("ERROR! Tensor shape: Data Number")); break;
    case -3: Serial.println(F("ERROR! Input tensor shape does not correspond to ANN inputs")); break;
    case -4: Serial.println(F("ERROR! Output tensor shape does not correspond to ANN outputs")); break;
    case -5: Serial.println(F("ERROR! Unknown activation function")); break;
    case -6: Serial.println(F("ERROR! Not enough memory")); break;
    default: Serial.println(F("Unknown error"));
  }
}