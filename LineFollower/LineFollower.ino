#include <aifes.h>
#include <esp_heap_caps.h>
#include "data.h" // x_train_data[3][3], y_train_data[3][2] in PROGMEM

#define DATASETS 3
#define INPUT_SIZE 3
#define HIDDEN_NEURONS 3
#define OUTPUT_SIZE 2
#define EPOCHS 100
#define BATCH_SIZE 3
#define PRINT_INTERVAL 10

// Model globals
aimodel_t model;
uint16_t input_layer_shape[] = {1, INPUT_SIZE};
ailayer_input_f32_t input_layer = AILAYER_INPUT_F32_A(2, input_layer_shape);
ailayer_dense_f32_t dense_layer_1 = AILAYER_DENSE_F32_A(HIDDEN_NEURONS);
ailayer_leaky_relu_f32_t leaky_relu_layer = AILAYER_LEAKY_RELU_F32_A(0.01f);
ailayer_dense_f32_t dense_layer_2 = AILAYER_DENSE_F32_A(OUTPUT_SIZE);
ailayer_sigmoid_f32_t sigmoid_layer = AILAYER_SIGMOID_F32_A();

// Initialize the neural network model
void init_model() {
  Serial.println(F("Initializing model..."));

  // Link layers sequentially
  ailayer_t *x;
  model.input_layer = ailayer_input_f32_default(&input_layer);
  x = ailayer_dense_f32_default(&dense_layer_1, model.input_layer);
  x = ailayer_leaky_relu_f32_default(&leaky_relu_layer, x);
  x = ailayer_dense_f32_default(&dense_layer_2, x);
  model.output_layer = ailayer_sigmoid_f32_default(&sigmoid_layer, x);

  // Compile the model to check connections
  aialgo_compile_model(&model);

  // Allocate parameter memory in PSRAM
  uint32_t parameter_memory_size = aialgo_sizeof_parameter_memory(&model);
  void *parameter_memory = ps_malloc(parameter_memory_size);
  if (!parameter_memory) {
    Serial.println(F("Model memory allocation failed"));
    while (1);
  }

  // Distribute memory to trainable parameters
  aialgo_distribute_parameter_memory(&model, parameter_memory, parameter_memory_size);
  Serial.printf("Model memory allocated: %u bytes, Free PSRAM: %u bytes\n",
                parameter_memory_size, ESP.getFreePsram());

  // Print model structure
  aiprint("\n-------------- Model structure ---------------\n");
  aialgo_print_model_structure(&model);
  aiprint("----------------------------------------------\n\n");

  Serial.println(F("Model initialized"));
}

// Train the neural network
void train() {
  Serial.println(F("Training..."));

  // Configure cross-entropy loss
  ailoss_crossentropy_f32_t crossentropy_loss;
  model.loss = ailoss_crossentropy_f32_default(&crossentropy_loss, model.output_layer);
  aialgo_print_loss_specs(model.loss);
  aiprint("\n");

  // Configure Adam optimizer
  aiopti_adam_f32_t adam_opti = {
    .learning_rate = 0.01f,
    .beta1 = 0.9f,
    .beta2 = 0.999f,
    .eps = 1e-7f
  };
  aiopti_t *optimizer = aiopti_adam_f32_default(&adam_opti);
  aialgo_print_optimizer_specs(optimizer);
  aiprint("\n");

  // Initialize trainable parameters with random seed
  srand(micros());
  aialgo_initialize_parameters_model(&model);

  // Allocate training memory in PSRAM
  uint32_t training_memory_size = aialgo_sizeof_training_memory(&model, optimizer);
  void *training_memory = ps_malloc(training_memory_size);
  if (!training_memory) {
    Serial.println(F("PSRAM training memory allocation failed"));
    while (1);
  }

  // Schedule training memory
  aialgo_schedule_training_memory(&model, optimizer, training_memory, training_memory_size);
  Serial.printf("Training memory allocated: %u bytes, Free PSRAM: %u bytes\n",
                training_memory_size, ESP.getFreePsram());

  aialgo_init_model_for_training(&model, optimizer);

  // Load training data from PROGMEM (data.h)
  const uint16_t x_train_shape[] = {DATASETS, INPUT_SIZE}; // [3,3]
  const uint16_t y_train_shape[] = {DATASETS, OUTPUT_SIZE}; // [3,2]
  aitensor_t x_train = AITENSOR_2D_F32(x_train_shape, (float*)x_train_data);
  aitensor_t y_train = AITENSOR_2D_F32(y_train_shape, (float*)y_train_data);
  aitensor_t *x_test = &x_train;
  aitensor_t *y_test = &y_train;

  // Perform training
  float loss;
  aiprint("\nStart training\n");
  for (int i = 0; i < EPOCHS; i++) {
    aialgo_train_model(&model, &x_train, &y_train, optimizer, BATCH_SIZE);
    if (i % PRINT_INTERVAL == 0) {
      aialgo_calc_loss_model_f32(&model, x_test, y_test, &loss);
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

  test();
}

// Test model accuracy
void test() {
  Serial.println(F("Testing..."));

  // Load test data from PROGMEM (data.h)
  const uint16_t x_test_shape[] = {DATASETS, INPUT_SIZE}; // [3,3]
  const uint16_t y_test_shape[] = {DATASETS, OUTPUT_SIZE}; // [3,2]
  aitensor_t x_test = AITENSOR_2D_F32(x_test_shape, (float*)x_train_data);
  aitensor_t y_test = AITENSOR_2D_F32(y_test_shape, (float*)y_train_data);

  // Allocate output tensor in SRAM
  uint16_t y_out_shape[] = {DATASETS, OUTPUT_SIZE}; // [3,2]
  float y_out_data[DATASETS * OUTPUT_SIZE];
  aitensor_t y_out = AITENSOR_2D_F32(y_out_shape, y_out_data);

  // Allocate inference memory in PSRAM
  uint32_t inference_memory_size = aialgo_sizeof_inference_memory(&model);
  void *inference_memory = ps_malloc(inference_memory_size);
  if (!inference_memory) {
    Serial.println(F("PSRAM inference memory allocation failed"));
    while (1);
  }
  aialgo_schedule_inference_memory(&model, inference_memory, inference_memory_size);
  Serial.printf("Inference memory allocated: %u bytes, Free PSRAM: %u bytes\n",
                inference_memory_size, ESP.getFreePsram());

  // Run inference
  aialgo_inference_model(&model, &x_test, &y_out);

  // Print results
  aiprint("x_test:\n");
  print_aitensor(&x_test);
  aiprint("y_test:\n");
  print_aitensor(&y_test);
  aiprint("NN output:\n");
  print_aitensor(&y_out);

  // Free inference memory
  free(inference_memory);
  Serial.printf("Inference memory freed, Free PSRAM: %u bytes\n", ESP.getFreePsram());
}

// Setup with diagnostics and PSRAM check
void setup() {
  Serial.begin(115200);
  while (!Serial);

  // Print ESP32 diagnostics
  Serial.println(F("\n##################################"));
  Serial.println(F("ESP32 Information:"));
  Serial.printf("Internal Total Heap %d, Internal Used Heap %d, Internal Free Heap %d\n",
                ESP.getHeapSize(), ESP.getHeapSize() - ESP.getFreeHeap(), ESP.getFreeHeap());
  Serial.printf("Sketch Size %d, Free Sketch Space %d\n",
                ESP.getSketchSize(), ESP.getFreeSketchSpace());
  Serial.printf("SPIRam Total heap %d, SPIRam Free Heap %d\n",
                ESP.getPsramSize(), ESP.getFreePsram());
  Serial.printf("Chip Model %s, Chip Revision %d, Cpu Freq %d, SDK Version %s\n",
                ESP.getChipModel(), ESP.getChipRevision(), ESP.getCpuFreqMHz(), ESP.getSdkVersion());
  Serial.printf("Flash Size %d, Flash Speed %d\n",
                ESP.getFlashChipSize(), ESP.getFlashChipSpeed());
  Serial.println(F("##################################\n\n"));

  // Initialize PSRAM
  if (!psramInit()) {
    Serial.println(F("PSRAM initialization failed"));
    while (1);
  }
  Serial.println(F("PSRAM initialized"));

  // Seed random number generator
  srand(analogRead(A5));

  // Initialize and compile model
  init_model();
  Serial.println(F("Type >train<"));
}

// Main loop for command input
void loop() {
  if (Serial.available() > 0) {
    String str = Serial.readString();
    if (str.indexOf("train") > -1) {
      train();
    } else {
      Serial.println(F("Unknown command"));
    }
  }
}