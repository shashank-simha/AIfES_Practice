#include <aifes.h>
#include <esp_heap_caps.h>

#define DATASETS 4        // Number of XOR samples
#define FNN_3_LAYERS 3    // Neural network layers: input, hidden, output
#define INPUT_SIZE 2      // Input neurons (XOR inputs)
#define HIDDEN_SIZE 3     // Hidden layer neurons
#define OUTPUT_SIZE 1     // Output neurons (XOR output)
#define PRINT_INTERVAL 10 // Epochs between loss prints

// Model globals for persistence across train, test, infer
float *flat_weights;                           // Network weights in PSRAM
AIFES_E_model_parameter_fnn_f32 fnn;          // AIfES-Express model struct
uint32_t fnn_structure[FNN_3_LAYERS] = {INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE}; // Layer sizes
AIFES_E_activations fnn_activations[FNN_3_LAYERS - 1]; // Activations (sigmoid)

// Training data in PROGMEM
const float train_input_data[DATASETS][INPUT_SIZE] PROGMEM = {
  {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}
};
const float train_target_data[DATASETS][OUTPUT_SIZE] PROGMEM = {
  {0.0f}, {1.0f}, {1.0f}, {0.0f}
};

uint32_t global_epoch_counter = 0; // Tracks epochs for loss printing

// Print loss during training
void printLoss(float loss) {
  global_epoch_counter++;
  Serial.print(F("Epoch: "));
  Serial.print(global_epoch_counter * PRINT_INTERVAL);
  Serial.print(F(" / Loss: "));
  Serial.println(loss, 5);
}

// Initialize neural network model
void init_model() {
  Serial.println(F("Initializing model..."));

  // Set sigmoid activations for hidden and output layers
  fnn_activations[0] = AIfES_E_sigmoid;
  fnn_activations[1] = AIfES_E_sigmoid;

  // Calculate and allocate weights in PSRAM
  uint32_t weight_number = AIFES_E_flat_weights_number_fnn_f32(fnn_structure, FNN_3_LAYERS);
  flat_weights = (float *)ps_malloc(weight_number * sizeof(float));
  if (!flat_weights) {
    Serial.println(F("Weight allocation failed"));
    while (1);
  }

  // Configure AIfES-Express model
  fnn.layer_count = FNN_3_LAYERS;
  fnn.fnn_structure = fnn_structure;
  fnn.fnn_activations = fnn_activations;
  fnn.flat_weights = flat_weights;

  Serial.println(F("Model initialized"));
}

// Train the neural network
void train() {
  Serial.println(F("Training..."));
  global_epoch_counter = 0;

  // Local data in PSRAM (copied from PROGMEM)
  float *train_input_data_psram = (float *)ps_malloc(DATASETS * INPUT_SIZE * sizeof(float));
  float *train_target_data_psram = (float *)ps_malloc(DATASETS * OUTPUT_SIZE * sizeof(float));
  if (!train_input_data_psram || !train_target_data_psram) {
    Serial.println(F("Data allocation failed"));
    while (1);
  }
  for (uint32_t i = 0; i < DATASETS; i++) {
    for (uint32_t j = 0; j < INPUT_SIZE; j++) {
      train_input_data_psram[i * INPUT_SIZE + j] = pgm_read_float(&train_input_data[i][j]);
    }
    train_target_data_psram[i] = pgm_read_float(&train_target_data[i][0]);
  }

  // Local tensors
  uint16_t train_input_shape[] = {DATASETS, INPUT_SIZE};
  aitensor_t train_input_tensor = AITENSOR_2D_F32(train_input_shape, train_input_data_psram);
  uint16_t train_target_shape[] = {DATASETS, OUTPUT_SIZE};
  aitensor_t train_target_tensor = AITENSOR_2D_F32(train_target_shape, train_target_data_psram);
  float *train_output_data = (float *)ps_malloc(DATASETS * OUTPUT_SIZE * sizeof(float));
  if (!train_output_data) {
    Serial.println(F("Output allocation failed"));
    while (1);
  }
  uint16_t train_output_shape[] = {DATASETS, OUTPUT_SIZE};
  aitensor_t train_output_tensor = AITENSOR_2D_F32(train_output_shape, train_output_data);

  // Training parameters
  AIFES_E_init_weights_parameter_fnn_f32 init_weights;
  init_weights.init_weights_method = AIfES_E_init_uniform;
  init_weights.min_init_uniform = -2.0f;
  init_weights.max_init_uniform = 2.0f;

  AIFES_E_training_parameter_fnn_f32 train_params;
  train_params.optimizer = AIfES_E_adam;
  train_params.loss = AIfES_E_mse;
  train_params.learn_rate = 0.05f;
  train_params.sgd_momentum = 0.0f;
  train_params.batch_size = DATASETS;
  train_params.epochs = 1000;
  train_params.epochs_loss_print_interval = PRINT_INTERVAL;
  train_params.loss_print_function = printLoss;
  train_params.early_stopping = AIfES_E_early_stopping_on;
  train_params.early_stopping_target_loss = 0.004f;

  // Train model
  int8_t error = AIFES_E_training_fnn_f32(&train_input_tensor, &train_target_tensor, &fnn, &train_params, &init_weights, &train_output_tensor);
  error_handling_training(error);

  // Clean up
  free(train_input_data_psram);
  free(train_target_data_psram);
  free(train_output_data);

  test();
}

// Run inference on a single input
void infer() {
  Serial.println(F("Inferring..."));
  float single_input[1][INPUT_SIZE] = {{0.0f, 0.0f}}; // Example input
  uint16_t single_input_shape[] = {1, INPUT_SIZE};
  aitensor_t single_input_tensor = AITENSOR_2D_F32(single_input_shape, (float *)single_input);
  float single_output[1];
  uint16_t single_output_shape[] = {1, OUTPUT_SIZE};
  aitensor_t single_output_tensor = AITENSOR_2D_F32(single_output_shape, single_output);
  int8_t error = AIFES_E_inference_fnn_f32(&single_input_tensor, &fnn, &single_output_tensor);
  error_handling_inference(error);
  Serial.print(F("Predicted: "));
  Serial.println(single_output[0] >= 0.5f ? 1 : 0);
}

// Test model accuracy
void test() {
  Serial.println(F("Testing..."));
  // Local data in PSRAM (copied from PROGMEM)
  float *test_input_data_psram = (float *)ps_malloc(DATASETS * INPUT_SIZE * sizeof(float));
  float *test_target_data_psram = (float *)ps_malloc(DATASETS * OUTPUT_SIZE * sizeof(float));
  if (!test_input_data_psram || !test_target_data_psram) {
    Serial.println(F("Data allocation failed"));
    while (1);
  }
  for (uint32_t i = 0; i < DATASETS; i++) {
    for (uint32_t j = 0; j < INPUT_SIZE; j++) {
      test_input_data_psram[i * INPUT_SIZE + j] = pgm_read_float(&train_input_data[i][j]);
    }
    test_target_data_psram[i] = pgm_read_float(&train_target_data[i][0]);
  }

  // Local tensors
  uint16_t test_input_shape[] = {DATASETS, INPUT_SIZE};
  aitensor_t test_input_tensor = AITENSOR_2D_F32(test_input_shape, test_input_data_psram);
  uint16_t test_target_shape[] = {DATASETS, OUTPUT_SIZE};
  aitensor_t test_target_tensor = AITENSOR_2D_F32(test_target_shape, test_target_data_psram);
  float *test_output_data = (float *)ps_malloc(DATASETS * OUTPUT_SIZE * sizeof(float));
  if (!test_output_data) {
    Serial.println(F("Output allocation failed"));
    while (1);
  }
  uint16_t test_output_shape[] = {DATASETS, OUTPUT_SIZE};
  aitensor_t test_output_tensor = AITENSOR_2D_F32(test_output_shape, test_output_data);

  // Run inference
  int8_t error = AIFES_E_inference_fnn_f32(&test_input_tensor, &fnn, &test_output_tensor);
  error_handling_inference(error);

  // Print results
  Serial.println(F(""));
  Serial.println(F("Results:"));
  Serial.println(F("input 1:\tinput 2:\treal output:\tcalculated output:"));
  uint32_t correct = 0;
  for (uint32_t i = 0; i < DATASETS; i++) {
    Serial.print(test_input_data_psram[i * INPUT_SIZE]);
    Serial.print(F("\t\t"));
    Serial.print(test_input_data_psram[i * INPUT_SIZE + 1]);
    Serial.print(F("\t\t"));
    Serial.print(test_target_data_psram[i]);
    Serial.print(F("\t\t"));
    Serial.println(test_output_data[i], 5);
    int pred = test_output_data[i] >= 0.5f ? 1 : 0;
    int true_label = test_target_data_psram[i] >= 0.5f ? 1 : 0;
    if (pred == true_label) correct++;
  }
  float accuracy = (float)correct / DATASETS * 100.0f;
  Serial.print(F("Accuracy: "));
  Serial.print(accuracy, 2);
  Serial.println(F("%"));

  // Clean up
  free(test_input_data_psram);
  free(test_target_data_psram);
  free(test_output_data);
}

// Setup function with PSRAM check
void setup() {
  Serial.begin(115200);
  while (!Serial);

  // Check PSRAM initialization
  if (!psramInit()) {
    Serial.println(F("PSRAM initialization failed"));
    while (1);
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