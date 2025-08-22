#include <aifes.h>

#define DATASETS 4
#define FNN_3_LAYERS 3
#define INPUT_SIZE 2
#define HIDDEN_SIZE 3
#define OUTPUT_SIZE 1
#define PRINT_INTERVAL 10

uint32_t global_epoch_counter = 0;

const float input_data[DATASETS][INPUT_SIZE] = {
  { 0.0f, 0.0f }, { 0.0f, 1.0f }, { 1.0f, 0.0f }, { 1.0f, 1.0f }
};
const float target_data[DATASETS][OUTPUT_SIZE] = {
  { 0.0f }, { 1.0f }, { 1.0f }, { 0.0f }
};

float *output_data;
float *flat_weights;
aitensor_t input_tensor, target_tensor, output_tensor;
AIFES_E_model_parameter_fnn_f32 fnn;

uint32_t fnn_structure[FNN_3_LAYERS] = { INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE };
uint16_t input_shape[] = { DATASETS, INPUT_SIZE };
uint16_t target_shape[] = { DATASETS, OUTPUT_SIZE };
uint16_t output_shape[] = { DATASETS, OUTPUT_SIZE };
AIFES_E_activations fnn_activations[FNN_3_LAYERS - 1];

void printLoss(float loss) {
  global_epoch_counter++;
  Serial.print(F("Epoch: "));
  Serial.print(global_epoch_counter * PRINT_INTERVAL);
  Serial.print(F(" / Loss: "));
  Serial.println(loss, 5);
}

void init_model() {
  Serial.println("Initializing model...");
  input_tensor = AITENSOR_2D_F32(input_shape, input_data);
  target_tensor = AITENSOR_2D_F32(target_shape, target_data);
  output_data = (float *)malloc(DATASETS * OUTPUT_SIZE * sizeof(float));
  output_tensor = AITENSOR_2D_F32(output_shape, output_data);

  fnn_activations[0] = AIfES_E_sigmoid;
  fnn_activations[1] = AIfES_E_sigmoid;

  uint32_t weight_number = AIFES_E_flat_weights_number_fnn_f32(fnn_structure, FNN_3_LAYERS);
  flat_weights = (float *)malloc(weight_number * sizeof(float));

  fnn.layer_count = FNN_3_LAYERS;
  fnn.fnn_structure = fnn_structure;
  fnn.fnn_activations = fnn_activations;
  fnn.flat_weights = flat_weights;
}

void train() {
  Serial.println("Training...");
  global_epoch_counter = 0;

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
  train_params.epochs = 10000;
  train_params.epochs_loss_print_interval = PRINT_INTERVAL;
  train_params.loss_print_function = printLoss;
  train_params.early_stopping = AIfES_E_early_stopping_on;
  train_params.early_stopping_target_loss = 0.004f;

  int8_t error = AIFES_E_training_fnn_f32(&input_tensor, &target_tensor, &fnn, &train_params, &init_weights, &output_tensor);
  error_handling_training(error);
  test();
}

void infer() {
  Serial.println("Inferring...");
  float single_input[1][INPUT_SIZE] = { { 0.0f, 0.0f } };  // Example input
  uint16_t single_input_shape[] = { 1, INPUT_SIZE };
  aitensor_t single_input_tensor = AITENSOR_2D_F32(single_input_shape, (float *)single_input);
  float single_output[1];
  uint16_t single_output_shape[] = { 1, OUTPUT_SIZE };
  aitensor_t single_output_tensor = AITENSOR_2D_F32(single_output_shape, single_output);
  int8_t error = AIFES_E_inference_fnn_f32(&single_input_tensor, &fnn, &single_output_tensor);
  error_handling_inference(error);
  Serial.print(F("Predicted: "));
  Serial.println(single_output[0] >= 0.5f ? 1 : 0);
}

void test() {
  Serial.println("Testing...");
  int8_t error = AIFES_E_inference_fnn_f32(&input_tensor, &fnn, &output_tensor);
  error_handling_inference(error);
  uint32_t correct = 0;
  for (uint32_t i = 0; i < DATASETS; i++) {
    int pred = output_data[i] >= 0.5f ? 1 : 0;
    int true_label = pgm_read_float(&target_data[i][0]) >= 0.5f ? 1 : 0;
    if (pred == true_label) correct++;
  }
  float accuracy = (float)correct / DATASETS * 100.0f;
  Serial.print(F("Accuracy: "));
  Serial.print(accuracy, 2);
  Serial.println(F("%"));
}

void setup() {
  Serial.begin(115200);
  while (!Serial)
    ;
  srand(analogRead(A5));
  init_model();
  Serial.println("Type >train< or >infer<");
}

void loop() {
  if (Serial.available() > 0) {
    String str = Serial.readString();
    if (str.indexOf("train") > -1) {
      train();
    } else if (str.indexOf("infer") > -1) {
      infer();
    } else {
      Serial.println("Unknown command");
    }
  }
}

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
