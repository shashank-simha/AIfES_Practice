#include <aifes.h>

void setup() {
  Serial.begin(115200);
  while (!Serial)
    ;

  /* ################## Create the neural network in AIfES ################### */
  // Declaration and configuration of the layers
  // The main model structure that holds the whole neural network
  aimodel_t model;

  // The layer structures for F32 data type and their configurations
  uint16_t input_layer_shape[] = { 1, 3 };
  ailayer_input_f32_t input_layer = AILAYER_INPUT_F32_A(2, input_layer_shape);
  ailayer_dense_f32_t dense_layer_1 = AILAYER_DENSE_F32_A(3);
  ailayer_leaky_relu_f32_t leaky_relu_layer = AILAYER_LEAKY_RELU_F32_A(0.01f);
  ailayer_dense_f32_t dense_layer_2 = AILAYER_DENSE_F32_A(2);
  ailayer_sigmoid_f32_t sigmoid_layer = AILAYER_SIGMOID_F32_A();


  // Connection and initialization of the layers
  // Layer pointer to perform the connection
  ailayer_t *x;

  model.input_layer = ailayer_input_f32_default(&input_layer);
  x = ailayer_dense_f32_default(&dense_layer_1, model.input_layer);
  x = ailayer_leaky_relu_f32_default(&leaky_relu_layer, x);
  x = ailayer_dense_f32_default(&dense_layer_2, x);
  x = ailayer_sigmoid_f32_default(&sigmoid_layer, x);
  model.output_layer = x;

  // Finish the model creation by checking the connections and setting some parameters for further processing
  aialgo_compile_model(&model);


  // Set the memory for the trainable parameters
  uint32_t parameter_memory_size = aialgo_sizeof_parameter_memory(&model);
  void *parameter_memory = malloc(parameter_memory_size);

  // Distribute the memory to the trainable parameters of the model
  aialgo_distribute_parameter_memory(&model, parameter_memory, parameter_memory_size);

  // Print the layer structure to the console
  aiprint("\n-------------- Model structure ---------------\n");
  aialgo_print_model_structure(&model);
  aiprint("----------------------------------------------\n\n");

  /* ############################################################################### */

  /* ################## Train the neural network ###################*/
  // Configure the loss
  ailoss_crossentropy_f32_t crossentropy_loss;
  model.loss = ailoss_crossentropy_f32_default(&crossentropy_loss, model.output_layer);
  aialgo_print_loss_specs(model.loss);
  aiprint("\n");

  // Configure the optimizer
  aiopti_adam_f32_t adam_opti = {
    .learning_rate = 0.01f,

    .beta1 = 0.9f,
    .beta2 = 0.999f,
    .eps = 1e-7f
  };
  aiopti_t *optimizer = aiopti_adam_f32_default(&adam_opti);
  aialgo_print_optimizer_specs(optimizer);
  aiprint("\n");

  // Initialize the trainable parameters
  // Set the seed for your configured random function for example with
  srand(time(0));
  aialgo_initialize_parameters_model(&model);

  // Allocate and initialize the working memory
  uint32_t training_memory_size = aialgo_sizeof_training_memory(&model, optimizer);
  void *training_memory = malloc(training_memory_size);

  // Schedule the memory to the model
  aialgo_schedule_training_memory(&model, optimizer, training_memory, training_memory_size);

  aialgo_init_model_for_training(&model, optimizer);

  // Perform the training
  uint16_t x_train_shape[2] = { 3, 3 };
  float x_train_data[3 * 3] = { 0.0f, 0.0f, 0.0f,
                                1.0f, 1.0f, 1.0f,
                                1.0f, 0.0f, 0.0f };
  aitensor_t x_train = AITENSOR_2D_F32(x_train_shape, x_train_data);

  // Target data / Labels for training
  uint16_t y_train_shape[2] = { 3, 2 };
  float y_train_data[3 * 2] = { 1.0f, 0.0f,
                                0.0f, 1.0f,
                                0.0f, 0.0f };
  aitensor_t y_train = AITENSOR_2D_F32(y_train_shape, y_train_data);

  aitensor_t *x_test = &x_train;
  aitensor_t *y_test = &y_train;

  int epochs = 100;
  int batch_size = 3;
  int print_interval = 10;

  float loss;
  aiprint("\nStart training\n");
  for (int i = 0; i < epochs; i++) {
    // One epoch of training. Iterates through the whole data once
    aialgo_train_model(&model, &x_train, &y_train, optimizer, batch_size);

    // Calculate and print loss every print_interval epochs
    if (i % print_interval == 0) {
      aialgo_calc_loss_model_f32(&model, x_test, y_test, &loss);

      // Print the loss to the console
      aiprint("Epoch ");
      aiprint_int("%5d", i);
      aiprint(": test loss: ");
      aiprint_float("%f", loss);
      aiprint("\n");
    }
  }
  aiprint("Finished training\n\n");
  /* ############################################################################### */

  /* ################## Test the trained model ###################*/
  // Create an empty tensor for the inference results
  uint16_t y_out_shape[2] = { 3, 2 };
  float y_out_data[3 * 2];
  aitensor_t y_out = AITENSOR_2D_F32(y_out_shape, y_out_data);

  aialgo_inference_model(&model, x_test, &y_out);

  aiprint("x_test:\n");
  print_aitensor(x_test);
  aiprint("NN output:\n");
  print_aitensor(&y_out);
  /* ############################################################################### */
}

void loop() {
  // put your main code here, to run repeatedly:
}
