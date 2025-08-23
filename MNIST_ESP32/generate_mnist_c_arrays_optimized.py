import numpy as np
from keras.datasets import mnist

# Configurable dataset sizes
NUM_TRAIN = 10  # Number of training samples
NUM_TEST = 5  # Number of test samples

# Load MNIST data (raw uint8)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Select specified number of samples
x_train = x_train[:NUM_TRAIN].astype(np.uint8)
y_train = y_train[:NUM_TRAIN].astype(np.uint8)
x_test = x_test[:NUM_TEST].astype(np.uint8)
y_test = y_test[:NUM_TEST].astype(np.uint8)

# Generate C header file
with open("mnist_data.h", "w") as f:
    f.write("#ifndef MNIST_DATA_H\n#define MNIST_DATA_H\n#include <pgmspace.h>\n\n")

    # Training input data (uint8)
    f.write(f"const uint8_t train_input_data[{NUM_TRAIN}][1][28][28] PROGMEM = ")
    f.write(str(x_train.tolist()).replace("[", "{").replace("]", "}"))
    f.write(";\n\n")

    # Training target data (uint8, 0-9)
    f.write(f"const uint8_t train_target_data[{NUM_TRAIN}] PROGMEM = ")
    f.write(str(y_train.tolist()).replace("[", "{").replace("]", "}"))
    f.write(";\n\n")

    # Test input data (uint8)
    f.write(f"const uint8_t test_input_data[{NUM_TEST}][1][28][28] PROGMEM = ")
    f.write(str(x_test.tolist()).replace("[", "{").replace("]", "}"))
    f.write(";\n\n")

    # Test target data (uint8, 0-9)
    f.write(f"const uint8_t test_target_data[{NUM_TEST}] PROGMEM = ")
    f.write(str(y_test.tolist()).replace("[", "{").replace("]", "}"))
    f.write(";\n#endif")
