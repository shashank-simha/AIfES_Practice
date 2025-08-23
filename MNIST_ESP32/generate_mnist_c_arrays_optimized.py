import numpy as np
from keras.datasets import mnist

# Configuration
NUM_TRAIN = 10  # Number of training samples
NUM_TEST = 5  # Number of test samples
NORMALIZED = True  # True: float (normalized), False: uint8 (raw)

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Select specified number of samples
x_train = x_train[:NUM_TRAIN]
y_train = y_train[:NUM_TRAIN].astype(np.uint8)
x_test = x_test[:NUM_TEST]
y_test = y_test[:NUM_TEST].astype(np.uint8)

# Normalize data if NORMALIZED=True
if NORMALIZED:
    x_train = (x_train.astype(np.float32) / 255.0 - 0.1307) / 0.3081
    x_test = (x_test.astype(np.float32) / 255.0 - 0.1307) / 0.3081
else:
    x_train = x_train.astype(np.uint8)
    x_test = x_test.astype(np.uint8)

# Convert labels to one-hot encoding if NORMALIZED=True
if NORMALIZED:
    y_train_onehot = np.zeros((NUM_TRAIN, 10), dtype=np.float32)
    y_test_onehot = np.zeros((NUM_TEST, 10), dtype=np.float32)
    for i in range(NUM_TRAIN):
        y_train_onehot[i, y_train[i]] = 1.0
    for i in range(NUM_TEST):
        y_test_onehot[i, y_test[i]] = 1.0
else:
    y_train_onehot = y_train
    y_test_onehot = y_test


# Function to format 4D array as C initializer
def format_4d_array(data, num_samples, is_float):
    result = "{"
    for i in range(num_samples):
        result += "{"
        result += "{"
        for row in range(28):
            result += "{"
            if is_float:
                result += ",".join([f"{x:.6f}" for x in data[i][row]])
            else:
                result += ",".join(map(str, data[i][row]))
            result += "}"
            if row < 27:
                result += ","
        result += "}"
        result += "}"
        if i < num_samples - 1:
            result += ","
    result += "}"
    return result


# Function to format 1D/2D array as C initializer
def format_array(data, is_float, is_2d=False):
    if is_2d:
        result = "{"
        for i in range(len(data)):
            result += (
                "{"
                + ",".join([f"{x:.1f}" if is_float else str(x) for x in data[i]])
                + "}"
            )
            if i < len(data) - 1:
                result += ","
        result += "}"
    else:
        result = (
            "{" + ",".join([f"{x:.1f}" if is_float else str(x) for x in data]) + "}"
        )
    return result


# Generate C header file
with open("mnist_data.h", "w") as f:
    f.write("#ifndef MNIST_DATA_H\n#define MNIST_DATA_H\n#include <pgmspace.h>\n\n")

    # Training input data
    data_type = "float" if NORMALIZED else "uint8_t"
    f.write(f"const {data_type} train_input_data[{NUM_TRAIN}][1][28][28] PROGMEM = ")
    f.write(format_4d_array(x_train, NUM_TRAIN, NORMALIZED))
    f.write(";\n\n")

    # Training target data
    if NORMALIZED:
        f.write(f"const float train_target_data[{NUM_TRAIN}][10] PROGMEM = ")
        f.write(format_array(y_train_onehot, NORMALIZED, is_2d=True))
    else:
        f.write(f"const uint8_t train_target_data[{NUM_TRAIN}] PROGMEM = ")
        f.write(format_array(y_train_onehot, NORMALIZED))
    f.write(";\n\n")

    # Test input data
    f.write(f"const {data_type} test_input_data[{NUM_TEST}][1][28][28] PROGMEM = ")
    f.write(format_4d_array(x_test, NUM_TEST, NORMALIZED))
    f.write(";\n\n")

    # Test target data
    if NORMALIZED:
        f.write(f"const float test_target_data[{NUM_TEST}][10] PROGMEM = ")
        f.write(format_array(y_test_onehot, NORMALIZED, is_2d=True))
    else:
        f.write(f"const uint8_t test_target_data[{NUM_TEST}] PROGMEM = ")
        f.write(format_array(y_test_onehot, NORMALIZED))
    f.write(";\n#endif")
