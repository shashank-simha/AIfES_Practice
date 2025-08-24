import numpy as np
from keras.datasets import mnist

# Configuration
NUM_TRAIN = 200  # Number of training samples
NUM_TEST = 200  # Number of test samples
NUM_CLASSES = 10  # Number of classes
NORMALIZED = True  # True: float (normalized), False: uint8 (raw)

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Select balanced training samples: 20 per class (0-9)
np.random.seed(42)  # For reproducibility
train_idx = []
for i in range(10):  # Classes 0-9
    class_indices = np.where(y_train == i)[0]
    train_idx.extend(
        np.random.choice(
            class_indices, size=int(NUM_TRAIN / NUM_CLASSES), replace=False
        )
    )
train_idx = np.array(train_idx)
x_train = x_train[train_idx]
y_train = y_train[train_idx].astype(np.uint8)

# Select balanced test samples: 2 per class (0-9)
test_idx = []
for i in range(10):  # Classes 0-9
    class_indices = np.where(y_test == i)[0]
    test_idx.extend(
        np.random.choice(class_indices, size=int(NUM_TEST / NUM_CLASSES), replace=False)
    )
test_idx = np.array(test_idx)
x_test = x_test[test_idx]
y_test = y_test[test_idx].astype(np.uint8)

# Print class counts
print("Training class counts (0-9):", np.bincount(y_train, minlength=10))
print("Test class counts (0-9):", np.bincount(y_test, minlength=10))

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
