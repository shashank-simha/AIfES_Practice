import numpy as np
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(x_train, y_train), _ = mnist.load_data()

# Select first 100 images and labels
x_train = x_train[:100].astype(np.float32)
y_train = y_train[:100]

# Normalize: mean=0.1307, std=0.3081
x_train = x_train / 255.0  # Scale to [0,1]
x_train = (x_train - 0.1307) / 0.3081  # Standardize
x_train = x_train.reshape(100, 1, 28, 28)  # NCHW format: [100,1,28,28]

# One-hot encode labels
y_train_one_hot = np.eye(10)[y_train].astype(np.float32)  # [100,10]


# Function to format array as C array
def array_to_c(array, name, file):
    file.write(f"const float {name}[{array.shape[0]}][{array.shape[1]}]")
    file.write(
        f"[{array.shape[2]}][{array.shape[3]}] PROGMEM = {{\n"
        if len(array.shape) == 4
        else f" PROGMEM = {{\n"
    )
    for i in range(array.shape[0]):
        file.write("  {" if len(array.shape) == 4 else "  {")
        if len(array.shape) == 4:
            for c in range(array.shape[1]):
                file.write("  {" if c == 0 else ",  {")
                for h in range(array.shape[2]):
                    file.write("  {" if h == 0 else ",  {")
                    for w in range(array.shape[3]):
                        file.write(
                            f"{array[i,c,h,w]:.6f}"
                            + ("," if w < array.shape[3] - 1 else "")
                        )
                    file.write("}" + ("," if h < array.shape[2] - 1 else ""))
                file.write("}" + ("," if c < array.shape[1] - 1 else ""))
        else:
            for j in range(array.shape[1]):
                file.write(
                    f"{array[i,j]:.6f}" + ("," if j < array.shape[1] - 1 else "")
                )
        file.write("}" + ("," if i < array.shape[0] - 1 else "") + "\n")
    file.write("};\n")


# Write C arrays to file
with open("mnist_data.h", "w") as file:
    file.write("#ifndef MNIST_DATA_H\n")
    file.write("#define MNIST_DATA_H\n\n")
    file.write("#include <pgmspace.h>\n\n")
    file.write(
        "// MNIST training data (100 images, NCHW, normalized mean=0.1307, std=0.3081)\n"
    )
    array_to_c(x_train, "train_input_data", file)
    file.write("\n// MNIST one-hot encoded labels (100 labels, 10 classes)\n")
    array_to_c(y_train_one_hot, "train_target_data", file)
    file.write("\n#endif // MNIST_DATA_H\n")
