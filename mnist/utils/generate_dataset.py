#!/usr/bin/env python3
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor

# ------------------ For Dataset generation ------------------
NUM_TRAIN_SUBSET = 200
NUM_TEST_SUBSET = 20
NUM_CLASSES = 10

# ------------------ Load MNIST ------------------
transform = ToTensor()
full_train = datasets.MNIST(root="data", train=True, transform=transform, download=True)
full_test = datasets.MNIST(root="data", train=False, transform=transform, download=True)

# ------------------ Select train subset ------------------
np.random.seed(42)
train_idx = []
for i in range(NUM_CLASSES):
    class_indices = np.where(full_train.targets == i)[0]
    train_idx.extend(
        np.random.choice(class_indices, size=NUM_TRAIN_SUBSET // NUM_CLASSES, replace=False)
    )
train_idx = np.array(train_idx)

train_images_subset = full_train.data[train_idx].numpy()
train_labels_subset = full_train.targets[train_idx].numpy()
print("Train subset class counts:", np.bincount(train_labels_subset, minlength=NUM_CLASSES))

# ------------------ Select test subset ------------------
test_idx = []
for i in range(NUM_CLASSES):
    class_indices = np.where(full_test.targets == i)[0]
    test_idx.extend(
        np.random.choice(class_indices, size=NUM_TEST_SUBSET // NUM_CLASSES, replace=False)
    )
test_idx = np.array(test_idx)

test_images_subset = full_test.data[test_idx].numpy()
test_labels_subset = full_test.targets[test_idx].numpy()
print("Test subset class counts:", np.bincount(test_labels_subset, minlength=NUM_CLASSES))

# ------------------ Generate dataset header ------------------
def generate_dataset():
    with open("mnist_data.h", "w") as f:
        f.write("#ifndef MNIST_DATA_H\n#define MNIST_DATA_H\n\n")

        # Training dataset
        f.write(f"// Training dataset\n")
        f.write(f"const uint8_t train_input_data[{NUM_TRAIN_SUBSET}][1][28][28] PROGMEM = {{\n")
        for i in range(NUM_TRAIN_SUBSET):
            f.write("  {\n")
            for j in range(1):  # channel
                f.write("    {\n")
                for k in range(28):  # row
                    row = ",".join([str(x) for x in train_images_subset[i, k]])
                    f.write(f"      {{{row}}}" + (",\n" if k < 27 else "\n"))
                f.write("    }\n")
            f.write("  }" + (",\n" if i < NUM_TRAIN_SUBSET - 1 else "\n"))
        f.write("};\n\n")

        labels = train_labels_subset.astype(np.uint8)
        f.write(f"const uint8_t train_target_data[{NUM_TRAIN_SUBSET}] PROGMEM = {{\n")
        for i in range(NUM_TRAIN_SUBSET):
            f.write(f"  {labels[i]}" + (",\n" if i < NUM_TRAIN_SUBSET - 1 else "\n"))
        f.write("};\n\n")

        # Test dataset
        f.write(f"// Test dataset\n")
        f.write(f"const uint8_t test_input_data[{NUM_TEST_SUBSET}][1][28][28] PROGMEM = {{\n")
        for i in range(NUM_TEST_SUBSET):
            f.write("  {\n")
            for j in range(1):  # channel
                f.write("    {\n")
                for k in range(28):  # row
                    row = ",".join([str(x) for x in test_images_subset[i, k]])
                    f.write(f"      {{{row}}}" + (",\n" if k < 27 else "\n"))
                f.write("    }\n")
            f.write("  }" + (",\n" if i < NUM_TEST_SUBSET - 1 else "\n"))
        f.write("};\n\n")

        labels = test_labels_subset.astype(np.uint8)
        f.write(f"const uint8_t test_target_data[{NUM_TEST_SUBSET}] PROGMEM = {{\n")
        for i in range(NUM_TEST_SUBSET):
            f.write(f"  {labels[i]}" + (",\n" if i < NUM_TEST_SUBSET - 1 else "\n"))
        f.write("};\n\n")

        f.write("#endif\n")

    print("Dataset saved to mnist_data.h (uint8_t raw)")

if __name__ == "__main__":
    generate_dataset()
