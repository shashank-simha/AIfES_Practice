#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np

# Configuration
NORMALIZED = True  # False: uint8_t [0, 255], True: float normalized (mu=0.1307, sigma=0.3081)
NUM_TRAIN_SUBSET = 200  # Training samples for ESP32 (20 per class)
NUM_TEST_SUBSET = 20    # Test samples for ESP32 (2 per class)
NUM_CLASSES = 10        # Number of classes (0-9)

# Load full MNIST dataset
transform = Compose([ToTensor()])  # Load as float [0, 1]
train_data = datasets.MNIST(root="data", train=True, transform=transform, download=True)
test_data = datasets.MNIST(root="data", train=False, transform=transform, download=True)

# Select subset for mnist_data.h
np.random.seed(42)
train_idx = []
for i in range(NUM_CLASSES):
    class_indices = np.where(train_data.targets == i)[0]
    train_idx.extend(np.random.choice(class_indices, size=NUM_TRAIN_SUBSET // NUM_CLASSES, replace=False))
train_idx = np.array(train_idx)
test_idx = []
for i in range(NUM_CLASSES):
    class_indices = np.where(test_data.targets == i)[0]
    test_idx.extend(np.random.choice(class_indices, size=NUM_TEST_SUBSET // NUM_CLASSES, replace=False))
test_idx = np.array(test_idx)

# Prepare subset data for mnist_data.h
train_images_subset = train_data.data[train_idx].numpy()  # [200, 28, 28]
train_labels_subset = train_data.targets[train_idx].numpy()
test_images_subset = test_data.data[test_idx].numpy()     # [20, 28, 28]
test_labels_subset = test_data.targets[test_idx].numpy()

# Convert subset images
if NORMALIZED:
    train_images_subset = (train_images_subset.astype(np.float32) / 255.0 - 0.1307) / 0.3081
    test_images_subset = (test_images_subset.astype(np.float32) / 255.0 - 0.1307) / 0.3081
else:
    train_images_subset = train_images_subset.astype(np.uint8)
    test_images_subset = test_images_subset.astype(np.uint8)

# Convert subset labels to one-hot float
train_onehot = np.zeros((NUM_TRAIN_SUBSET, NUM_CLASSES), dtype=np.float32)
test_onehot = np.zeros((NUM_TEST_SUBSET, NUM_CLASSES), dtype=np.float32)
for i in range(NUM_TRAIN_SUBSET):
    train_onehot[i, train_labels_subset[i]] = 1.0
for i in range(NUM_TEST_SUBSET):
    test_onehot[i, test_labels_subset[i]] = 1.0

# Print subset class counts
print("Training subset class counts:", np.bincount(train_labels_subset, minlength=NUM_CLASSES))
print("Test subset class counts:", np.bincount(test_labels_subset, minlength=NUM_CLASSES))
print(f"Input type for ESP32: {'float (normalized)' if NORMALIZED else 'uint8_t [0, 255]'}")


# Custom dataset for full training
class CustomMNIST(torch.utils.data.Dataset):
    def __init__(self, data, targets, normalized=NORMALIZED):
        self.data = data
        self.targets = targets
        self.normalized = normalized

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        if self.normalized:
            img = (img.astype(np.float32) / 255.0 - 0.1307) / 0.3081
        else:
            img = img.astype(np.float32)  # Convert uint8 [0, 255] to float [0, 255]
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # [1, 28, 28]
        label = torch.tensor(self.targets[idx], dtype=torch.long)
        return img, label

# Data loaders for full dataset
loaders = {
    "train": torch.utils.data.DataLoader(
        CustomMNIST(train_data.data.numpy(), train_data.targets.numpy(), normalized=NORMALIZED),
        batch_size=100,
        shuffle=True,
        num_workers=1
    ),
    "test": torch.utils.data.DataLoader(
        CustomMNIST(test_data.data.numpy(), test_data.targets.numpy(), normalized=NORMALIZED),
        batch_size=100,
        shuffle=False,
        num_workers=1
    ),
}

# CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = x.view(-1, 16 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # Raw logits

# Train model on full dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loaders["train"]):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loaders["train"].dataset)} ({100. * batch_idx / len(loaders["train"]):.0f}%)]\t{loss.item():.6f}'
            )

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loaders["test"]:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(loaders["test"].dataset)
    print(
        f'Test set: Average loss: {test_loss:.4f}, Accuracy {correct}/{len(loaders["test"].dataset)} ({100. * correct / len(loaders["test"].dataset):.0f}%)\n'
    )

# Train for 20 epochs
for epoch in range(1, 21):
    train(epoch)
    test()

# Generate mnist_weights.h
def generate_weights():
    with open("mnist_weights.h", "w") as f:
        f.write("#ifndef MNIST_WEIGHTS_H\n#define MNIST_WEIGHTS_H\n#include <pgmspace.h>\n\n")
        # Conv1 weights [1, 8, 3, 3]
        conv1_w = model.conv1.weight.data.cpu().numpy()
        conv1_w = np.transpose(conv1_w, (1, 0, 2, 3))
        f.write(f"const float conv1_weights[{conv1_w.size}] PROGMEM = {{")
        f.write(",".join([f"{x:.6f}f" for x in conv1_w.flatten()]))
        f.write("};\n")
        # Conv1 bias [8]
        conv1_b = model.conv1.bias.data.cpu().numpy()
        f.write(f"const float conv1_bias[{conv1_b.size}] PROGMEM = {{")
        f.write(",".join([f"{x:.6f}f" for x in conv1_b]))
        f.write("};\n")
        # Conv2 weights [8, 16, 3, 3]
        conv2_w = model.conv2.weight.data.cpu().numpy()
        conv2_w = np.transpose(conv2_w, (1, 0, 2, 3))
        f.write(f"const float conv2_weights[{conv2_w.size}] PROGMEM = {{")
        f.write(",".join([f"{x:.6f}f" for x in conv2_w.flatten()]))
        f.write("};\n")
        # Conv2 bias [16]
        conv2_b = model.conv2.bias.data.cpu().numpy()
        f.write(f"const float conv2_bias[{conv2_b.size}] PROGMEM = {{")
        f.write(",".join([f"{x:.6f}f" for x in conv2_b]))
        f.write("};\n")
        # FC1 weights [784, 64]
        fc1_w = model.fc1.weight.data.cpu().numpy()
        fc1_w = np.transpose(fc1_w, (1, 0))
        f.write(f"const float fc1_weights[{fc1_w.size}] PROGMEM = {{")
        f.write(",".join([f"{x:.6f}f" for x in fc1_w.flatten()]))
        f.write("};\n")
        # FC1 bias [64]
        fc1_b = model.fc1.bias.data.cpu().numpy()
        f.write(f"const float fc1_bias[{fc1_b.size}] PROGMEM = {{")
        f.write(",".join([f"{x:.6f}f" for x in fc1_b]))
        f.write("};\n")
        # FC2 weights [64, 10]
        fc2_w = model.fc2.weight.data.cpu().numpy()
        fc2_w = np.transpose(fc2_w, (1, 0))
        f.write(f"const float fc2_weights[{fc2_w.size}] PROGMEM = {{")
        f.write(",".join([f"{x:.6f}f" for x in fc2_w.flatten()]))
        f.write("};\n")
        # FC2 bias [10]
        fc2_b = model.fc2.bias.data.cpu().numpy()
        f.write(f"const float fc2_bias[{fc2_b.size}] PROGMEM = {{")
        f.write(",".join([f"{x:.6f}f" for x in fc2_b]))
        f.write("};\n")
        f.write("#endif")
    print("Weights saved to mnist_weights.h")

# Generate mnist_data.h
def generate_dataset():
    with open("mnist_data.h", "w") as f:
        f.write("#ifndef MNIST_DATA_H\n#define MNIST_DATA_H\n#include <pgmspace.h>\n\n")

        # Training input data [200, 1, 28, 28]
        data_type = "float" if NORMALIZED else "uint8_t"
        f.write(f"const {data_type} train_input_data[{NUM_TRAIN_SUBSET}][1][28][28] PROGMEM = {{")
        for i in range(NUM_TRAIN_SUBSET):
            f.write("{")
            f.write("{")
            for row in range(28):
                f.write("{")
                if NORMALIZED:
                    f.write(",".join([f"{x:.6f}f" for x in train_images_subset[i][row]]))
                else:
                    f.write(",".join(map(str, train_images_subset[i][row])))
                f.write("}" if row == 27 else "},")
            f.write("}")
            f.write("}" if i == NUM_TRAIN_SUBSET - 1 else "},")
        f.write("};\n\n")

        # Training target data [200, 10] (float one-hot)
        f.write(f"const float train_target_data[{NUM_TRAIN_SUBSET}][10] PROGMEM = {{")
        for i in range(NUM_TRAIN_SUBSET):
            f.write("{")
            f.write(",".join([f"{x:.1f}f" for x in train_onehot[i]]))
            f.write("}" if i == NUM_TRAIN_SUBSET - 1 else "},")
        f.write("};\n\n")

        # Test input data [20, 1, 28, 28]
        f.write(f"const {data_type} test_input_data[{NUM_TEST_SUBSET}][1][28][28] PROGMEM = {{")
        for i in range(NUM_TEST_SUBSET):
            f.write("{")
            f.write("{")
            for row in range(28):
                f.write("{")
                if NORMALIZED:
                    f.write(",".join([f"{x:.6f}f" for x in test_images_subset[i][row]]))
                else:
                    f.write(",".join(map(str, test_images_subset[i][row])))
                f.write("}" if row == 27 else "},")
            f.write("}")
            f.write("}" if i == NUM_TEST_SUBSET - 1 else "},")
        f.write("};\n\n")

        # Test target data [20, 10] (float one-hot)
        f.write(f"const float test_target_data[{NUM_TEST_SUBSET}][10] PROGMEM = {{")
        for i in range(NUM_TEST_SUBSET):
            f.write("{")
            f.write(",".join([f"{x:.1f}f" for x in test_onehot[i]]))
            f.write("}" if i == NUM_TEST_SUBSET - 1 else "},")
        f.write("};\n")
        f.write("#endif")
    print("Dataset saved to mnist_data.h")

# Generate files
generate_weights()
generate_dataset()
