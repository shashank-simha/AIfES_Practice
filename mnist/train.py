#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor
import numpy as np

# Configuration
NORMALIZED = True  # Float normalized (mu=0.1307, sigma=0.3081)
NUM_TEST_SUBSET = 20  # Test samples (2 per class)
NUM_CLASSES = 10      # Number of classes (0-9)

# Load full MNIST dataset
transform = Compose([ToTensor()])  # Load as float [0, 1]
test_data = datasets.MNIST(root="data", train=False, transform=transform, download=True)

# Select test subset for mnist_data.h
np.random.seed(42)
test_idx = []
for i in range(NUM_CLASSES):
    class_indices = np.where(test_data.targets == i)[0]
    test_idx.extend(np.random.choice(class_indices, size=NUM_TEST_SUBSET // NUM_CLASSES, replace=False))
test_idx = np.array(test_idx)

# Prepare test subset data
test_images_subset = test_data.data[test_idx].numpy()  # [20, 28, 28]
test_labels_subset = test_data.targets[test_idx].numpy()

# Normalize test images
test_images_subset = (test_images_subset.astype(np.float32) / 255.0 - 0.1307) / 0.3081

# Convert test labels to one-hot float
test_onehot = np.zeros((NUM_TEST_SUBSET, NUM_CLASSES), dtype=np.float32)
for i in range(NUM_TEST_SUBSET):
    test_onehot[i, test_labels_subset[i]] = 1.0

# Print test subset class counts
print("Test subset class counts:", np.bincount(test_labels_subset, minlength=NUM_CLASSES))
print("Input type for ESP32: float (normalized)")

# Custom dataset for full training
class CustomMNIST(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        img = (img.astype(np.float32) / 255.0 - 0.1307) / 0.3081  # Normalize
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # [1, 28, 28]
        label = torch.tensor(self.targets[idx], dtype=torch.long)
        return img, label

# Data loaders for full dataset
loaders = {
    "train": torch.utils.data.DataLoader(
        CustomMNIST(datasets.MNIST(root="data", train=True, transform=transform).data.numpy(),
                    datasets.MNIST(root="data", train=True).targets.numpy()),
        batch_size=100,
        shuffle=True,
        num_workers=1
    ),
    "test": torch.utils.data.DataLoader(
        CustomMNIST(test_data.data.numpy(), test_data.targets.numpy()),
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
        f.write("#ifndef MNIST_WEIGHTS_H\n#define MNIST_WEIGHTS_H\n\n")
        # Conv1 weights [8, 1, 3, 3]
        conv1_w = model.conv1.weight.data.cpu().numpy()
        conv1_w = np.transpose(conv1_w, (0, 1, 2, 3))  # Keep NCHW for AIfES
        f.write("float conv1_weights[8][1][3][3] = {\n")
        for i in range(8):
            f.write("  {\n")  # [1][3][3]
            for j in range(1):
                f.write("    {\n")  # [3][3]
                for k in range(3):
                    row = ",".join([f"{x:.6f}f" for x in conv1_w[i, j, k]])
                    f.write(f"      {{{row}}}")
                    f.write(",\n" if k < 2 else "\n")
                f.write("    }\n")
            f.write("  }" + (",\n" if i < 7 else "\n"))
        f.write("};\n")
        # Conv1 bias [8]
        conv1_b = model.conv1.bias.data.cpu().numpy()
        f.write(f"float conv1_bias[8] = {{{','.join([f'{x:.6f}f' for x in conv1_b])}}};\n")
        # Conv2 weights [16, 8, 3, 3]
        conv2_w = model.conv2.weight.data.cpu().numpy()
        conv2_w = np.transpose(conv2_w, (0, 1, 2, 3))  # Keep NCHW
        f.write("float conv2_weights[16][8][3][3] = {\n")
        for i in range(16):
            f.write("  {\n")  # [8][3][3]
            for j in range(8):
                f.write("    {\n")  # [3][3]
                for k in range(3):
                    row = ",".join([f"{x:.6f}f" for x in conv2_w[i, j, k]])
                    f.write(f"      {{{row}}}")
                    f.write(",\n" if k < 2 else "\n")
                f.write("    }" + (",\n" if j < 7 else "\n"))
            f.write("  }" + (",\n" if i < 15 else "\n"))
        f.write("};\n")
        # Conv2 bias [16]
        conv2_b = model.conv2.bias.data.cpu().numpy()
        f.write(f"float conv2_bias[16] = {{{','.join([f'{x:.6f}f' for x in conv2_b])}}};\n")
        # FC1 weights [784, 64]
        fc1_w = model.fc1.weight.data.cpu().numpy()
        fc1_w = np.transpose(fc1_w, (1, 0))  # [784, 64] for AIfES
        f.write("float fc1_weights[784][64] = {\n")
        for i in range(784):
            row = ",".join([f"{x:.6f}f" for x in fc1_w[i]])
            f.write(f"  {{{row}}}" + (",\n" if i < 783 else "\n"))
        f.write("};\n")
        # FC1 bias [64]
        fc1_b = model.fc1.bias.data.cpu().numpy()
        f.write(f"float fc1_bias[64] = {{{','.join([f'{x:.6f}f' for x in fc1_b])}}};\n")
        # FC2 weights [64, 10]
        fc2_w = model.fc2.weight.data.cpu().numpy()
        fc2_w = np.transpose(fc2_w, (1, 0))  # [64, 10] for AIfES
        f.write("float fc2_weights[64][10] = {\n")
        for i in range(64):
            row = ",".join([f"{x:.6f}f" for x in fc2_w[i]])
            f.write(f"  {{{row}}}" + (",\n" if i < 63 else "\n"))
        f.write("};\n")
        # FC2 bias [10]
        fc2_b = model.fc2.bias.data.cpu().numpy()
        f.write(f"float fc2_bias[10] = {{{','.join([f'{x:.6f}f' for x in fc2_b])}}};\n")
        f.write("#endif")
    print("Weights saved to mnist_weights.h")

# Generate mnist_data.h
def generate_dataset():
    with open("mnist_data.h", "w") as f:
        f.write("#ifndef MNIST_DATA_H\n#define MNIST_DATA_H\n\n")
        # Test input data [20, 1, 28, 28]
        f.write(f"float test_input_data[{NUM_TEST_SUBSET}][1][28][28] = {{\n")
        for i in range(NUM_TEST_SUBSET):
            f.write("  {\n")  # [1][28][28]
            for j in range(1):
                f.write("    {\n")  # [28][28]
                for k in range(28):
                    row = ",".join([f"{x:.6f}f" for x in test_images_subset[i, k]])
                    f.write(f"      {{{row}}}")
                    f.write(",\n" if k < 27 else "\n")
                f.write("    }\n")
            f.write("  }" + (",\n" if i < NUM_TEST_SUBSET - 1 else "\n"))
        f.write("};\n\n")
        # Test target data [20, 10]
        f.write(f"float test_target_data[{NUM_TEST_SUBSET}][10] = {{\n")
        for i in range(NUM_TEST_SUBSET):
            row = ",".join([f"{x:.1f}f" for x in test_onehot[i]])
            f.write(f"  {{{row}}}" + (",\n" if i < NUM_TEST_SUBSET - 1 else "\n"))
        f.write("};\n")
        f.write("#endif")
    print("Dataset saved to mnist_data.h")


# Generate files
generate_weights()
generate_dataset()
