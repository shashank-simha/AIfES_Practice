#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np
import re

# Configuration
NORMALIZED = True  # True: normalized input, False: float [0, 255]
NUM_TEST_SUBSET = 20  # Test samples in mnist_data.h
NUM_CLASSES = 10      # Number of classes (0-9)

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

# Load weights from mnist_weights.h
def load_weights(model, weights_file='mnist_weights.h'):
    with open(weights_file, 'r') as f:
        content = f.read()

    # Parse float arrays from C header
    def parse_array(name, expected_size):
        pattern = rf"const float {name}\[\d+\] PROGMEM = {{(.*?)}};"
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            raise ValueError(f"Array {name} not found in {weights_file}")
        values = [float(v.strip('f')) for v in match.group(1).split(',') if v.strip()]
        if len(values) != expected_size:
            raise ValueError(f"Expected {expected_size} values for {name}, got {len(values)}")
        return np.array(values)

    # Load and reshape weights with float32
    model.conv1.weight.data = torch.tensor(
        parse_array('conv1_weights', 1 * 8 * 3 * 3).reshape(1, 8, 3, 3),
        dtype=torch.float32
    ).permute(1, 0, 2, 3).to(device)
    model.conv1.bias.data = torch.tensor(
        parse_array('conv1_bias', 8),
        dtype=torch.float32
    ).to(device)

    model.conv2.weight.data = torch.tensor(
        parse_array('conv2_weights', 8 * 16 * 3 * 3).reshape(8, 16, 3, 3),
        dtype=torch.float32
    ).permute(1, 0, 2, 3).to(device)
    model.conv2.bias.data = torch.tensor(
        parse_array('conv2_bias', 16),
        dtype=torch.float32
    ).to(device)

    model.fc1.weight.data = torch.tensor(
        parse_array('fc1_weights', 784 * 64).reshape(784, 64),
        dtype=torch.float32
    ).permute(1, 0).to(device)
    model.fc1.bias.data = torch.tensor(
        parse_array('fc1_bias', 64),
        dtype=torch.float32
    ).to(device)

    model.fc2.weight.data = torch.tensor(
        parse_array('fc2_weights', 64 * 10).reshape(64, 10),
        dtype=torch.float32
    ).permute(1, 0).to(device)
    model.fc2.bias.data = torch.tensor(
        parse_array('fc2_bias', 10),
        dtype=torch.float32
    ).to(device)

    # Print sample weights for verification
    print("First 10 conv1 weights:", model.conv1.weight.data.flatten()[:10].cpu().numpy())
    print("First 5 fc2 weights:", model.fc2.weight.data.flatten()[:5].cpu().numpy())

# Load test inputs from mnist_data.h
def load_mnist_data(data_file='mnist_data.h'):
    with open(data_file, 'r') as f:
        content = f.read()

    # Parse test_input_data array
    pattern = rf"const (float|uint8_t) test_input_data\[\d+\]\[\d+\]\[\d+\]\[\d+\] PROGMEM = {{(.*?)}};"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError("test_input_data not found in mnist_data.h")

    data_type, data_str = match.groups()
    # Extract values between innermost braces
    values = []
    for image_str in data_str.strip('{}').split('},{'):
        for channel_str in image_str.strip('{}').split('},{'):
            for row_str in channel_str.strip('{}').split('},'):
                row_values = [float(v.strip('f')) for v in row_str.strip('{}').split(',') if v.strip()]
                values.extend(row_values)

    expected_size = NUM_TEST_SUBSET * 1 * 28 * 28
    if len(values) != expected_size:
        raise ValueError(f"Expected {expected_size} values for test_input_data, got {len(values)}")
    inputs = np.array(values).reshape(NUM_TEST_SUBSET, 1, 28, 28)

    # Parse test_target_data (one-hot)
    pattern = rf"const float test_target_data\[\d+\]\[\d+\] PROGMEM = {{(.*?)}};"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError("test_target_data not found in mnist_data.h")
    values = []
    for row_str in match.group(1).strip('{}').split('},'):
        row_values = [float(v.strip('f')) for v in row_str.strip('{}').split(',') if v.strip()]
        values.extend(row_values)

    expected_size = NUM_TEST_SUBSET * NUM_CLASSES
    if len(values) != expected_size:
        raise ValueError(f"Expected {expected_size} values for test_target_data, got {len(values)}")
    targets = np.array(values).reshape(NUM_TEST_SUBSET, NUM_CLASSES)
    labels = np.argmax(targets, axis=1)

    # Convert inputs based on data type
    if data_type == 'uint8_t':
        inputs = inputs.astype(np.float32)  # Convert uint8 [0, 255] to float [0, 255]
    # Else, inputs are float (normalized for NORMALIZED=True)

    return inputs, labels

# Inference on full MNIST test set
def inference_full(model, normalized=NORMALIZED):
    def to_float_tensor(img):
        # Convert PIL Image to NumPy array
        img = np.array(img, dtype=np.float32)  # uint8 [0, 255] to float [0, 255]
        if not normalized:
            return torch.from_numpy(img).unsqueeze(0)  # [1, 28, 28]
        else:
            img = img / 255.0
            img = (img - 0.1307) / 0.3081
            return torch.from_numpy(img).unsqueeze(0)  # [1, 28, 28]

    transform = Compose([to_float_tensor])
    test_data = datasets.MNIST(root="data", train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False, num_workers=1)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    accuracy = 100. * correct / total
    print(f"Full test set accuracy (normalized={normalized}): {accuracy:.2f}% ({correct}/{total})")

# Inference on mnist_data.h subset
def inference_subset(model, data_file='mnist_data.h', normalized=NORMALIZED):
    inputs, true_labels = load_mnist_data(data_file)
    inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
    # For NORMALIZED=True, inputs are already normalized; for False, inputs are float [0, 255]
    model.eval()
    correct = 0
    with torch.no_grad():
        output = model(inputs)
        probs = F.softmax(output, dim=1)
        pred = output.argmax(dim=1)
        for i in range(NUM_TEST_SUBSET):
            print(f"Image {i} logits:", output[i].cpu().numpy())
            print(f"Image {i} probabilities:", probs[i].cpu().numpy())
            print(f"Image {i} predicted: {pred[i].item()}, true: {true_labels[i]}")
            if pred[i].item() == true_labels[i]:
                correct += 1
    accuracy = 100. * correct / NUM_TEST_SUBSET
    print(f"Subset test accuracy (normalized={normalized}): {accuracy:.2f}% ({correct}/{NUM_TEST_SUBSET})")

# Main
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
load_weights(model)
inference_full(model, normalized=NORMALIZED)
inference_subset(model, normalized=NORMALIZED)
