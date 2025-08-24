#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize
import matplotlib.pyplot as plt

# Load MNIST dataset with normalization
transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
train_data = datasets.MNIST(root="data", train=True, transform=transform, download=True)
test_data = datasets.MNIST(root="data", train=False, transform=transform, download=True)

# Print dataset info
print(f"training_data: {train_data}")
print(f"test_data: {test_data}")
print(f"training_data data shape: {train_data.data.shape}")
print(f"training_data targets shape: {train_data.targets.shape}")
print(f"test_data data shape: {test_data.data.shape}")
print(f"test_data targets shape: {test_data.targets.shape}")

# Data loaders
loaders = {
    "train": torch.utils.data.DataLoader(
        train_data, batch_size=100, shuffle=True, num_workers=1
    ),
    "test": torch.utils.data.DataLoader(
        test_data, batch_size=100, shuffle=False, num_workers=1
    ),
}


# CNN to match ESP32 model (8 conv1 filters, 16 conv2 filters, 64 dense neurons, 3x3 kernels, padding)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)  # [8, 28, 28]
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # [8, 14, 14]
        self.conv2 = nn.Conv2d(
            8, 16, kernel_size=3, stride=1, padding=1
        )  # [16, 14, 14]
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # [16, 7, 7]
        self.fc1 = nn.Linear(16 * 7 * 7, 64)  # [64]
        self.fc2 = nn.Linear(64, 10)  # [10]

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = x.view(-1, 16 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


# Initialize model, optimizer, loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()


# Training function
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loaders["train"]):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loaders["train"].dataset)} ({100. * batch_idx / len(loaders["train"]):.0f}%)]\t{loss.item():.6f}'
            )


# Test function
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


# Export weights for AIfES
def export_weights():
    print("Exporting weights for AIfES...")
    with open("mnist_weights.h", "w") as f:
        f.write(
            "#ifndef MNIST_WEIGHTS_H\n#define MNIST_WEIGHTS_H\n#include <pgmspace.h>\n\n"
        )
        # Conv1 weights [8, 1, 3, 3] = 72
        conv1_w = model.conv1.weight.data.cpu().numpy()
        f.write(f"const float conv1_weights[{conv1_w.size}] PROGMEM = {{")
        f.write(",".join([f"{x:.6f}" for x in conv1_w.flatten()]))
        f.write("};\n")
        # Conv1 bias [8]
        conv1_b = model.conv1.bias.data.cpu().numpy()
        f.write(f"const float conv1_bias[{conv1_b.size}] PROGMEM = {{")
        f.write(",".join([f"{x:.6f}" for x in conv1_b]))
        f.write("};\n")
        # Conv2 weights [16, 8, 3, 3] = 1152
        conv2_w = model.conv2.weight.data.cpu().numpy()
        f.write(f"const float conv2_weights[{conv2_w.size}] PROGMEM = {{")
        f.write(",".join([f"{x:.6f}" for x in conv2_w.flatten()]))
        f.write("};\n")
        # Conv2 bias [16]
        conv2_b = model.conv2.bias.data.cpu().numpy()
        f.write(f"const float conv2_bias[{conv2_b.size}] PROGMEM = {{")
        f.write(",".join([f"{x:.6f}" for x in conv2_b]))
        f.write("};\n")
        # FC1 weights [64, 784] = 50176
        fc1_w = model.fc1.weight.data.cpu().numpy()
        f.write(f"const float fc1_weights[{fc1_w.size}] PROGMEM = {{")
        f.write(",".join([f"{x:.6f}" for x in fc1_w.flatten()]))
        f.write("};\n")
        # FC1 bias [64]
        fc1_b = model.fc1.bias.data.cpu().numpy()
        f.write(f"const float fc1_bias[{fc1_b.size}] PROGMEM = {{")
        f.write(",".join([f"{x:.6f}" for x in fc1_b]))
        f.write("};\n")
        # FC2 weights [10, 64] = 640
        fc2_w = model.fc2.weight.data.cpu().numpy()
        f.write(f"const float fc2_weights[{fc2_w.size}] PROGMEM = {{")
        f.write(",".join([f"{x:.6f}" for x in fc2_w.flatten()]))
        f.write("};\n")
        # FC2 bias [10]
        fc2_b = model.fc2.bias.data.cpu().numpy()
        f.write(f"const float fc2_bias[{fc2_b.size}] PROGMEM = {{")
        f.write(",".join([f"{x:.6f}" for x in fc2_b]))
        f.write("};\n")
        f.write("#endif")


# Train for 20 epochs
for epoch in range(1, 21):
    train(epoch)
    test()

# Export weights
export_weights()
print("Weights saved to mnist_weights.h")
