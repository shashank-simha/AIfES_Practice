#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np

# ------------------ Configuration ------------------
BATCH_SIZE = 100
EPOCHS = 20
NUM_CLASSES = 10
TRAIN_SUBSET = 10000   # Use only 10k samples for training

# ------------------ Load MNIST ------------------
transform = ToTensor()
full_train = datasets.MNIST(root="data", train=True, transform=transform, download=True)
full_test = datasets.MNIST(root="data", train=False, transform=transform, download=True)

# Subset the training data
subset_indices = torch.randperm(len(full_train))[:TRAIN_SUBSET]
train_data = full_train.data[subset_indices].numpy()
train_targets = full_train.targets[subset_indices].numpy()

# ------------------ Custom Dataset ------------------
class CustomMNIST(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].astype(np.float32) / 255.0
        img = (img - 0.1307) / 0.3081
        img = torch.tensor(img).unsqueeze(0)  # [1,28,28]
        label = torch.tensor(self.targets[idx], dtype=torch.long)
        return img, label

train_dataset = CustomMNIST(train_data, train_targets)
test_dataset = CustomMNIST(full_test.data.numpy(), full_test.targets.numpy())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

# ------------------ Lightweight CNN model ------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)   # 1→4
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)   # 4→8
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8*7*7, 32)  # smaller dense
        self.fc2 = nn.Linear(32, NUM_CLASSES)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = x.view(-1, 8*7*7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ------------------ Training ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

# Try SGD (lighter than Adam)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

loss_fn = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")

def test():
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    acc = 100.0 * correct / len(test_loader.dataset)
    print(f"Test Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.2f}%)")

for epoch in range(1, EPOCHS+1):
    train(epoch)
    test()

# ------------------ Save parameters to binary ------------------
def save_params_bin():
    """
    Save all model parameters to a binary file compatible with ESP loading.
    Conv layers are saved as-is (C-order).
    Fully-connected (dense) layers are transposed to match [in_features][out_features] layout.
    """
    with open("params.bin", "wb") as f:
        for name, param in model.named_parameters():
            arr = param.detach().cpu().numpy().astype(np.float32)

            # Identify FC layers by shape heuristics: 2D and not 4D
            if arr.ndim == 2:
                arr_to_save = arr.T  # transpose for ESP [in][out]
            else:
                arr_to_save = arr  # conv weights keep original shape

            arr_to_save.tofile(f)
            print(f"Saved {name}, shape={arr_to_save.shape}, bytes={arr_to_save.nbytes}")

    print(f"All weights saved to params.bin")

if __name__ == "__main__":
    save_params_bin()
