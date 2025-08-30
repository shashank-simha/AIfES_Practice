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

# ------------------ Load MNIST ------------------
transform = ToTensor()
full_train = datasets.MNIST(root="data", train=True, transform=transform, download=True)
full_test = datasets.MNIST(root="data", train=False, transform=transform, download=True)

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

train_dataset = CustomMNIST(full_train.data.numpy(), full_train.targets.numpy())
test_dataset = CustomMNIST(full_test.data.numpy(), full_test.targets.numpy())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

# ------------------ CNN model ------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*7*7, 64)
        self.fc2 = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = x.view(-1, 16*7*7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ------------------ Training ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
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

# ------------------ Generate weights header ------------------
def generate_weights():
    with open("mnist_weights.h", "w") as f:
        f.write("#ifndef MNIST_WEIGHTS_H\n#define MNIST_WEIGHTS_H\n\n")
        # Conv1 weights [8,1,3,3]
        conv1_w = model.conv1.weight.data.cpu().numpy()
        f.write("const float conv1_weights[8][1][3][3] PROGMEM = {\n")
        for i in range(8):
            f.write("  {\n")
            for j in range(1):
                f.write("    {\n")
                for k in range(3):
                    row = ",".join([f"{x:.6f}f" for x in conv1_w[i,j,k]])
                    f.write(f"      {{{row}}}" + (",\n" if k<2 else "\n"))
                f.write("    }\n")
            f.write("  }" + (",\n" if i<7 else "\n"))
        f.write("};\n")
        # Conv1 bias
        conv1_b = model.conv1.bias.data.cpu().numpy()
        f.write(f"const float conv1_bias[8] PROGMEM = {{{','.join([f'{x:.6f}f' for x in conv1_b])}}};\n")
        # Conv2 weights [16,8,3,3]
        conv2_w = model.conv2.weight.data.cpu().numpy()
        f.write("const float conv2_weights[16][8][3][3] PROGMEM = {\n")
        for i in range(16):
            f.write("  {\n")
            for j in range(8):
                f.write("    {\n")
                for k in range(3):
                    row = ",".join([f"{x:.6f}f" for x in conv2_w[i,j,k]])
                    f.write(f"      {{{row}}}" + (",\n" if k<2 else "\n"))
                f.write("    }" + (",\n" if j<7 else "\n"))
            f.write("  }" + (",\n" if i<15 else "\n"))
        f.write("};\n")
        # Conv2 bias
        conv2_b = model.conv2.bias.data.cpu().numpy()
        f.write(f"const float conv2_bias[16] PROGMEM = {{{','.join([f'{x:.6f}f' for x in conv2_b])}}};\n")
        # FC1 weights [784,64]
        fc1_w = model.fc1.weight.data.cpu().numpy().T
        f.write("const float fc1_weights[784][64] PROGMEM = {\n")
        for i in range(784):
            row = ",".join([f"{x:.6f}f" for x in fc1_w[i]])
            f.write(f"  {{{row}}}" + (",\n" if i<783 else "\n"))
        f.write("};\n")
        # FC1 bias
        fc1_b = model.fc1.bias.data.cpu().numpy()
        f.write(f"const float fc1_bias[64] PROGMEM = {{{','.join([f'{x:.6f}f' for x in fc1_b])}}};\n")
        # FC2 weights [64,10]
        fc2_w = model.fc2.weight.data.cpu().numpy().T
        f.write("const float fc2_weights[64][10] PROGMEM = {\n")
        for i in range(64):
            row = ",".join([f"{x:.6f}f" for x in fc2_w[i]])
            f.write(f"  {{{row}}}" + (",\n" if i<63 else "\n"))
        f.write("};\n")
        # FC2 bias
        fc2_b = model.fc2.bias.data.cpu().numpy()
        f.write(f"const float fc2_bias[10] PROGMEM = {{{','.join([f'{x:.6f}f' for x in fc2_b])}}};\n")
        f.write("#endif\n")
    print("Weights saved to mnist_weights.h")

if __name__ == "__main__":
    save_params_bin()
    generate_weights()
