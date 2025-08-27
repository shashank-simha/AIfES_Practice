import torch
import torch.nn as nn
import numpy as np
import re

# Define simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=(3, 4), padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

# Function to parse C array from header file
def parse_c_array(file_path, array_name):
    with open(file_path, 'r') as f:
        content = f.read()
    pattern = rf'{array_name}\s*\[.*?\]\s*=\s*\{{(.*?)\}};'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError(f"Array {array_name} not found in {file_path}")

    array_content = match.group(1)
    numbers = re.findall(r'-?\d+\.\d+', array_content)
    return [float(num) for num in numbers]

# Read weights and biases from weights.h
weights = parse_c_array('weights.h', 'conv1_weight')
weights = np.array(weights).reshape(2, 1, 3, 4)  # 2 filters, 1 input channel, 3x4 kernel
bias = parse_c_array('weights.h', 'conv1_bias')
bias = np.array(bias).reshape(2)

# Read input data from data.h
input_data = parse_c_array('data.h', 'input_data')
input_data = np.array(input_data).reshape(4, 1, 6, 8)  # 4 inputs, 1 channel, 6x8

# Initialize model
model = SimpleCNN()

# Set weights and biases
with torch.no_grad():
    model.conv1.weight.copy_(torch.tensor(weights, dtype=torch.float32))
    model.conv1.bias.copy_(torch.tensor(bias, dtype=torch.float32))

# Inference for each input
model.eval()
for i in range(input_data.shape[0]):
    input_tensor = torch.tensor(input_data[i:i+1], dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)

    # Print input and output tensors
    print(f"\nInput {i} shape: {input_tensor.shape}")
    print(f"Input {i}: {input_tensor.flatten().numpy().tolist()}")
    print(f"Conv1 output shape: {output.shape}")
    for j in range(output.size(1)):
        print(f"Conv1 output channel {j} for input {i}: {output[0, j].flatten().numpy().tolist()}")
