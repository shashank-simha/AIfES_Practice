import torch
import torch.nn as nn
import numpy as np
import re

# Define simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(2, 3), padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

# Function to parse C array from header file
def parse_c_array(file_path, array_name):
    with open(file_path, 'r') as f:
        content = f.read()
    # Find the specific array definition
    pattern = rf'{array_name}\s*\[.*?\]\s*=\s*\{{(.*?)\}};'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError(f"Array {array_name} not found in {file_path}")

    # Extract numbers from the matched array content only
    array_content = match.group(1)
    numbers = re.findall(r'-?\d+\.\d+', array_content)
    return [float(num) for num in numbers]

# Read weights and biases from weights.h
weights = parse_c_array('weights.h', 'conv1_weight')
weights = np.array(weights).reshape(1, 1, 2, 3)
bias = parse_c_array('weights.h', 'conv1_bias')
bias = np.array(bias).reshape(1)

# Read input data from data.h
input_data = parse_c_array('data.h', 'input_data')
input_data = np.array(input_data).reshape(4, 1, 4, 5)

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
    print(f"Conv1 output: {output[0, 0].flatten().numpy().tolist()}")
