import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re

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
        x_conv1 = self.conv1(x)
        x_relu1 = F.relu(x_conv1)
        x_pool1 = self.pool1(x_relu1)
        x_conv2 = self.conv2(x_pool1)
        x_relu2 = F.relu(x_conv2)
        x_pool2 = self.pool2(x_relu2)
        x_flatten = x_pool2.view(-1, 16 * 7 * 7)
        x_fc1 = self.fc1(x_flatten)
        x_relu3 = F.relu(x_fc1)
        x = self.fc2(x_relu3)
        return x  # Raw logits

# Function to parse C array from header file
def parse_c_array(file_path, array_name, expected_size):
    with open(file_path, 'r') as f:
        content = f.read()
    # Updated regex to handle optional 'const' and 'PROGMEM' before the type
    pattern = rf'(?:const\s+)?float\s+{array_name}\s*(?:\[[^\]]*\]\s*)*\s*(?:PROGMEM\s*)?=\s*\{{(.*?)\}};'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError(f"Array {array_name} not found in {file_path}")

    array_content = match.group(1)
    # Extract numbers, allowing optional 'f' at the end
    numbers = re.findall(r'-?\d+\.\d+(?:[eE][-+]?\d+)?f?', array_content)
    clean_numbers = [float(num.rstrip('f')) for num in numbers]

    if len(clean_numbers) != expected_size:
        raise ValueError(f"Expected {expected_size} values for {array_name}, got {len(clean_numbers)}")
    return clean_numbers

# Load weights from mnist_weights.h
def load_weights(model, file_path='mnist_weights.h'):
    conv1_w = parse_c_array(file_path, 'conv1_weights', 8*1*3*3)
    conv1_w = np.array(conv1_w).reshape(8, 1, 3, 3)
    model.conv1.weight.data = torch.tensor(conv1_w, dtype=torch.float32)

    conv1_b = parse_c_array(file_path, 'conv1_bias', 8)
    model.conv1.bias.data = torch.tensor(conv1_b, dtype=torch.float32)

    conv2_w = parse_c_array(file_path, 'conv2_weights', 16*8*3*3)
    conv2_w = np.array(conv2_w).reshape(16, 8, 3, 3)
    model.conv2.weight.data = torch.tensor(conv2_w, dtype=torch.float32)

    conv2_b = parse_c_array(file_path, 'conv2_bias', 16)
    model.conv2.bias.data = torch.tensor(conv2_b, dtype=torch.float32)

    fc1_w = parse_c_array(file_path, 'fc1_weights', 784*64)
    fc1_w = np.array(fc1_w).reshape(784, 64).T  # Transpose to [64, 784]
    model.fc1.weight.data = torch.tensor(fc1_w, dtype=torch.float32)

    fc1_b = parse_c_array(file_path, 'fc1_bias', 64)
    model.fc1.bias.data = torch.tensor(fc1_b, dtype=torch.float32)

    fc2_w = parse_c_array(file_path, 'fc2_weights', 64*10)
    fc2_w = np.array(fc2_w).reshape(64, 10).T  # Transpose to [10, 64]
    model.fc2.weight.data = torch.tensor(fc2_w, dtype=torch.float32)

    fc2_b = parse_c_array(file_path, 'fc2_bias', 10)
    model.fc2.bias.data = torch.tensor(fc2_b, dtype=torch.float32)

# Load input and target data from mnist_data.h
def load_data(file_path='mnist_data.h'):
    input_data = parse_c_array(file_path, 'test_input_data', 20*1*28*28)
    input_data = np.array(input_data).reshape(20, 1, 28, 28)
    target_data = parse_c_array(file_path, 'test_target_data', 20*10)
    target_data = np.array(target_data).reshape(20, 10)
    return torch.tensor(input_data, dtype=torch.float32), torch.tensor(target_data, dtype=torch.float32)

# Inference with layer output printing and file writing
def inference(model, input_tensor, target_tensor, num_inputs=1, print_layers=[]):
    model.eval()
    correct = 0
    total = input_tensor.shape[0]

    with open('output_python.txt', 'w') as f:
        with torch.no_grad():
            for i in range(total):
                x = input_tensor[i:i+1]  # Process one image [1, 1, 28, 28]
                outputs = {}

                x = model.conv1(x); outputs['conv1'] = x.clone()
                x = F.relu(x);       outputs['relu1'] = x.clone()
                x = model.pool1(x);  outputs['pool1'] = x.clone()

                x = model.conv2(x);  outputs['conv2'] = x.clone()
                x = F.relu(x);       outputs['relu2'] = x.clone()
                x = model.pool2(x);  outputs['pool2'] = x.clone()

                x = x.view(-1, 16 * 7 * 7); outputs['flatten'] = x.clone()
                x = model.fc1(x);          outputs['fc1'] = x.clone()
                x = F.relu(x);             outputs['relu3'] = x.clone()
                x = model.fc2(x);          outputs['fc2'] = x.clone()
                outputs['output'] = x.clone()

                pred_class = outputs['output'].argmax(dim=1).item()
                target_class = target_tensor[i].argmax(dim=0).item()
                matches = pred_class == target_class
                if matches:
                    correct += 1

                # Console output for every input
                print(f"Prediction for input {i}: {pred_class}, Target: {target_class}, Matches: {matches}")

                # File output only for first num_inputs
                if i < num_inputs:
                    for layer_name in print_layers:
                        if layer_name in outputs:
                            tensor = outputs[layer_name]
                            shape_str = "x".join(map(str, tensor.shape))  # e.g., "1x8x14x14"
                            flat_vals = ",".join([f"{val:.6f}" for val in tensor.flatten().numpy()])
                            f.write(f"{layer_name} for input {i} (shape: {shape_str}): [{flat_vals}]\n")
                    f.write(f"Prediction for input {i}: {pred_class}, Target: {target_class}, Matches: {matches}\n\n")

    acc = 100.0 * correct / total
    print(f"Accuracy: {correct}/{total} ({acc:.2f}%)")

# Main
if __name__ == "__main__":
    model = CNN()
    load_weights(model)
    input_tensor, target_tensor = load_data()
    print_layers = ['fc2']  # Specify layers to dump into file
    num_inputs = 1  # Only first N inputs go to file
    inference(model, input_tensor, target_tensor, num_inputs, print_layers)
    print("Layer outputs written to output_python.txt")
