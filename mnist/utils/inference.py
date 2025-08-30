import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
import os

# -----------------------------
# CNN model
# -----------------------------
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
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # raw logits


# -----------------------------
# Load weights from params.bin
# -----------------------------
def load_weights_from_bin(model, file_path="params.bin"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")

    with open(file_path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.float32)

    # Expected tensor shapes (in AIfES save order!)
    shapes = [
        (8, 1, 3, 3),   # conv1.weight
        (8,),           # conv1.bias
        (16, 8, 3, 3),  # conv2.weight
        (16,),          # conv2.bias
        (64, 784),      # fc1.weight (PyTorch expects [64,784])
        (64,),          # fc1.bias
        (10, 64),       # fc2.weight (PyTorch expects [10,64])
        (10,)           # fc2.bias
    ]

    idx = 0
    tensors = []
    for shape in shapes:
        size = np.prod(shape)
        arr = data[idx:idx+size].reshape(shape)
        tensors.append(arr)
        idx += size

    # Assign tensors into model
    model.conv1.weight.data = torch.tensor(tensors[0], dtype=torch.float32)
    model.conv1.bias.data   = torch.tensor(tensors[1], dtype=torch.float32)

    model.conv2.weight.data = torch.tensor(tensors[2], dtype=torch.float32)
    model.conv2.bias.data   = torch.tensor(tensors[3], dtype=torch.float32)

    # Linear layers: confirm orientation (AIfES might store transposed!)
    model.fc1.weight.data   = torch.tensor(tensors[4], dtype=torch.float32)
    model.fc1.bias.data     = torch.tensor(tensors[5], dtype=torch.float32)

    model.fc2.weight.data   = torch.tensor(tensors[6], dtype=torch.float32)
    model.fc2.bias.data     = torch.tensor(tensors[7], dtype=torch.float32)

    print(f"Loaded weights from {file_path}")
    print("conv1.weight[0,0]:", model.conv1.weight.data[0,0])


# -----------------------------
# Parse input & target data
# -----------------------------
def parse_c_array(file_path, array_name, expected_size, dtype='float'):
    with open(file_path, 'r') as f:
        content = f.read()

    if dtype == 'uint8':
        pattern = rf'(?:const\s+)?uint8_t\s+{array_name}\s*(?:\[[^\]]*\])*?\s*(?:PROGMEM\s*)?=\s*\{{(.*?)\}};'
        number_regex = r'\d+'
    else:
        raise ValueError("Only uint8 supported here (for data).")

    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError(f"Array {array_name} not found in {file_path}")

    numbers = re.findall(number_regex, match.group(1))
    values = [int(num) for num in numbers]

    if len(values) != expected_size:
        raise ValueError(f"Expected {expected_size}, got {len(values)}")

    return np.array(values, dtype=np.uint8)


def load_data(file_path='mnist_data.h', dataset_size=None):
    N = dataset_size if dataset_size is not None else 20

    # Input
    input_data_raw = parse_c_array(file_path, 'test_input_data', N*1*28*28, dtype='uint8')
    input_data = np.array(input_data_raw, dtype=np.float32).reshape(N, 1, 28, 28)
    input_data /= 255.0  # normalize

    # Targets
    target_data_raw = parse_c_array(file_path, 'test_target_data', N, dtype='uint8')
    target_data = np.array(target_data_raw, dtype=np.int64)

    return torch.tensor(input_data, dtype=torch.float32), torch.tensor(target_data, dtype=torch.long)


# -----------------------------
# Inference
# -----------------------------
def inference(model, input_tensor, target_tensor, num_inputs=1, print_layers=[]):
    model.eval()
    correct = 0
    total = input_tensor.shape[0]

    with open('output_python.txt', 'w') as f:
        with torch.no_grad():
            for i in range(total):
                x = input_tensor[i:i+1]
                outputs = {}

                x = model.conv1(x); outputs['conv1'] = x.clone()
                x = F.relu(x);       outputs['relu1'] = x.clone()
                x = model.pool1(x);  outputs['pool1'] = x.clone()

                x = model.conv2(x);  outputs['conv2'] = x.clone()
                x = F.relu(x);       outputs['relu2'] = x.clone()
                x = model.pool2(x);  outputs['pool2'] = x.clone()

                x = x.view(-1, 16*7*7); outputs['flatten'] = x.clone()
                x = model.fc1(x);          outputs['fc1'] = x.clone()
                x = F.relu(x);             outputs['relu3'] = x.clone()
                x = model.fc2(x);          outputs['fc2'] = x.clone()
                outputs['output'] = x.clone()

                pred_class = outputs['output'].argmax(dim=1).item()
                target_class = target_tensor[i].item()
                matches = pred_class == target_class
                if matches:
                    correct += 1

                print(f"Prediction {i}: {pred_class}, Target: {target_class}, Match: {matches}")

                if i < num_inputs:
                    for layer_name in print_layers:
                        if layer_name in outputs:
                            tensor = outputs[layer_name]
                            shape_str = "x".join(map(str, tensor.shape))
                            flat_vals = ",".join([f"{val:.6f}" for val in tensor.flatten().numpy()])
                            f.write(f"{layer_name} for input {i} (shape: {shape_str}): [{flat_vals}]\n")
                    f.write(f"Prediction {i}: {pred_class}, Target: {target_class}, Match: {matches}\n\n")

    acc = 100.0 * correct / total
    print(f"Accuracy: {correct}/{total} ({acc:.2f}%)")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    DATASET_SIZE = None  # None → default 20

    model = CNN()
    load_weights_from_bin(model, "params.bin")
    input_tensor, target_tensor = load_data(dataset_size=DATASET_SIZE)

    print_layers = ['fc2']
    num_inputs = 1
    inference(model, input_tensor, target_tensor, num_inputs, print_layers)
    print("Layer outputs written to output_python.txt")
