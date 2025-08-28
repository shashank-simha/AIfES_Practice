import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re

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
# Parse C array from header file
# -----------------------------
def parse_c_array(file_path, array_name, expected_size, dtype='float'):
    with open(file_path, 'r') as f:
        content = f.read()

    if dtype == 'float':
        # Matches: const float conv1_weights[8][1][3][3] PROGMEM = { ... };
        pattern = rf'(?:const\s+)?float\s+{array_name}\s*(?:\[[^\]]*\])*?\s*(?:PROGMEM\s*)?=\s*\{{(.*?)\}};'
        number_regex = r'-?\d+\.\d+(?:[eE][-+]?\d+)?f?'
    elif dtype == 'uint8':
        # Matches: const uint8_t test_input_data[20][1][28][28] PROGMEM = { ... };
        pattern = rf'(?:const\s+)?uint8_t\s+{array_name}\s*(?:\[[^\]]*\])*?\s*(?:PROGMEM\s*)?=\s*\{{(.*?)\}};'
        number_regex = r'\d+'
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError(f"Array {array_name} not found in {file_path}")

    array_content = match.group(1)
    numbers = re.findall(number_regex, array_content)

    if dtype == 'float':
        values = [float(num.rstrip('f')) for num in numbers]
    else:
        values = [int(num) for num in numbers]

    if len(values) != expected_size:
        raise ValueError(f"Expected {expected_size} values for {array_name}, got {len(values)}")

    return np.array(values, dtype=np.float32 if dtype == 'float' else np.uint8)



# -----------------------------
# Load weights from header
# -----------------------------
def load_weights(model, file_path='mnist_weights.h'):
    conv1_w = np.array(parse_c_array(file_path, 'conv1_weights', 8*1*3*3)).reshape(8,1,3,3)
    model.conv1.weight.data = torch.tensor(conv1_w, dtype=torch.float32)
    conv1_b = parse_c_array(file_path, 'conv1_bias', 8)
    model.conv1.bias.data = torch.tensor(conv1_b, dtype=torch.float32)

    conv2_w = np.array(parse_c_array(file_path, 'conv2_weights', 16*8*3*3)).reshape(16,8,3,3)
    model.conv2.weight.data = torch.tensor(conv2_w, dtype=torch.float32)
    conv2_b = parse_c_array(file_path, 'conv2_bias', 16)
    model.conv2.bias.data = torch.tensor(conv2_b, dtype=torch.float32)

    fc1_w = np.array(parse_c_array(file_path, 'fc1_weights', 784*64)).reshape(784,64).T
    model.fc1.weight.data = torch.tensor(fc1_w, dtype=torch.float32)
    fc1_b = parse_c_array(file_path, 'fc1_bias', 64)
    model.fc1.bias.data = torch.tensor(fc1_b, dtype=torch.float32)

    fc2_w = np.array(parse_c_array(file_path, 'fc2_weights', 64*10)).reshape(64,10).T
    model.fc2.weight.data = torch.tensor(fc2_w, dtype=torch.float32)
    fc2_b = parse_c_array(file_path, 'fc2_bias', 10)
    model.fc2.bias.data = torch.tensor(fc2_b, dtype=torch.float32)


# -----------------------------
# Load input & target data
# -----------------------------
def load_data(file_path='mnist_data.h', dataset_size=None):
    """
    Load input and target data from C header file.
    Input: stored as uint8 -> converted to float and normalized to [0,1]
    Targets: stored as uint8 digits (0–9)
    """
    N = dataset_size if dataset_size is not None else 20  # default fallback

    # Input
    input_data_raw = parse_c_array(file_path, 'test_input_data', N*1*28*28, dtype='uint8')
    input_data = np.array(input_data_raw, dtype=np.float32).reshape(N, 1, 28, 28)
    input_data /= 255.0  # normalize to [0,1]

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
    DATASET_SIZE = None  # None → default 20, or set to custom N

    model = CNN()
    load_weights(model)
    input_tensor, target_tensor = load_data(dataset_size=DATASET_SIZE)

    print_layers = ['fc2']
    num_inputs = 1
    inference(model, input_tensor, target_tensor, num_inputs, print_layers)
    print("Layer outputs written to output_python.txt")
