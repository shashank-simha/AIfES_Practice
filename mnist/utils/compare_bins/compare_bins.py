import numpy as np

def read_params(path):
    with open(path, "rb") as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.float32)
    return arr

# Layer sizes
conv1_w_size = 8 * 1 * 3 * 3   # 72
conv1_b_size = 8
conv2_w_size = 16 * 8 * 3 * 3  # 1152
conv2_b_size = 16
fc1_w_size   = 16 * 7 * 7 * 64 # 50176
fc1_b_size   = 64
fc2_w_size   = 64 * 10         # 640
fc2_b_size   = 10

# Offsets (same order AIfES allocates in memory)
offsets = {
    "conv1_w": (0, conv1_w_size),
    "conv1_b": (conv1_w_size, conv1_w_size + conv1_b_size),
    "conv2_w": (conv1_w_size + conv1_b_size, conv1_w_size + conv1_b_size + conv2_w_size),
    "conv2_b": (conv1_w_size + conv1_b_size + conv2_w_size, conv1_w_size + conv1_b_size + conv2_w_size + conv2_b_size),
    "fc1_w":   (conv1_w_size + conv1_b_size + conv2_w_size + conv2_b_size,
                conv1_w_size + conv1_b_size + conv2_w_size + conv2_b_size + fc1_w_size),
    "fc1_b":   (conv1_w_size + conv1_b_size + conv2_w_size + conv2_b_size + fc1_w_size,
                conv1_w_size + conv1_b_size + conv2_w_size + conv2_b_size + fc1_w_size + fc1_b_size),
    "fc2_w":   (conv1_w_size + conv1_b_size + conv2_w_size + conv2_b_size + fc1_w_size + fc1_b_size,
                conv1_w_size + conv1_b_size + conv2_w_size + conv2_b_size + fc1_w_size + fc1_b_size + fc2_w_size),
    "fc2_b":   (conv1_w_size + conv1_b_size + conv2_w_size + conv2_b_size + fc1_w_size + fc1_b_size + fc2_w_size,
                conv1_w_size + conv1_b_size + conv2_w_size + conv2_b_size + fc1_w_size + fc1_b_size + fc2_w_size + fc2_b_size),
}

def extract_layer(arr, name):
    start, end = offsets[name]
    return arr[start:end]

if __name__ == "__main__":
    esp_params   = read_params("esp_params.bin")
    train_params = read_params("train_params.bin")

    conv1_esp_w, conv1_train_w = extract_layer(esp_params, "conv1_w").reshape(8,1,3,3), extract_layer(train_params, "conv1_w").reshape(8,1,3,3)
    conv1_esp_b, conv1_train_b = extract_layer(esp_params, "conv1_b"), extract_layer(train_params, "conv1_b")

    np.set_printoptions(precision=2, suppress=True)

    print("=== Conv1 Weights ESP ===")
    print(conv1_esp_w)
    print("=== Conv1 Weights Train ===")
    print(conv1_train_w)

    print("=== Conv1 Bias ESP ===")
    print(conv1_esp_b)
    print("=== Conv1 Bias Train ===")
    print(conv1_train_b)
