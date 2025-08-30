import numpy as np
import os

# -----------------------------
# Load Python-trained params
# -----------------------------
def load_python_params(file_path):
    """Load all model parameters from single Python bin."""
    return np.fromfile(file_path, dtype=np.float32)

# -----------------------------
# Load ESP layer bin safely
# -----------------------------
def load_esp_layer(file_path, shape):
    """Load ESP layer from file; return None if file empty or missing."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        print(f"Warning: {file_path} missing or empty, skipping")
        return None
    data = np.fromfile(file_path, dtype=np.float32)
    expected_size = np.prod(shape)
    if data.size != expected_size:
        print(f"Warning: {file_path} size mismatch (expected {expected_size}, got {data.size})")
        return None
    return data.reshape(shape)

# -----------------------------
# Compare layers
# -----------------------------
def compare_layers(layer_name, esp_array, py_array):
    if esp_array is None:
        print(f"Skipping {layer_name} (no ESP data)")
        return
    diff = esp_array - py_array
    print(f"=== {layer_name} Comparison ===")
    print("ESP array (2 digits):")
    np.set_printoptions(precision=2, suppress=True)
    print(esp_array)
    print("Python array (2 digits):")
    print(py_array)
    print("Difference (ESP - Python, 2 digits):")
    print(diff)
    print("\n")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Path to Python-trained params bin
    py_params_file = "train_params.bin"
    py_params = load_python_params(py_params_file)

    # Layer shapes
    layer_shapes = {
        "l1_w": (8, 1, 3, 3),
        "l1_b": (8,),
        "l4_w": (16, 8, 3, 3),
        "l4_b": (16,),
        "l8_w": (64, 16, 7, 7),
        "l8_b": (64,),
        "l10_w": (10, 64),  # FC2 weights
        "l10_b": (10,),      # FC2 bias
    }

    # Load ESP layer files
    esp_layers = {}
    for name, shape in layer_shapes.items():
        esp_layers[name] = load_esp_layer(f"{name}.bin", shape)

    # Extract corresponding Python slices
    py_index = 0
    py_layers = {}
    for name, shape in layer_shapes.items():
        size = np.prod(shape)
        py_slice = py_params[py_index: py_index + size]
        py_layers[name] = py_slice.reshape(shape)
        py_index += size

    # Compare all layers
    # for name in layer_shapes.keys():
    #     compare_layers(name, esp_layers.get(name), py_layers[name])

    layer_name = "l1_w"  # the layer you want to check

    conv1_esp = esp_layers[layer_name]
    conv1_train = py_layers[layer_name]

    orderings = {
        "O/I/H/W": (0,1,2,3),  # current Python shape
        "I/O/H/W": (1,0,2,3),
        "H/W/O/I": (2,3,0,1),
        "H/W/I/O": (2,3,1,0),
        "O/H/W/I": (0,2,3,1),
    }

    for name, axes in orderings.items():
        rearranged = np.transpose(conv1_train, axes).flatten()
        esp_flat = conv1_esp.flatten()
        diff = np.abs(rearranged - esp_flat)
        print(f"{name}: max difference = {diff.max():.6f}, mean difference = {diff.mean():.6f}")
        print(f"python: {rearranged}")
        print(f"esp: {esp_flat}")
