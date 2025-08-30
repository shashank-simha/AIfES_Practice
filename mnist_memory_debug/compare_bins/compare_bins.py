import numpy as np
import os

# -----------------------------
# Load ESP layer bin safely
# -----------------------------
def load_layer(file_path, shape):
    """Load layer from file; return None if missing/empty or size mismatch."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        print(f"Warning: {file_path} missing or empty, skipping")
        return None
    data = np.fromfile(file_path, dtype=np.float32)
    if data.size != np.prod(shape):
        print(f"Warning: {file_path} size mismatch (expected {np.prod(shape)}, got {data.size})")
        return None
    return data.reshape(shape)

# -----------------------------
# Compare two layers
# -----------------------------
def compare_layers(layer_name, array1, array2):
    if array1 is None or array2 is None:
        print(f"Skipping {layer_name} (missing data)")
        return

    diff = array1.flatten() - array2.flatten()
    print(f"=== {layer_name} Comparison ===")
    print(f"Max abs difference: {np.abs(diff).max():.6f}")
    print(f"Mean abs difference: {np.abs(diff).mean():.6f}")
    print("First 10 values (array1 vs array2):")
    for a, b in zip(array1.flatten()[:10], array2.flatten()[:10]):
        print(f"{a:.6f}  {b:.6f}")
    print("\n")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Layer shapes
    layer_shapes = {
        "l1_w": (8, 1, 3, 3),
        "l1_b": (8,),
        "l4_w": (16, 8, 3, 3),
        "l4_b": (16,),
        "l8_w": (64, 16, 7, 7),
        "l8_b": (64,),
        "l10_w": (10, 64),
        "l10_b": (10,),
    }

    # Load progmem and bin dumps
    progmem_layers = {name: load_layer(f"progmem_{name}.bin", shape)
                      for name, shape in layer_shapes.items()}
    bin_layers     = {name: load_layer(f"bin_{name}.bin", shape)
                      for name, shape in layer_shapes.items()}

    # Compare all layers
    for name in layer_shapes.keys():
        compare_layers(name, progmem_layers.get(name), bin_layers.get(name))

    # Just flatten both layers in row-major (C-style) and compare
    diff = progmem_layers["l10_w"].flatten(order='C') - bin_layers["l10_w"].flatten(order='C')
    print("Max diff:", np.max(np.abs(diff)))
