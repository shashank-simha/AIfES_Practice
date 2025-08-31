#!/usr/bin/env python3
import numpy as np
from torchvision import datasets
import os

# ------------------ Config ------------------
NUM_CLASSES = 10
CHUNK_SIZE_TRAIN = 100   # Target images per chunk
CHUNK_SIZE_TEST = 100    # Target images per chunk
OUTPUT_DIR = "mnist_chunks"

# ------------------ Load MNIST ------------------
full_train = datasets.MNIST(root="data", train=True, download=True)
full_test = datasets.MNIST(root="data", train=False, download=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_chunks(images, labels, chunk_size, prefix):
    """
    Split dataset into chunks of size chunk_size.
    Any leftover images are distributed to chunks to fill them.
    Images are saved as raw uint8 (0-255) without normalization.
    """
    total_samples = len(labels)
    num_chunks = (total_samples + chunk_size - 1) // chunk_size  # ceil division

    # Shuffle dataset globally
    perm = np.random.permutation(total_samples)
    images, labels = images[perm], labels[perm]

    # Split into chunks
    start_idx = 0
    for chunk_id in range(num_chunks):
        end_idx = start_idx + chunk_size
        chunk_imgs = images[start_idx:end_idx]
        chunk_lbls = labels[start_idx:end_idx]

        # If last chunk is smaller than chunk_size, distribute leftovers from earlier samples
        if len(chunk_lbls) < chunk_size:
            needed = chunk_size - len(chunk_lbls)
            extra_idx = np.random.choice(total_samples, size=needed, replace=False)
            chunk_imgs = np.concatenate([chunk_imgs, images[extra_idx]], axis=0)
            chunk_lbls = np.concatenate([chunk_lbls, labels[extra_idx]], axis=0)

        # Shuffle within chunk
        perm_chunk = np.random.permutation(len(chunk_lbls))
        chunk_imgs, chunk_lbls = chunk_imgs[perm_chunk], chunk_lbls[perm_chunk]

        # Save as raw uint8 binary
        img_file = os.path.join(OUTPUT_DIR, f"{prefix}_images_chunk{chunk_id}.bin")
        lbl_file = os.path.join(OUTPUT_DIR, f"{prefix}_labels_chunk{chunk_id}.bin")
        chunk_imgs.astype(np.uint8).tofile(img_file)
        chunk_lbls.astype(np.uint8).tofile(lbl_file)

        print(f"Saved {prefix} chunk {chunk_id}: {len(chunk_lbls)} samples "
              f"(class counts {np.bincount(chunk_lbls, minlength=NUM_CLASSES)})")

        start_idx += chunk_size

# ------------------ Process train/test ------------------
create_chunks(full_train.data.numpy(), full_train.targets.numpy(),
              CHUNK_SIZE_TRAIN, prefix="train")

create_chunks(full_test.data.numpy(), full_test.targets.numpy(),
              CHUNK_SIZE_TEST, prefix="test")

print(f"All chunks saved in {OUTPUT_DIR}/")
