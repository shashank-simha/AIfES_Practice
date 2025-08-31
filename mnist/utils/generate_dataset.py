#!/usr/bin/env python3
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
import os

# ------------------ Config ------------------
NUM_CLASSES = 10
NUM_TRAIN_CHUNKS = 600
NUM_TEST_CHUNKS = 100
OUTPUT_DIR = "mnist_chunks"

# ------------------ Load MNIST ------------------
transform = ToTensor()
full_train = datasets.MNIST(root="data", train=True, transform=transform, download=True)
full_test = datasets.MNIST(root="data", train=False, transform=transform, download=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def stratified_chunks(images, labels, num_chunks, prefix):
    """
    Split dataset into balanced chunks with fair representation of each class.
    Saves each chunk to {prefix}_images_chunk{i}.bin and {prefix}_labels_chunk{i}.bin
    """
    class_indices = [np.where(labels == i)[0] for i in range(NUM_CLASSES)]
    # Shuffle each class
    for ci in class_indices:
        np.random.shuffle(ci)

    # Number of samples per class per chunk
    per_class_per_chunk = [len(ci) // num_chunks for ci in class_indices]

    # Build chunks
    for chunk_id in range(num_chunks):
        chunk_imgs, chunk_lbls = [], []
        for cls, ci in enumerate(class_indices):
            start = chunk_id * per_class_per_chunk[cls]
            end = (chunk_id + 1) * per_class_per_chunk[cls]
            idxs = ci[start:end]
            chunk_imgs.append(images[idxs])
            chunk_lbls.append(labels[idxs])
        # Concatenate classes for this chunk
        chunk_imgs = np.concatenate(chunk_imgs, axis=0)
        chunk_lbls = np.concatenate(chunk_lbls, axis=0)

        # Shuffle within chunk
        perm = np.random.permutation(len(chunk_lbls))
        chunk_imgs, chunk_lbls = chunk_imgs[perm], chunk_lbls[perm]

        # Save as binary
        img_file = os.path.join(OUTPUT_DIR, f"{prefix}_images_chunk{chunk_id}.bin")
        lbl_file = os.path.join(OUTPUT_DIR, f"{prefix}_labels_chunk{chunk_id}.bin")
        chunk_imgs.astype(np.uint8).tofile(img_file)
        chunk_lbls.astype(np.uint8).tofile(lbl_file)

        print(f"Saved {prefix} chunk {chunk_id}: {len(chunk_lbls)} samples "
              f"(class counts {np.bincount(chunk_lbls, minlength=NUM_CLASSES)})")

# ------------------ Process train/test ------------------
stratified_chunks(full_train.data.numpy(), full_train.targets.numpy(),
                  NUM_TRAIN_CHUNKS, prefix="train")

stratified_chunks(full_test.data.numpy(), full_test.targets.numpy(),
                  NUM_TEST_CHUNKS, prefix="test")

print(f"All chunks saved in {OUTPUT_DIR}/")
