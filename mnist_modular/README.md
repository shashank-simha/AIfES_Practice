# MNIST Classifier on ESP32 with AIfES

## 1. Description
This project implements an **MNIST digit classifier** on the ESP32 using the [AIfES framework](https://aifes.ai/).
It demonstrates **on-device training and inference** of a compact convolutional neural network (CNN), allowing the ESP32 to both learn and recognize handwritten digits. Complete training process on a subset (10000 samples) of the MNIST dataset is performed on-device and weights are saved for further usage.

The dataset (MNIST) is pre-processed into balanced chunks stored on the SD card. During runtime, the ESP32 loads these chunks into PSRAM, trains the model in batches, and can run inference directly on-device.

Additionaly, training of a model with same architecture (and similar dataset count) is carried out on a host system (Python). The resulting weights are exported as **binary parameter dumps** and loaded into the ESP32 to demonstrate inference with pretrained parameters and host-assisted training.

Key aspects:
- CNN model built directly with AIfES layers
- Training and inference performed on-device
- Dataset preparation and export utilities included
- Parameter dumps for debugging and verification

---

## 2. Requirements
- **Hardware (tested on ESP32-S3)**
  - ESP32-S3 module with:
    - **8MB Flash** (with **SPIFFS** enabled in the partition scheme)
    - **8MB PSRAM**
    - Onboard or external **SD card slot**
  - **Minimum Hardware (sufficient to run as-is)**
    - **4MB** Flash (with SPIFFS enabled)
    - **4MB PSRAM**
    - **SD card** with ~20MB free space
  - **Notes**
    - The SD card is used to store dataset chunks and model parameters.
    - PSRAM is required for batch buffers, training data, and model weights.
    - The example can be ported with **minimal changes** to boards with different memory configurations or to projects using alternative storage/streaming strategies.

## 3. Folder Structure
```
├── dataset.cpp # Dataset loading and interface
├── dataset.h
├── mnist.ino # Arduino sketch (main entry for ESP32)
├── mnist_model.cpp # CNN model implementation in AIfES
├── mnist_model.h
├── params.bin # Binary dump of trained parameters
├── utils/
│ ├── generate_dataset.py # Convert MNIST data into binary chunks
│ ├── train.py # Train and export model weights from Python
```

---

## 4. Model Structure
The neural network follows a **compact CNN architecture** optimized for embedded systems.
Below is the layer-by-layer structure with shapes and parameters:

| #  | Layer      | Details / Parameters                           | Output Shape        |
|----|------------|-----------------------------------------------|---------------------|
| 1  | **Input**  | Shape: `[1, 1, 28, 28]`                       | `[1, 1, 28, 28]`    |
| 2  | **Conv2D** | Filters: 4, Kernel: (3×3), Stride: (1,1), Padding: (1,1), Dilation: (1,1) | `[1, 4, 28, 28]` |
| 3  | **ReLU**   | Activation                                    | `[1, 4, 28, 28]`    |
| 4  | **MaxPool**| Pool: (2×2), Stride: (2,2), Padding: (0, 0)   | `[1, 4, 14, 14]`    |
| 5  | **Conv2D** | Filters: 8, Kernel: (3×3), Stride: (1,1), Padding: (1,1), Dilation: (1,1) | `[1, 8, 14, 14]` |
| 6  | **ReLU**   | Activation                                    | `[1, 8, 14, 14]`   |
| 7  | **MaxPool**| Pool: (2×2), Stride: (2,2), Padding: (0, 0)   | `[1, 8, 7, 7]`     |
| 8  | **Flatten**| Reshape                                       | `[1, 392]`          |
| 9  | **Dense**  | Neurons: 64                                   | `[1, 32]`           |
| 10 | **ReLU**   | Activation                                    | `[1, 32]`           |
| 11 | **Dense**  | Neurons: 10                                   | `[1, 10]`           |
| 12 | **Softmax**| Classification output                         | `[1, 10]`           |

**Note:**
- All Convolution and Maxpool layers use NCHW (Channel axis: 1) ordering.

---

## 5. Usage

### Dataset Preparation
The MNIST dataset is divided into **balanced chunks** to ensure fair representation of classes in each split.

- Each **training chunk** contains **2000 images** (and corresponding labels).
- Each **test chunk** contains **2000 images**.
- Images and labels are stored in **raw `uint8` binary format** (`.bin` files).
- For this example, **5 training chunks** (10,000 images) and **1 test chunk** (2,000 images) are copied to the SD card.
- The dataset chunks are generated using the script:
  ```bash
  python utils/generate_dataset.py
  ```
- On the ESP32, datasets are initialized in code as:
  ```cpp
  #define NUM_TRAIN_CHUNKS 5
  #define NUM_IMAGES_PER_TRAIN_CHUNK 2000
  #define NUM_TEST_CHUNKS 1
  #define NUM_IMAGES_PER_TEST_CHUNK 2000

  const char* train_image_files[] = {"/mnist_chunks/train_images_chunk0.bin", "/mnist_chunks/train_images_chunk1.bin", "/mnist_chunks/train_images_chunk2.bin", "/mnist_chunks/train_images_chunk3.bin", "/mnist_chunks/train_images_chunk4.bin"};
  const char* train_label_files[] = {"/mnist_chunks/train_labels_chunk0.bin", "/mnist_chunks/train_labels_chunk1.bin", "/mnist_chunks/train_labels_chunk2.bin", "/mnist_chunks/train_labels_chunk3.bin", "/mnist_chunks/train_labels_chunk4.bin"};
  const char* test_image_files[] = {"/mnist_chunks/test_images_chunk0.bin"};
  const char* test_label_files[] = {"/mnist_chunks/test_labels_chunk0.bin",};

  ...

  train_ds = new Dataset(train_image_files, train_label_files, NUM_TRAIN_CHUNKS);
  test_ds  = new Dataset(test_image_files,  test_label_files,  NUM_TEST_CHUNKS);
  ```

### Model API
The model provides a straightforward API for initialization, inference, testing, and training.

#### Initialize Model
```cpp
bool init();
```
- Builds layers and allocates memory for the CNN model.
- Loads pretrained weights from `params.bin` in SPIFFS if available.
- Returns `true` on success.

#### Run Inference
```cpp
uint32_t infer(float* input_buffer);
```
- Takes a normalized input image `[1×28×28]`.
- Performs forward inference.
- Returns predicted digit (`0–9`).

#### Test Dataset
```cpp
void test(Dataset& ds, uint32_t num_samples);
```
- Evaluates model on dataset (`ds`) with `num_samples` images.
- Outputs accuracy and classification results.

#### Train Model
```cpp
void train(Dataset& ds, uint32_t num_samples, uint32_t batch_size, uint32_t num_epoch, bool retrain, bool early_stopping, float_t early_stopping_target_loss);
```
- Trains model on dataset (`ds`).
- If `retrain == true`, reinitializes model parameters.
- Saves updated weights to `params.bin` in SPIFFS after each epoch.
- Early exits if required target loss is achieved

**⚡ Note:**
- Use `utils/train.py` to generate pretrained `params.bin`.
- For validating with pretrained parameters:
  - Upload `params.bin` to ESP32 flash memory using `SPIFFS_Server` Arduino sketch (in root folder)
  - Upload the `mnist.ino` again. The program should load the model parameters from `params.bin` in flash

---

## 6. Sample output from Serial terminal
```
[2025-08-31 21:10:31.889] Initializing MNISTModel...
[2025-08-31 21:10:31.890] Parameter memory allocated: 52968 bytes, Free PSRAM: 5151252 bytes
[2025-08-31 21:10:31.895] Loading model parameters.......
[2025-08-31 21:10:32.015] Inference memory allocated: 25088 bytes, Free PSRAM: 5125604 bytes
[2025-08-31 21:10:32.022] Type 't' to train and test the model
[2025-08-31 21:10:45.930] Loaded image chunk 0 with 2000 samples
[2025-08-31 21:10:45.944] Loaded label chunk 0 with 2000 labels
[2025-08-31 21:10:45.945] Training on 10000 images...
[2025-08-31 21:10:45.947] Retraining the model: assiging random initial values to params
[2025-08-31 21:10:45.953] Training memory allocated: 153816 bytes, Free PSRAM: 4969952 bytes
[2025-08-31 21:14:48.947] Progress: [==========>                                       ] 20% 20/100 Steps (1/5 epoch)...
[2025-08-31 21:14:49.716] Loaded image chunk 1 with 2000 samples
[2025-08-31 21:14:49.724] Loaded label chunk 1 with 2000 labels
[2025-08-31 21:18:52.728] Progress: [====================>                             ] 40% 40/100 Steps (1/5 epoch)...
[2025-08-31 21:18:53.494] Loaded image chunk 2 with 2000 samples
[2025-08-31 21:18:53.502] Loaded label chunk 2 with 2000 labels
...
[2025-08-31 21:31:04.054] Progress: [==================================================] 100% 100/100 Steps (1/5 epoch)...
[2025-08-31 21:31:04.072] Epoch 1/5 - Loss: 91.5105
[2025-08-31 21:31:04.077] Storing model parameters.......
[2025-08-31 21:31:05.320] Parameters saved to /params.bin, total bytes written=52968
[2025-08-31 21:31:06.089] Loaded image chunk 0 with 2000 samples
[2025-08-31 21:31:06.097] Loaded label chunk 0 with 2000 labels
...
[2025-08-31 22:52:24.912] Progress: [==================================================] 100% 100/100 Steps (5/5 epoch)...
[2025-08-31 22:52:24.924] Epoch 5/5 - Loss: 7.2545
[2025-08-31 22:52:24.927] Storing model parameters.......
[2025-08-31 22:52:26.247] Parameters saved to /params.bin, total bytes written=52968
[2025-08-31 22:52:26.255] Finished training
[2025-08-31 22:52:26.255] Training memory freed, Free PSRAM: 5125604 bytes
[2025-08-31 22:52:27.007] Loaded image chunk 0 with 2000 samples
[2025-08-31 22:52:27.015] Loaded label chunk 0 with 2000 labels
[2025-08-31 22:52:27.017] Testing 2000 images...
[2025-08-31 22:53:32.289] Progress: [==================================================] 100% 2000/2000 Images...
[2025-08-31 22:53:32.294] Accuracy: 1910/2000 (95.50%)
```

---

## 7. Key Features
- **Robust Parameter Persistence**
  - Model weights saved/loaded from `params.bin` in SPIFFS.
  - Training can resume or restart (retrain option).
- **Flexible Dataset Handling**
  - Dataset class manages **chunked loading from SD card**, automatically swapping chunks in and out of PSRAM.
  - Model only requests the *next batch* — it does not depend on how the dataset is managed.
  - This means the `Dataset` implementation can be swapped or modified (e.g., streaming, compressed storage, alternative datasets) **without changing model code**.
- **On-Device Training & Inference**
  - Entire training and evaluation run directly on the ESP32-S3.
  - Host-side training utilities are optional, useful for debugging or preprocessing but **not required**.
- **Clear Modular Design**
  - Separation of concerns: dataset handling, model definition, utilities, and optional host-side scripts.
  - Easy to extend/adapt for other datasets and training strategies with minimal code changes.
- **Scalable Architecture**
  - CNN model design supports minimal changes for other tasks.
  - Binary parameter handling simplifies model porting/updates.

---

## 8. TODO
- **Reduce training latency**:
  - Current on-device training is slow (~1h45m for 10000 training samples over 5 epochs).
  - Explore lighter architectures or hybrid training (host-assisted).
  - Optimize AIfES training loops for ESP32.
