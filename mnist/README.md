# MNIST Classifier on ESP32 with AIfES

## 1. Description
This project implements an **MNIST digit classifier** on the ESP32 using the [AIfES framework](https://aifes.ai/).
It demonstrates **on-device training and inference** of a compact convolutional neural network (CNN), allowing the ESP32 to both learn and recognize handwritten digits. Training on small subset of the MNIST dataset is performed for demonstration purposes. The full training process is carried out on a host system (Python). The resulting weights are exported as **binary parameter dumps** and loaded into the ESP32 for inference.

Key aspects:
- CNN model built directly with AIfES layers
- Training (partial) and inference performed on-device
- Dataset preparation and export utilities included
- Parameter dumps for debugging and verification

---

## 2. Folder Structure
```
├── dataset.cpp # Dataset loading and interface
├── dataset.h
├── mnist_data.h # Small MNIST dataset stored in PROGMEM
├── mnist.ino # Arduino sketch (main entry for ESP32)
├── mnist_model.cpp # CNN model implementation in AIfES
├── mnist_model.h
├── params.bin # Binary dump of trained parameters
├── utils/
│ ├── generate_dataset.py # Convert MNIST data into C header format
│ ├── train.py # Train and export model weights from Python
```

---

## 3. Model Structure
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

## 4. Usage

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
void train(Dataset& ds, uint32_t num_samples, uint32_t batch_size, uint32_t num_epoch, bool retrain);
```
- Trains model on dataset (`ds`).
- If `retrain == true`, reinitializes model parameters.
- Saves updated weights to `params.bin` in SPIFFS after each epoch.

**⚡ Note:**
- Use `utils/generate_dataset.py` to generate a small subset of MNIST dataset. Script generates `mnist_data.h`.
- Use `utils/train.py` to generate pretrained `params.bin`.
- Upload `params.bin` to ESP32 flash memory using `SPIFFS_Server` Arduino sketch for validating with pretrained parameters.

---

## 5. Sample output from Serial terminal
```
07:57:26.697 -> Initializing MNISTModel...
07:57:26.697 -> Parameter memory allocated: 52968 bytes, Free PSRAM: 8330808 bytes
07:57:26.697 -> Loading model parameters.......
07:57:26.826 -> Parameters loaded from /params.bin
07:57:26.826 -> Inference memory allocated: 25088 bytes, Free PSRAM: 8305160 bytes
07:57:26.826 -> Type 't' to test the model
07:57:32.418 -> Training on 200 images...
07:57:32.418 -> Training memory allocated: 153816 bytes, Free PSRAM: 8149508 bytes
07:57:57.214 -> Epoch 1/3 - Loss: 0.3916
07:57:57.214 -> Storing model parameters.......
07:57:58.730 -> Parameters saved to /params.bin, total bytes written=52968
07:58:23.538 -> Epoch 2/3 - Loss: 0.3087
07:58:23.538 -> Storing model parameters.......
07:58:24.965 -> Parameters saved to /params.bin, total bytes written=52968
07:58:49.746 -> Epoch 3/3 - Loss: 0.2574
07:58:49.746 -> Storing model parameters.......
07:58:51.195 -> Parameters saved to /params.bin, total bytes written=52968
07:58:51.195 -> Finished training
07:58:51.195 -> Training memory freed, Free PSRAM: 8305160 bytes
07:58:51.195 -> Testing 20 images...
07:58:51.227 -> Image 0: Predicted 0, Actual 0, Correct
07:58:51.259 -> Image 1: Predicted 0, Actual 0, Correct
07:58:51.291 -> Image 2: Predicted 1, Actual 1, Correct
07:58:51.323 -> Image 3: Predicted 1, Actual 1, Correct
07:58:51.355 -> Image 4: Predicted 2, Actual 2, Correct
07:58:51.386 -> Image 5: Predicted 2, Actual 2, Correct
07:58:51.419 -> Image 6: Predicted 3, Actual 3, Correct
07:58:51.451 -> Image 7: Predicted 3, Actual 3, Correct
07:58:51.482 -> Image 8: Predicted 4, Actual 4, Correct
07:58:51.482 -> Image 9: Predicted 4, Actual 4, Correct
07:58:51.515 -> Image 10: Predicted 5, Actual 5, Correct
07:58:51.547 -> Image 11: Predicted 5, Actual 5, Correct
07:58:51.579 -> Image 12: Predicted 6, Actual 6, Correct
07:58:51.612 -> Image 13: Predicted 6, Actual 6, Correct
07:58:51.644 -> Image 14: Predicted 7, Actual 7, Correct
07:58:51.676 -> Image 15: Predicted 7, Actual 7, Correct
07:58:51.708 -> Image 16: Predicted 8, Actual 8, Correct
07:58:51.740 -> Image 17: Predicted 8, Actual 8, Correct
07:58:51.773 -> Image 18: Predicted 9, Actual 9, Correct
07:58:51.807 -> Image 19: Predicted 9, Actual 9, Correct
07:58:51.807 -> Accuracy: 20/20 (100.00%)
```

---

## 6. Key Features
- **Robust Parameter Persistence**
  - Model weights saved/loaded from `params.bin` in SPIFFS.
  - Training can resume or restart (retrain option).
- **Dataset Handling**
  - Small MNIST subsets stored in PROGMEM for testing.
  - Python scripts for dataset generation/conversion.
- **Clear Modular Design**
  - Separates dataset handling, model definition, utilities, and training (host side) scripts.
  - Easy to extend/adapt for other datasets.
- **Scalable Architecture**
  - CNN model design supports minimal changes for other tasks.
  - Binary parameter handling simplifies model porting/updates.

---

## 7. TODO
- **Improve dataset handling**:
  - Store and load larger datasets in binary format instead of headers.
  - Explore streaming from SD card or SPIFFS to handle full MNIST or custom datasets.

- **Reduce training latency**:
  - Current on-device training is slow (~25s per epoch for 200 samples).
  - Explore lighter architectures or hybrid training (host-assisted).
  - Optimize AIfES training loops for ESP32.
