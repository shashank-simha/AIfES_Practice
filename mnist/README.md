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
| 2  | **Conv2D** | Filters: 8, Kernel: (3×3), Stride: (1,1), Padding: (1,1) | `[1, 8, 28, 28]` |
| 3  | **ReLU**   | Activation                                    | `[1, 8, 28, 28]`    |
| 4  | **MaxPool**| Pool: (2×2), Stride: (2,2)                    | `[1, 8, 14, 14]`    |
| 5  | **Conv2D** | Filters: 16, Kernel: (3×3), Stride: (1,1), Padding: (1,1) | `[1, 16, 14, 14]` |
| 6  | **ReLU**   | Activation                                    | `[1, 16, 14, 14]`   |
| 7  | **MaxPool**| Pool: (2×2), Stride: (2,2)                    | `[1, 16, 7, 7]`     |
| 8  | **Flatten**| Reshape                                       | `[1, 784]`          |
| 9  | **Dense**  | Neurons: 64                                   | `[1, 64]`           |
| 10 | **ReLU**   | Activation                                    | `[1, 64]`           |
| 11 | **Dense**  | Neurons: 10                                   | `[1, 10]`           |
| 12 | **Softmax**| Classification output                         | `[1, 10]`           |

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
15:43:44.542 -> Initializing MNISTModel...
15:43:44.542 -> Parameter memory allocated: 208552 bytes, Free PSRAM: 8175160 bytes
15:43:44.542 -> Loading model parameters.......
15:43:44.704 -> Parameters loaded from /params.bin
15:43:44.704 -> Inference memory allocated: 50176 bytes, Free PSRAM: 8123912 bytes
15:43:44.704 -> Type 't' to test the model
15:43:48.604 -> Training on 200 images...
15:43:48.604 -> Training memory allocated: 507224 bytes, Free PSRAM: 7616004 bytes
15:45:09.284 -> Epoch 1/3 - Loss: 0.2499
15:45:09.284 -> Storing model parameters.......
15:45:14.371 -> Parameters saved to /params.bin, total bytes written=208552
15:46:35.050 -> Epoch 2/3 - Loss: 0.1717
15:46:35.050 -> Storing model parameters.......
15:46:40.358 -> Parameters saved to /params.bin, total bytes written=208552
15:48:01.058 -> Epoch 3/3 - Loss: 0.1310
15:48:01.058 -> Storing model parameters.......
15:48:06.370 -> Parameters saved to /params.bin, total bytes written=208552
15:48:06.370 -> Finished training
15:48:06.370 -> Training memory freed, Free PSRAM: 8123912 bytes
15:48:06.370 -> Testing 20 images...
15:48:06.467 -> Image 0: Predicted 0, Actual 0, Correct
15:48:06.563 -> Image 1: Predicted 0, Actual 0, Correct
15:48:06.660 -> Image 2: Predicted 1, Actual 1, Correct
15:48:06.757 -> Image 3: Predicted 1, Actual 1, Correct
15:48:06.853 -> Image 4: Predicted 2, Actual 2, Correct
15:48:06.951 -> Image 5: Predicted 2, Actual 2, Correct
15:48:07.048 -> Image 6: Predicted 3, Actual 3, Correct
15:48:07.143 -> Image 7: Predicted 3, Actual 3, Correct
15:48:07.241 -> Image 8: Predicted 4, Actual 4, Correct
15:48:07.338 -> Image 9: Predicted 4, Actual 4, Correct
15:48:07.434 -> Image 10: Predicted 5, Actual 5, Correct
15:48:07.531 -> Image 11: Predicted 5, Actual 5, Correct
15:48:07.661 -> Image 12: Predicted 6, Actual 6, Correct
15:48:07.725 -> Image 13: Predicted 6, Actual 6, Correct
15:48:07.855 -> Image 14: Predicted 7, Actual 7, Correct
15:48:07.951 -> Image 15: Predicted 7, Actual 7, Correct
15:48:08.047 -> Image 16: Predicted 8, Actual 8, Correct
15:48:08.144 -> Image 17: Predicted 8, Actual 8, Correct
15:48:08.241 -> Image 18: Predicted 9, Actual 9, Correct
15:48:08.339 -> Image 19: Predicted 9, Actual 9, Correct
15:48:08.339 -> Accuracy: 20/20 (100.00%)
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
  - Current on-device training is slow (~1m20s for 200 samples).
  - Explore lighter architectures or hybrid training (host-assisted).
  - Optimize AIfES training loops for ESP32.
