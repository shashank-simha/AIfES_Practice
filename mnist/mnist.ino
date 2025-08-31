#include <Arduino.h>
#include <SD_MMC.h>
#include "mnist_model.h"
#include "dataset.h"
#include "mnist_data.h"

// Set stack size for loopTask to handle large buffers and AIfES internals
SET_LOOP_TASK_STACK_SIZE(256 * 1024);  // 256KB

// SD card pins
#define SD_MMC_CMD 38
#define SD_MMC_CLK 39
#define SD_MMC_D0  40

#define TRAIN_DATASET 200
#define TEST_DATASET 996 * 2      // 996 images per chunk
#define BATCH_SIZE 4
#define EPOCHS 3
#define RETRAIN false

const char* test_image_files[] = {"/mnist_chunks/test_images_chunk0.bin", "/mnist_chunks/test_images_chunk1.bin"};
const char* test_label_files[] = {"/mnist_chunks/test_labels_chunk0.bin", "/mnist_chunks/test_labels_chunk1.bin"};

MNISTModel model;
// Dataset train_ds(train_input_data, train_target_data, TRAIN_DATASET);
Dataset* test_ds = nullptr;

void setup() {
    Serial.begin(115200);
    while (!Serial);

    SD_MMC.setPins(SD_MMC_CLK, SD_MMC_CMD, SD_MMC_D0);
    if (!SD_MMC.begin("/sdcard", true, true, SDMMC_FREQ_DEFAULT, 5)) {
      Serial.println("Card Mount Failed");
        while (1);
    }
    uint8_t cardType = SD_MMC.cardType();
    if(cardType == CARD_NONE) {
        Serial.println("No SD_MMC card attached");
        while (1);
    }

    if (!psramInit()) {
        Serial.println(F("PSRAM init failed"));
        while (1);
    }

    // Instantiate Dataset after memory/SD ready
    test_ds = new Dataset(test_image_files, test_label_files, TEST_DATASET);
    Serial.println("Dataset initialized");

    if (!model.init()) {
        Serial.println(F("Model init failed"));
        while (1);
    }

    Serial.println(F("Type 't' to test the model"));
}

void loop() {
    if (Serial.available() > 0) {
        String cmd = Serial.readString();
        if (cmd.indexOf("t") > -1) {
            // train_ds.reset();
            // model.train(train_ds, TRAIN_DATASET, BATCH_SIZE, EPOCHS, RETRAIN);
            test_ds->reset();
            model.test(*test_ds, TEST_DATASET);
        }
    }
}
