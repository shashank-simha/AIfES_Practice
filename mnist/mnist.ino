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

#define NUM_TRAIN_CHUNKS 2
#define NUM_IMAGES_PER_TRAIN_CHUNK 100
#define NUM_TEST_CHUNKS 5
#define NUM_IMAGES_PER_TEST_CHUNK 2000
#define BATCH_SIZE 100
#define EPOCHS 5
#define RETRAIN false
#define EARLY_STOPPING true
#define EARLY_STOPPING_TARGET_LOSS 0.075

const char* train_image_files[] = {
    "/mnist_chunks/train_images_chunk0.bin",
    "/mnist_chunks/train_images_chunk1.bin",
    "/mnist_chunks/train_images_chunk2.bin",
    "/mnist_chunks/train_images_chunk3.bin",
    "/mnist_chunks/train_images_chunk4.bin"
};
const char* train_label_files[] = {
    "/mnist_chunks/train_labels_chunk0.bin",
    "/mnist_chunks/train_labels_chunk1.bin",
    "/mnist_chunks/train_labels_chunk2.bin",
    "/mnist_chunks/train_labels_chunk3.bin",
    "/mnist_chunks/train_labels_chunk4.bin"
};
const char* test_image_files[] = {
    "/mnist_chunks/test_images_chunk0.bin",
    "/mnist_chunks/test_images_chunk1.bin",
    "/mnist_chunks/test_images_chunk2.bin",
    "/mnist_chunks/test_images_chunk3.bin",
    "/mnist_chunks/test_images_chunk4.bin"
};
const char* test_label_files[] = {
    "/mnist_chunks/test_labels_chunk0.bin",
    "/mnist_chunks/test_labels_chunk1.bin",
    "/mnist_chunks/test_labels_chunk2.bin",
    "/mnist_chunks/test_labels_chunk3.bin",
    "/mnist_chunks/test_labels_chunk4.bin"
};

MNISTModel model;
Dataset* train_ds = nullptr;
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
    // train_ds = new Dataset(train_image_files, train_label_files, NUM_TRAIN_CHUNKS);
    test_ds = new Dataset(test_image_files, test_label_files, NUM_TEST_CHUNKS);
    Serial.println("Dataset initialized");

    if (!model.init()) {
        Serial.println(F("Model init failed"));
        while (1);
    }

    Serial.println(F("Type 't' to train and test the model"));
}

void loop() {
    if (Serial.available() > 0) {
        String cmd = Serial.readString();
        if (cmd.indexOf("t") > -1) {
            // train_ds->reset();
            // model.train(*train_ds, NUM_TRAIN_CHUNKS * NUM_IMAGES_PER_TRAIN_CHUNK, BATCH_SIZE, EPOCHS, RETRAIN, EARLY_STOPPING, EARLY_STOPPING_TARGET_LOSS);
            test_ds->reset();
            model.test(*test_ds, NUM_TEST_CHUNKS * NUM_IMAGES_PER_TEST_CHUNK);
        }
    }
}
