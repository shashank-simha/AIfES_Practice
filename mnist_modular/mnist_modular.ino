#include <Arduino.h>
#include <SD_MMC.h>
#include "logger.h"
#include <vector>
#include <string>
#include "SDCardDataset.h"
#include "SDMMCFileAdapter.h"
#include "MNISTModel.h"

#define LOG_LEVEL LOG_LEVEL_DEBUG

// Set stack size for loopTask to handle large buffers and AIfES internals
SET_LOOP_TASK_STACK_SIZE(256 * 1024);  // 256KB

// SD card pins
#define SD_MMC_CMD 38
#define SD_MMC_CLK 39
#define SD_MMC_D0  40

#define NUM_TRAIN_CHUNKS 5
#define NUM_IMAGES_PER_TRAIN_CHUNK 2000
#define NUM_TEST_CHUNKS 1
#define NUM_IMAGES_PER_TEST_CHUNK 2000
#define BATCH_SIZE 100
#define EPOCHS 5
#define RETRAIN true
#define EARLY_STOPPING true
#define EARLY_STOPPING_TARGET_LOSS 0.075

std::vector<std::string> train_image_files = {
    "/mnist_chunks/train_images_chunk0.bin",
    "/mnist_chunks/train_images_chunk1.bin",
    "/mnist_chunks/train_images_chunk2.bin",
    "/mnist_chunks/train_images_chunk3.bin",
    "/mnist_chunks/train_images_chunk4.bin"
};
std::vector<std::string> train_label_files = {
    "/mnist_chunks/train_labels_chunk0.bin",
    "/mnist_chunks/train_labels_chunk1.bin",
    "/mnist_chunks/train_labels_chunk2.bin",
    "/mnist_chunks/train_labels_chunk3.bin",
    "/mnist_chunks/train_labels_chunk4.bin"
};
std::vector<std::string> test_image_files = {
    "/mnist_chunks/test_images_chunk0.bin",
    "/mnist_chunks/test_images_chunk1.bin",
    "/mnist_chunks/test_images_chunk2.bin",
    "/mnist_chunks/test_images_chunk3.bin",
    "/mnist_chunks/test_images_chunk4.bin"
};
std::vector<std::string> test_label_files = {
    "/mnist_chunks/test_labels_chunk0.bin",
    "/mnist_chunks/test_labels_chunk1.bin",
    "/mnist_chunks/test_labels_chunk2.bin",
    "/mnist_chunks/test_labels_chunk3.bin",
    "/mnist_chunks/test_labels_chunk4.bin"
};

MNISTModel* model = nullptr;
SDCardDataset<uint8_t, uint8_t>* train_ds = nullptr;
SDCardDataset<uint8_t, uint8_t>* test_ds = nullptr;

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

    // --- Build dataset config ---
    DatasetConfig db_cfg;
    db_cfg.input_shape = {1, 28, 28};
    db_cfg.label_shape = {1};
    db_cfg.alloc_strategy = AllocationStrategy::Lazy;
    db_cfg.end_policy = BatchEndPolicy::DropLast;
    db_cfg.allocator_fn = ps_malloc;
    db_cfg.free_fn = free;

    // Instantiate Dataset after memory/SD ready
    train_ds = new SDCardDataset<uint8_t, uint8_t>(db_cfg, train_image_files, train_label_files, new SDMMCFileAdapter());
    test_ds = new SDCardDataset<uint8_t, uint8_t>(db_cfg, test_image_files, test_label_files, new SDMMCFileAdapter());

    test_ds->set_transform([](void* input, void* label, size_t batch_size) {
    uint8_t* in_u8 = reinterpret_cast<uint8_t*>(input);   // raw dataset input
    float* in_f32 = reinterpret_cast<float*>(input);      // buffer in test() already float*
    
    for (size_t i = 0; i < batch_size * INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH; ++i) {
        in_f32[i] = static_cast<float>(in_u8[i]) / 255.0f;
    }

    // Convert labels from uint8_t → uint32_t
    uint8_t* lbl_u8 = reinterpret_cast<uint8_t*>(label);
    uint32_t* lbl_u32 = reinterpret_cast<uint32_t*>(label); // points to test() target_buffer
    for (size_t i = 0; i < batch_size; ++i) {
        lbl_u32[i] = static_cast<uint32_t>(lbl_u8[i]);
    }
});
    Serial.println("Dataset initialized");

    // --- Build dataset config ---
    ModelConfig model_cfg;
    model_cfg.input_shape = {1, 28, 28};
    model_cfg.output_shape = {1};
    model_cfg.allocator_fn = ps_malloc;
    model_cfg.free_fn = free;

    model = new MNISTModel(model_cfg);
    model->init();
    Serial.println("Model initialized");

    Serial.println(F("Type 't' to train and test the model"));
}

void loop() {
    if (Serial.available() > 0) {
        String cmd = Serial.readString();
        if (cmd.indexOf("t") > -1) {
            train_ds->reset();
            // model->train(*train_ds, NUM_TRAIN_CHUNKS * NUM_IMAGES_PER_TRAIN_CHUNK, BATCH_SIZE, EPOCHS, RETRAIN, EARLY_STOPPING, EARLY_STOPPING_TARGET_LOSS);
            test_ds->reset();
            model->test(*test_ds, NUM_TEST_CHUNKS * NUM_IMAGES_PER_TEST_CHUNK);
        }
    }
}
