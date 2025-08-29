#include <Arduino.h>
#include "mnist_model.h"
#include "dataset.h"

#include "mnist_data.h"

// Set stack size for loopTask to handle large buffers and AIfES internals
SET_LOOP_TASK_STACK_SIZE(256 * 1024);  // 256KB

#define TRAIN_DATASET 200
#define TEST_DATASET 20
#define BATCH_SIZE 4
#define EPOCHS 3

MNISTModel model;
Dataset train_ds(train_input_data, train_target_data, TRAIN_DATASET);
Dataset test_ds(test_input_data, test_target_data, TEST_DATASET);

void setup() {
    Serial.begin(115200);
    while (!Serial);

    if (!psramInit()) {
        Serial.println(F("PSRAM init failed"));
        while (1);
    }

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
            test_ds.reset();
            model.train(train_ds, TRAIN_DATASET, BATCH_SIZE, EPOCHS);
            model.test(test_ds, TEST_DATASET);
        }
    }
}
