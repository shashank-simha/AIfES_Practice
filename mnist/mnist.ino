#include <Arduino.h>
#include "mnist_model.h"
#include "dataset.h"

#include "mnist_data.h"

#define TRAIN_DATASET 200
#define TEST_DATASET 20

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
            // model.test(test_ds, TEST_DATASET);
            model.test(train_ds, TRAIN_DATASET);
        }
    }
}
