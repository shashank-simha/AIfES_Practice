#include <Arduino.h>
#include "mnist_model.h"
#include "dataset.h"

#define TEST_DATASET 20

MNISTModel model;
Dataset test_ds(TEST_DATASET);

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
            model.test(test_ds, TEST_DATASET);
        }
    }
}
