#include <aifes.h>


void init_model() {
  Serial.println(F("Initializing model..."));
}

void train() {
  Serial.println("Training...");
}

void infer() {
  Serial.println("Inferring...");
}

void test() {
  Serial.println("Testing...");
}

// Setup function with PSRAM check
void setup() {
  Serial.begin(115200);
  while (!Serial);

  Serial.println(F("\n##################################"));
  Serial.println(F("ESP32 Information:"));
  Serial.printf("Internal Total Heap %d, Internal Used Heap %d, Internal Free Heap %d\n", ESP.getHeapSize(), ESP.getHeapSize()-ESP.getFreeHeap(), ESP.getFreeHeap());
  Serial.printf("Sketch Size %d, Free Sketch Space %d\n", ESP.getSketchSize(), ESP.getFreeSketchSpace());
  Serial.printf("SPIRam Total heap %d, SPIRam Free Heap %d\n", ESP.getPsramSize(), ESP.getFreePsram());
  Serial.printf("Chip Model %s, ChipRevision %d, Cpu Freq %d, SDK Version %s\n", ESP.getChipModel(), ESP.getChipRevision(), ESP.getCpuFreqMHz(), ESP.getSdkVersion());
  Serial.printf("Flash Size %d, Flash Speed %d\n", ESP.getFlashChipSize(), ESP.getFlashChipSpeed());
  Serial.println(F("##################################\n\n"));

  // Check PSRAM initialization
  if (!psramInit()) {
    Serial.println(F("PSRAM initialization failed"));
    while (1);
  }
  Serial.println(F("PSRAM initialized"));

  srand(analogRead(A5));
  init_model();
  Serial.println(F("Type >train< or >infer<"));
}

// Main loop for command input
void loop() {
  if (Serial.available() > 0) {
    String str = Serial.readString();
    if (str.indexOf("train") > -1) {
      train();
    } else if (str.indexOf("infer") > -1) {
      infer();
    } else {
      Serial.println(F("Unknown command"));
    }
  }
}

// Handle training errors
void error_handling_training(int8_t error_nr) {
  switch (error_nr) {
    case 0: break;
    case -1: Serial.println(F("ERROR! Tensor dtype")); break;
    case -2: Serial.println(F("ERROR! Tensor shape: Data Number")); break;
    case -3: Serial.println(F("ERROR! Input tensor shape does not correspond to ANN inputs")); break;
    case -4: Serial.println(F("ERROR! Output tensor shape does not correspond to ANN outputs")); break;
    case -5: Serial.println(F("ERROR! Use crossentropy as loss for softmax")); break;
    case -6: Serial.println(F("ERROR! learn_rate or sgd_momentum negative")); break;
    case -7: Serial.println(F("ERROR! Init uniform weights min - max wrong")); break;
    case -8: Serial.println(F("ERROR! batch_size: min = 1 / max = Number of training data")); break;
    case -9: Serial.println(F("ERROR! Unknown activation function")); break;
    case -10: Serial.println(F("ERROR! Unknown loss function")); break;
    case -11: Serial.println(F("ERROR! Unknown init weights method")); break;
    case -12: Serial.println(F("ERROR! Unknown optimizer")); break;
    case -13: Serial.println(F("ERROR! Not enough memory")); break;
    default: Serial.println(F("Unknown error"));
  }
}

// Handle inference errors
void error_handling_inference(int8_t error_nr) {
  switch (error_nr) {
    case 0: break;
    case -1: Serial.println(F("ERROR! Tensor dtype")); break;
    case -2: Serial.println(F("ERROR! Tensor shape: Data Number")); break;
    case -3: Serial.println(F("ERROR! Input tensor shape does not correspond to ANN inputs")); break;
    case -4: Serial.println(F("ERROR! Output tensor shape does not correspond to ANN outputs")); break;
    case -5: Serial.println(F("ERROR! Unknown activation function")); break;
    case -6: Serial.println(F("ERROR! Not enough memory")); break;
    default: Serial.println(F("Unknown error"));
  }
}