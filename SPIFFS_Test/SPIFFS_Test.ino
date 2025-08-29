#include <SPIFFS.h>

const char* filename = "/numbers.bin";

void setup() {
  Serial.begin(115200);
  delay(1000);

  // Mount SPIFFS
  if (!SPIFFS.begin(true)) {
    Serial.println("SPIFFS Mount Failed!");
    return;
  }

  // ==== Write 0 to 100 ====
  File f = SPIFFS.open(filename, FILE_WRITE);
  if (!f) {
    Serial.println("Failed to open file for writing");
    return;
  }

  for (int i = 0; i <= 100; i++) {
    f.write((uint8_t*)&i, sizeof(int));
  }
  f.close();
  Serial.println("File written");

  // ==== Read back ====
  f = SPIFFS.open(filename, FILE_READ);
  if (!f) {
    Serial.println("Failed to open file for reading");
    return;
  }

  int val;
  int index = 0;
  while (f.available() >= sizeof(int)) {
    f.readBytes((char*)&val, sizeof(int));
    Serial.printf("Index %d: %d\n", index++, val);
  }
  f.close();
}

void loop() {
}
