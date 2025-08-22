#include <esp_heap_caps.h>
void setup() {
  Serial.begin(115200);
  while (!Serial);
  if (psramInit()) {
    Serial.println("PSRAM initialized");
    float *test = (float *)ps_malloc(1024 * sizeof(float));
    if (test) {
      Serial.println("PSRAM allocation OK");
      free(test);
    } else {
      Serial.println("PSRAM allocation failed");
    }
  } else {
    Serial.println("PSRAM init failed");
  }
}
void loop() {}