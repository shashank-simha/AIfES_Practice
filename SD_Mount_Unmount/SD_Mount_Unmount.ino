#include <Arduino.h>
#include "SD_MMC.h"

// Pin definitions for Freenove ESP32 board
#define SD_MMC_CMD 38
#define SD_MMC_CLK 39
#define SD_MMC_D0  40

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("SD card mount test");

  // Set custom SD_MMC pins
  SD_MMC.setPins(SD_MMC_CLK, SD_MMC_CMD, SD_MMC_D0);

  // Try to mount the card
  if (!SD_MMC.begin("/sdcard", true, true, SDMMC_FREQ_DEFAULT, 5)) {
    Serial.println("Card mount failed!");
    while (1); // halt
  }

  uint8_t cardType = SD_MMC.cardType();
  if(cardType == CARD_NONE){
      Serial.println("No SD card attached");
      while (1); // halt
  }

  Serial.print("Card Type: ");
  switch(cardType) {
    case CARD_MMC:  Serial.println("MMC"); break;
    case CARD_SD:   Serial.println("SDSC"); break;
    case CARD_SDHC: Serial.println("SDHC"); break;
    default:        Serial.println("UNKNOWN"); break;
  }

  uint64_t cardSize = SD_MMC.cardSize() / (1024 * 1024);
  Serial.printf("Card Size: %llu MB\n", cardSize);

  // List files in root
  Serial.println("Listing files:");
  File root = SD_MMC.open("/");
  File file = root.openNextFile();
  while(file) {
    Serial.print(file.name());
    if(file.isDirectory()){
      Serial.println(" <DIR>");
    } else {
      Serial.print("  Size: ");
      Serial.println(file.size());
    }
    file = root.openNextFile();
  }

  Serial.println("Unmounting SD card...");
  SD_MMC.end();
  Serial.println("SD card unmounted safely. You can remove it now.");
}

void loop() {
  // Nothing to do
}
