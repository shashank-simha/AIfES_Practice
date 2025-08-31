#include <SD_MMC.h>

#define SD_MMC_CMD 38 //Please do not modify it.
#define SD_MMC_CLK 39 //Please do not modify it.
#define SD_MMC_D0  40 //Please do not modify it.

// ---------------- Config ----------------
#define IMG_SIZE          (28 * 28)  // one MNIST image = 784 bytes
#define TRAIN_CHUNK_SIZE  2996      // number of train images per chunk
#define TEST_CHUNK_SIZE   996       // number of test images per chunk

#define TRAIN_BUF_SIZE  (TRAIN_CHUNK_SIZE * IMG_SIZE)
#define TEST_BUF_SIZE   (TEST_CHUNK_SIZE * IMG_SIZE)

// Buffers (allocate in PSRAM if available)
uint8_t *train_buffer;
uint8_t *test_buffer;

void setup() {
  Serial.begin(115200);
  delay(1000);

  SD_MMC.setPins(SD_MMC_CLK, SD_MMC_CMD, SD_MMC_D0);
    if (!SD_MMC.begin("/sdcard", true, true, SDMMC_FREQ_DEFAULT, 5)) {
      Serial.println("Card Mount Failed");
      return;
    }
    uint8_t cardType = SD_MMC.cardType();
    if(cardType == CARD_NONE){
        Serial.println("No SD_MMC card attached");
        return;
    }

    Serial.print("SD_MMC Card Type: ");
    if(cardType == CARD_MMC){
        Serial.println("MMC");
    } else if(cardType == CARD_SD){
        Serial.println("SDSC");
    } else if(cardType == CARD_SDHC){
        Serial.println("SDHC");
    } else {
        Serial.println("UNKNOWN");
    }

    uint64_t cardSize = SD_MMC.cardSize() / (1024 * 1024);
    Serial.printf("SD_MMC Card Size: %lluMB\n", cardSize);

  // List files
  listDir(SD_MMC, "/", 1);

  // --- Allocate buffers in PSRAM if available ---
  if (psramFound()) {
    Serial.println("PSRAM detected. Allocating buffers...");
    train_buffer = (uint8_t *)ps_malloc(TRAIN_BUF_SIZE);
    test_buffer  = (uint8_t *)ps_malloc(TEST_BUF_SIZE);
  } else {
    Serial.println("No PSRAM! Using normal heap.");
    train_buffer = (uint8_t *)malloc(TRAIN_BUF_SIZE);
    test_buffer  = (uint8_t *)malloc(TEST_BUF_SIZE);
  }

  if (!train_buffer || !test_buffer) {
    Serial.println("Failed to allocate buffers!");
    return;
  }

  // --- Test read ---
  readChunk("/mnist_chunks/train_images_chunk0.bin", train_buffer, TRAIN_BUF_SIZE);
  readChunk("/mnist_chunks/test_images_chunk0.bin", test_buffer, TEST_BUF_SIZE);
}

void loop() {
  // Example: swap buffers every 5 seconds
  static uint32_t lastSwap = 0;
  static int chunkIdx = 0;

  if (millis() - lastSwap > 5000) {
    char fname[64];

    snprintf(fname, sizeof(fname), "/mnist_chunks/train_images_chunk%d.bin", chunkIdx);
    readChunk(fname, train_buffer, TRAIN_BUF_SIZE);

    snprintf(fname, sizeof(fname), "/mnist_chunks/test_images_chunk%d.bin", chunkIdx);
    readChunk(fname, test_buffer, TEST_BUF_SIZE);

    chunkIdx = (chunkIdx + 1) % 3;  // rotate among 3 chunks for demo
    lastSwap = millis();
  }
}

// ---------------- Helper Functions ----------------
void listDir(fs::FS &fs, const char *dirname, uint8_t levels) {
  Serial.printf("Listing directory: %s\n", dirname);

  File root = fs.open(dirname);
  if (!root || !root.isDirectory()) {
    Serial.println("Failed to open directory");
    return;
  }

  File file = root.openNextFile();
  while (file) {
    if (file.isDirectory()) {
      Serial.print("  DIR : ");
      Serial.println(file.name());
      if (levels) listDir(fs, file.name(), levels - 1);
    } else {
      Serial.print("  FILE: ");
      Serial.print(file.name());
      Serial.print("  SIZE: ");
      Serial.println(file.size());
    }
    file = root.openNextFile();
  }
}

void readChunk(const char *filename, uint8_t *buffer, size_t bufsize) {
  uint32_t start = millis();

  File f = SD_MMC.open(filename, FILE_READ);
  if (!f) {
    Serial.printf("Failed to open %s\n", filename);
    return;
  }

  size_t bytesRead = f.read(buffer, bufsize);
  f.close();

  uint32_t elapsed = millis() - start;
  Serial.printf("Read %s: %u bytes in %u ms\n", filename, bytesRead, elapsed);
}
