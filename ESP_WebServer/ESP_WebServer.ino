#include "ESPWebLogger.h"

// Replace with your network credentials
const char* ssid = "fedora";
const char* password = "12345678";

ESPWebLogger logger;

void button1Pressed() {
  logger.log("Button 1 pressed!");
  // User code here
}

void button2Pressed() {
  logger.log("Button 2 pressed!");
  // User code here
}

void button3Pressed() {
  logger.log("Button 3 pressed!");
  // User code here
}

void setup() {
  logger.addButton("Button1", button1Pressed);
  logger.addButton("Button2", button2Pressed);
  logger.addButton("Button3", button3Pressed);
  logger.begin(ssid, password);
}

void loop() {
  logger.handle();
  delay(2000);
  logger.log("Hello from ESP32");
}