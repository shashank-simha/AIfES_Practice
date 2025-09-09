#ifndef ESP_WEB_LOGGER_H
#define ESP_WEB_LOGGER_H

#include <WiFi.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>
#include <map>

class ESPWebLogger {
public:
  AsyncWebServer server;
  AsyncWebSocket ws;
  std::map<String, void(*)()> buttonCallbacks;
  String htmlButtons;

  ESPWebLogger();
  void begin(const char* ssid, const char* password);
  void addButton(const char* label, void (*callback)());
  void log(const String& message);
  void handle();

private:
  void onWsEvent(AsyncWebSocket *server, AsyncWebSocketClient *client, AwsEventType type, void *arg, uint8_t *data, size_t len);
  void handleUpload(AsyncWebServerRequest *request, String filename, size_t index, uint8_t *data, size_t len, bool final);
  String getTimestamp();
};

#endif