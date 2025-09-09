// WebServer.h
#ifndef WEB_SERVER_H
#define WEB_SERVER_H

#include <WiFi.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>
#include <map>

class WebLogger; // Forward declaration

class WebServer {
public:
  AsyncWebServer server;
  AsyncWebSocket ws;
  std::map<String, void(*)()> actionCallbacks;
  String imageBase64;
  WebLogger* logger; // Add logger reference

  WebServer();
  void begin(const char* ssid, const char* password, const char* html, const String& loggerUI, WebLogger* logger);
  void addAction(const String& name, void (*callback)());
  void handle();

private:
  void onWsEvent(AsyncWebSocket *server, AsyncWebSocketClient *client, AwsEventType type, void *arg, uint8_t *data, size_t len);
  void handleUpload(AsyncWebServerRequest *request, String filename, size_t index, uint8_t *data, size_t len, bool final);
};

#endif