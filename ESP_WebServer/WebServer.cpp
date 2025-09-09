// WebServer.cpp
#include "WebServer.h"
#include "WebLogger.h"
#include <base64.h>

WebServer::WebServer() : server(80), ws("/ws"), logger(nullptr) {}

void WebServer::begin(const char* ssid, const char* password, const char* html, const String& loggerUI, WebLogger* logger) {
  this->logger = logger;
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    if (logger) logger->log("Connecting to WiFi...");
  }

  ws.onEvent([this](AsyncWebSocket *s, AsyncWebSocketClient *c, AwsEventType t, void *a, uint8_t *d, size_t l) { onWsEvent(s, c, t, a, d, l); });
  server.addHandler(&ws);

  server.on("/", HTTP_GET, [html, loggerUI](AsyncWebServerRequest *request) {
    String page = String(html);
    // Replace </body> with loggerUI + </body> to ensure correct placement
    page.replace("</body>", loggerUI + "</body>");
    request->send(200, "text/html", page);
  });

  server.on("/upload", HTTP_POST, [this, logger](AsyncWebServerRequest *request) {
    if (logger) logger->log("Upload completed");
    request->send(200, "text/plain", "Upload complete");
  }, [this](auto* r, auto f, auto i, auto* d, auto l, auto fin) { handleUpload(r, f, i, d, l, fin); });

  server.on("/image", HTTP_GET, [this, logger](AsyncWebServerRequest *request) {
    if (imageBase64.length() > 0) {
      request->send(200, "image/jpeg", imageBase64);
    } else {
      if (logger) logger->log("No image available");
      request->send(404, "text/plain", "No image available");
    }
  });

  server.on("/action", HTTP_GET, [this](AsyncWebServerRequest *request) {
    if (request->hasParam("name")) {
      String name = request->getParam("name")->value();
      if (actionCallbacks.count(name)) {
        actionCallbacks[name]();
      }
    }
    request->send(200, "text/plain", "OK");
  });

  server.begin();
}

void WebServer::addAction(const String& name, void (*callback)()) {
  actionCallbacks[name] = callback;
}

void WebServer::handle() {
  ws.cleanupClients();
}

void WebServer::onWsEvent(AsyncWebSocket *server, AsyncWebSocketClient *client, AwsEventType type, void *arg, uint8_t *data, size_t len) {
  if (type == WS_EVT_CONNECT && logger) {
    logger->log("WebSocket client connected");
  } else if (type == WS_EVT_DISCONNECT && logger) {
    logger->log("WebSocket client disconnected");
  }
}

void WebServer::handleUpload(AsyncWebServerRequest *request, String filename, size_t index, uint8_t *data, size_t len, bool final) {
  static String tempBuffer;
  if (!index) {
    tempBuffer = "";
    if (logger) logger->log("UploadStart: " + filename);
  }
  for (size_t i = 0; i < len; i++) {
    tempBuffer += (char)data[i];
  }
  if (final) {
    imageBase64 = base64::encode((uint8_t*)tempBuffer.c_str(), tempBuffer.length());
    if (logger) logger->log("Image size: " + String(tempBuffer.length()) + " bytes");
    tempBuffer = "";
  }
}