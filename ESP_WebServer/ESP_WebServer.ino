#include <WiFi.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>

// Replace with your network credentials
const char* ssid = "fedora";
const char* password = "12345678";

AsyncWebServer server(80);
AsyncWebSocket ws("/ws");

void onWsEvent(AsyncWebSocket *server, AsyncWebSocketClient *client, AwsEventType type, void *arg, uint8_t *data, size_t len) {
  if (type == WS_EVT_CONNECT) {
    Serial.println("WebSocket client connected");
  } else if (type == WS_EVT_DISCONNECT) {
    Serial.println("WebSocket client disconnected");
  }
}

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println(WiFi.localIP());

  ws.onEvent(onWsEvent);
  server.addHandler(&ws);

  server.on("/", HTTP_GET, [](AsyncWebServerRequest *request) {
    String html = "<html><body><h1>ESP32 Serial Log</h1><div id='log' style='white-space: pre-wrap;'></div>"
                  "<script>var ws = new WebSocket('ws://' + location.hostname + '/ws');"
                  "ws.onmessage = function(event) { document.getElementById('log').innerHTML += event.data + '<br>'; };"
                  "</script></body></html>";
    request->send(200, "text/html", html);
  });

  server.begin();
}

void loop() {
  ws.cleanupClients();
  // Example logging
  Serial.println("Hello from ESP32");
  ws.textAll("Hello from ESP32");
  delay(500);
}