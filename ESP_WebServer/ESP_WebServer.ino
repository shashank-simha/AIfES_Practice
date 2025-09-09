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

void handleUpload(AsyncWebServerRequest *request, String filename, size_t index, uint8_t *data, size_t len, bool final) {
  if (!index) {
    Serial.printf("UploadStart: %s\n", filename.c_str());
  }
  if (final) {
    size_t fileSize = index + len;
    String msg = "Image size: " + String(fileSize) + " bytes";
    Serial.println(msg);
    ws.textAll(msg);
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
    String html = "<html><body><h1>ESP32 Serial Log</h1>"
                  "<form method='POST' enctype='multipart/form-data' action='/upload'>"
                  "<input type='file' name='image' accept='image/*'>"
                  "<button type='submit'>Upload Image</button></form>"
                  "<div id='log-container' style='position: fixed; bottom: 0; left: 0; width: 100%; height: 200px; overflow: auto; resize: vertical; background: black;'>"
                  "<pre id='log' style='color: white; font-family: monospace; padding: 10px; margin: 0; white-space: pre-wrap;'></pre></div>"
                  "<script>var ws = new WebSocket('ws://' + location.hostname + '/ws');"
                  "ws.onmessage = function(event) { var log = document.getElementById('log'); log.innerHTML += event.data + '\\n'; log.scrollTop = log.scrollHeight; };"
                  "</script></body></html>";
    request->send(200, "text/html", html);
  });

  server.on("/upload", HTTP_POST, [](AsyncWebServerRequest *request) {
    request->send(200, "text/plain", "Upload complete");
  }, handleUpload);

  server.begin();
}

void loop() {
  ws.cleanupClients();
  // Example logging
  Serial.println("Hello from ESP32");
  ws.textAll("Hello from ESP32");
  delay(2000);
}