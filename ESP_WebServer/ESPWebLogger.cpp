#include "ESPWebLogger.h"

ESPWebLogger::ESPWebLogger() : server(80), ws("/ws") {}

void ESPWebLogger::begin(const char* ssid, const char* password) {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    log("Connecting to WiFi...");
  }
  log(WiFi.localIP().toString());

  ws.onEvent([this](AsyncWebSocket *s, AsyncWebSocketClient *c, AwsEventType t, void *a, uint8_t *d, size_t l) { onWsEvent(s, c, t, a, d, l); });
  server.addHandler(&ws);

  server.on("/", HTTP_GET, [this](AsyncWebServerRequest *request) {
    String html = R"rawliteral(
    <!DOCTYPE html>
    <html><body><h1>ESP32 Serial Log</h1>
    <input type='file' id='image' accept='image/*'>
    <button onclick='uploadImage()'>Upload Image</button>)rawliteral";
    html += htmlButtons;
    html += R"rawliteral(
    <div id='log-container' style='position: fixed; bottom: 0; left: 0; width: 100%; height: 200px; overflow: auto; resize: vertical; background: black;'>
    <pre id='log' style='color: white; font-family: monospace; padding: 10px; margin: 0; white-space: pre-wrap;'></pre>
    </div>
    <script>
    var ws = new WebSocket('ws://' + location.hostname + '/ws');
    ws.onmessage = function(event) { 
      var log = document.getElementById('log'); 
      log.innerHTML += event.data + '\n'; 
      log.scrollTop = log.scrollHeight; 
    };
    async function uploadImage() {
      const fileInput = document.getElementById('image');
      const file = fileInput.files[0];
      if (!file) return;
      const formData = new FormData();
      formData.append('image', file);
      await fetch('/upload', { method: 'POST', body: formData });
    }
    </script></body></html>)rawliteral";
    request->send(200, "text/html", html);
  });

  server.on("/upload", HTTP_POST, [](AsyncWebServerRequest *request) {
    request->send(200, "text/plain", "Upload complete");
  }, [this](auto* r, auto f, auto i, auto* d, auto l, auto fin) { handleUpload(r, f, i, d, l, fin); });

  server.on("/action", HTTP_GET, [this](AsyncWebServerRequest *request) {
    if (request->hasParam("btn")) {
      String btn = request->getParam("btn")->value();
      if (buttonCallbacks.count(btn)) {
        buttonCallbacks[btn]();
      }
    }
    request->send(200, "text/plain", "OK");
  });

  server.begin();
}

void ESPWebLogger::addButton(const char* label, void (*callback)()) {
  String btnLabel = label;
  htmlButtons += "<button onclick='fetch(\"/action?btn=" + btnLabel + "\")'>" + btnLabel + "</button>";
  buttonCallbacks[btnLabel] = callback;
}

void ESPWebLogger::log(const String& message) {
  Serial.println(message);
  ws.textAll(message);
}

void ESPWebLogger::handle() {
  ws.cleanupClients();
}

void ESPWebLogger::onWsEvent(AsyncWebSocket *server, AsyncWebSocketClient *client, AwsEventType type, void *arg, uint8_t *data, size_t len) {
  if (type == WS_EVT_CONNECT) {
    log("WebSocket client connected");
  } else if (type == WS_EVT_DISCONNECT) {
    log("WebSocket client disconnected");
  }
}

void ESPWebLogger::handleUpload(AsyncWebServerRequest *request, String filename, size_t index, uint8_t *data, size_t len, bool final) {
  if (!index) {
    log("UploadStart: " + filename);
  }
  if (final) {
    size_t fileSize = index + len;
    log("Image size: " + String(fileSize) + " bytes");
  }
}