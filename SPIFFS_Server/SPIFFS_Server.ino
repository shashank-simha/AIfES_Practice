#include <WiFi.h>
#include <SPIFFS.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>

const char* ssid     = "fedora";
const char* password = "12345678";
AsyncWebServer server(80);

// HTML page with file upload form, file list, and delete buttons
const char index_html[] PROGMEM = R"rawliteral(
<!DOCTYPE html><html>
<head>
  <title>SPIFFS File Server</title>
  <style>
    body { font-family: Arial; }
    li { margin: 6px 0; }
    button { margin-left: 12px; }
  </style>
</head>
<body>
  <h2>Upload File</h2>
  <form method='POST' action='/upload' enctype='multipart/form-data'>
    <input type='file' name='file'><input type='submit' value='Upload'>
  </form>
  <h2>Files:</h2><ul>
    %x%
  </ul>
  <script>
    function deleteFile(filename) {
      if (confirm("Delete " + filename + "?")) {
        fetch('/delete?name=' + filename).then(() => location.reload());
      }
    }
  </script>
</body></html>
)rawliteral";

String listFiles() {
  String fileList;
  File root = SPIFFS.open("/");
  File file = root.openNextFile();
  while (file) {
    fileList += "<li><a href='";
    fileList += file.name();
    fileList += "'>";
    fileList += file.name();
    fileList += "</a> (";
    fileList += file.size();
    fileList += " bytes)";
    fileList += " <button onclick=\"deleteFile('";
    fileList += file.name();
    fileList += "')\">Delete</button></li>";
    file = root.openNextFile();
  }
  return fileList;
}

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500); Serial.print('.');
  }
  Serial.printf("\nConnected! IP: %s\n", WiFi.localIP().toString().c_str());

  if (!SPIFFS.begin(true)) {
    Serial.println("SPIFFS Mount Failed");
    return;
  }

  // Root page
  server.on("/", HTTP_GET, [](AsyncWebServerRequest *request){
    String page = index_html;
    page.replace("%x%", listFiles());
    request->send(200, "text/html", page);
  });

  // Handle file upload
  server.on("/upload", HTTP_POST, [](AsyncWebServerRequest *req){
    req->send(200, "text/plain", "Upload complete");
  }, [](AsyncWebServerRequest *req, String filename, size_t index, uint8_t *data, size_t len, bool final){
    static File uploadFile;
    if (index == 0){
      Serial.printf("Upload Start: %s\n", filename.c_str());
      uploadFile = SPIFFS.open("/"+filename, FILE_WRITE);
    }
    uploadFile.write(data, len);
    if (final) {
      Serial.printf("Upload End: %s, %u bytes\n", filename.c_str(), index + len);
      uploadFile.close();
    }
  });

  // Handle file delete
  server.on("/delete", HTTP_GET, [](AsyncWebServerRequest *request){
    if (request->hasParam("name")) {
      String filename = request->getParam("name")->value();

      // Ensure filename starts with '/'
      if (!filename.startsWith("/")) {
        filename = "/" + filename;
      }

      if (SPIFFS.exists(filename)) {
        SPIFFS.remove(filename);
        Serial.printf("Deleted: %s\n", filename.c_str());
        request->send(200, "text/plain", "Deleted " + filename);
        return;
      } else {
        request->send(404, "text/plain", "File not found");
        return;
      }
    }
    request->send(400, "text/plain", "Missing filename");
  });

  // Serve static files
  server.serveStatic("/", SPIFFS, "/");

  server.begin();
}

void loop() {}
