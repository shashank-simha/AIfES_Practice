// WebLogger.ino
#include "WebServer.h"
#include "WebLogger.h"

// Replace with your network credentials
const char* ssid = "fedora";
const char* password = "12345678";

const char* html = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
<meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:;">
</head>
<body>
<h1>ESP32 Control Panel</h1>
<input type='file' id='image' accept='image/*'>
<button onclick='uploadImage()'>Upload Image</button>
<button onclick='fetch("/action?name=Button1")'>Button1</button>
<button onclick='fetch("/action?name=Button2")'>Button2</button>
<button onclick='fetch("/action?name=Button3")'>Button3</button>
<img id='uploadedImage' src='' style='display: none; max-width: 100%; margin-top: 10px;'>
<script>
async function uploadImage() {
  const fileInput = document.getElementById('image');
  const file = fileInput.files[0];
  if (!file) {
    console.error('No file selected');
    return;
  }
  const formData = new FormData();
  formData.append('image', file);
  try {
    const response = await fetch('/upload', { method: 'POST', body: formData });
    if (response.ok) {
      const imgResponse = await fetch('/image?' + new Date().getTime());
      if (imgResponse.ok) {
        const imgData = await imgResponse.text();
        const img = document.getElementById('uploadedImage');
        img.src = 'data:image/jpeg;base64,' + imgData;
        img.style.display = 'block';
        console.log('Image loaded successfully');
      } else {
        console.error('Failed to load image:', imgResponse.status);
      }
    } else {
      console.error('Upload failed:', response.status);
    }
  } catch (error) {
    console.error('Upload error:', error);
  }
}
</script>
</body>
</html>
)rawliteral";

WebServer webServer;
WebLogger* logger;

void custom_log(const String& message) {
  Serial.println(message);
  if (logger) logger->log(message);
}

void button1Pressed() {
  custom_log("Button 1 pressed!");
  // User code here
}

void button2Pressed() {
  custom_log("Button 2 pressed!");
  // User code here
}

void button3Pressed() {
  custom_log("Button 3 pressed!");
  // User code here
}

void setup() {
  Serial.begin(115200);
  logger = new WebLogger(webServer.ws);
  webServer.addAction("Button1", button1Pressed);
  webServer.addAction("Button2", button2Pressed);
  webServer.addAction("Button3", button3Pressed);
  webServer.begin(ssid, password, html, logger->getLoggerUI(), logger);
  custom_log("WiFi Connected: " + WiFi.localIP().toString());
}

void loop() {
  webServer.handle();
  delay(2000);
  custom_log("Hello from ESP32");
}