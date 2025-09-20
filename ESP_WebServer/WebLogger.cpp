// WebLogger.cpp
#include "WebLogger.h"

WebLogger::WebLogger(AsyncWebSocket& ws) : ws(ws) {}

void WebLogger::log(const String& message) {
  String timedMsg = getTimestamp() + message;
  ws.textAll(timedMsg);
}

String WebLogger::getLoggerUI() {
  return R"rawliteral(
  <div id='log-container' style='position: fixed; bottom: 0; left: 0; width: 100%; height: 200px; overflow-y: auto; resize: vertical; background: black;'>
  <pre id='log' style='color: white; font-family: monospace; padding: 10px; margin: 0; white-space: pre-wrap;'></pre>
  </div>
  <script>
  var ws = new WebSocket('ws://' + location.hostname + '/ws');
  ws.onmessage = function(event) {
    var log = document.getElementById('log');
    var isAtBottom = Math.abs(log.scrollHeight - log.scrollTop - log.clientHeight) < 1;
    log.innerHTML += event.data + '\n';
    if (isAtBottom) log.scrollTop = log.scrollHeight;
  };
  </script></body></html>)rawliteral";
}

String WebLogger::getTimestamp() {
  unsigned long ms = millis();
  unsigned long seconds = ms / 1000;
  unsigned long minutes = seconds / 60;
  unsigned long hours = minutes / 60;
  seconds %= 60;
  minutes %= 60;
  return String(hours) + ":" + (minutes < 10 ? "0" : "") + String(minutes) + ":" + (seconds < 10 ? "0" : "") + String(seconds) + " -> ";
}
