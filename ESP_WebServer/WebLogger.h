// WebLogger.h
#ifndef WEB_LOGGER_H
#define WEB_LOGGER_H

#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>

class WebLogger {
public:
  WebLogger(AsyncWebSocket& ws);
  void log(const String& message);
  String getLoggerUI();

private:
  AsyncWebSocket& ws;
  String getTimestamp();
};

#endif