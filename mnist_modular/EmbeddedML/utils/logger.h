#pragma once
#include <cstdarg>
#include <cstdio>
#include <functional>

// ===== Logger levels =====
#define LOG_LEVEL_ERROR 0
#define LOG_LEVEL_WARN  1
#define LOG_LEVEL_INFO  2
#define LOG_LEVEL_DEBUG 3

// ---------------- Sink type ----------------
using LogSink = std::function<void(const char* fmt, va_list args)>;

// ---------------- Default sinks ----------------
inline void default_printf_sink(const char* fmt, va_list args) {
    vprintf(fmt, args);
}

#if defined(ARDUINO)
#include <Arduino.h>
inline void serial_printf_sink(const char* fmt, va_list args) {
    Serial.vprintf(fmt, args);
}
#endif

// ---------------- Global config ----------------
inline LogSink& get_log_sink() {
    static LogSink sink = default_printf_sink; // default = printf
    return sink;
}

inline void set_log_sink(LogSink sink) {
    get_log_sink() = sink;
}

inline int& get_log_level() {
    static int level = LOG_LEVEL_INFO; // default runtime log level
    return level;
}

inline void set_log_level(int level) {
    get_log_level() = level;
}

// ---------------- Internal log helper ----------------
inline void log_message(int level, int threshold,
                        const char* prefix, const char* fmt, ...) {
    if (level > threshold) return;

    char buffer[256];
    int n = snprintf(buffer, sizeof(buffer), "%s", prefix);

    va_list args;
    va_start(args, fmt);

    // print prefix
    get_log_sink()(buffer, args);

    // print user message
    get_log_sink()(fmt, args);

    va_end(args);

    // newline
    va_list nl;
    get_log_sink()("\n", nl);
}

// ---------------- Logging macros ----------------
#define LOG_ERROR(...) do { \
    if (get_log_level() >= LOG_LEVEL_ERROR) { \
        log_message(LOG_LEVEL_ERROR, get_log_level(), "[ERROR] ", __VA_ARGS__); \
    } \
} while(0)

#define LOG_WARN(...) do { \
    if (get_log_level() >= LOG_LEVEL_WARN) { \
        log_message(LOG_LEVEL_WARN, get_log_level(), "[WARN] ", __VA_ARGS__); \
    } \
} while(0)

#define LOG_INFO(...) do { \
    if (get_log_level() >= LOG_LEVEL_INFO) { \
        log_message(LOG_LEVEL_INFO, get_log_level(), "[INFO] ", __VA_ARGS__); \
    } \
} while(0)

#define LOG_DEBUG(...) do { \
    if (get_log_level() >= LOG_LEVEL_DEBUG) { \
        log_message(LOG_LEVEL_DEBUG, get_log_level(), "[DEBUG] ", __VA_ARGS__); \
    } \
} while(0)

// ---------------- Progress bar ----------------
inline void LOG_PROGRESS(int current, int total, const char* metric) {
    if (total <= 0) total = 1;
    int percent = (current * 100) / total;
    int barWidth = 50;
    int pos = (percent * barWidth) / 100;

    char buf[256];
    int n = snprintf(buf, sizeof(buf), "Progress: [");
    for (int i = 0; i < barWidth; i++) {
        n += snprintf(buf + n, sizeof(buf) - n, "%c",
                      (i < pos ? '=' : (i == pos ? '>' : ' ')));
    }
    snprintf(buf + n, sizeof(buf) - n, "] %d%% (%d/%d) %s\r\n",
             percent, current, total, metric);

    va_list empty;
    get_log_sink()(buf, empty);
}
