#pragma once
#include <cstdio>
#include <Arduino.h>

// ===== Logger configuration =====
#ifndef LOG_LEVEL
#define LOG_LEVEL LOG_LEVEL_INFO
#endif

#define LOG_LEVEL_ERROR 0
#define LOG_LEVEL_WARN  1
#define LOG_LEVEL_INFO  2
#define LOG_LEVEL_DEBUG 3

/**
 * @file logger.h
 * @brief Cross-platform logging utilities.
 *
 * By default, logs use printf. On embedded platforms (e.g. Arduino),
 * users can override LOG_INFO, LOG_WARN, LOG_ERROR, or LOG_PROGRESS
 * before including this header, e.g.:
 *
 * #define LOG_INFO(...)    Serial.printf("[INFO] " __VA_ARGS__ "\n")
 * #define LOG_WARN(...)    Serial.printf("[WARN] " __VA_ARGS__ "\n")
 * #define LOG_ERROR(...)   Serial.printf("[ERROR] " __VA_ARGS__ "\n")
 * #define LOG_PROGRESS(...) MyCustomProgressBar(__VA_ARGS__)
 * #include "logger.h"
 *
 * Note: the override only reflects in the file/unit that makes the changes
 * For global override, feel free to modify the macros here
 */

/* ---------------- Default logging macros ---------------- */

#ifndef LOG_ERROR
#define LOG_ERROR(...) do { \
    if (LOG_LEVEL >= LOG_LEVEL_ERROR) { \
        Serial.printf("[ERROR] "); Serial.printf(__VA_ARGS__); Serial.printf("\n"); \
    } \
} while(0)
#endif

#ifndef LOG_WARN
#define LOG_WARN(...) do { \
    if (LOG_LEVEL >= LOG_LEVEL_WARN) { \
        Serial.printf("[WARN] "); Serial.printf(__VA_ARGS__); Serial.printf("\n"); \
    } \
} while(0)
#endif

#ifndef LOG_INFO
#define LOG_INFO(...) do { \
    if (LOG_LEVEL >= LOG_LEVEL_INFO) { \
        Serial.printf("[INFO] "); Serial.printf(__VA_ARGS__); Serial.printf("\n"); \
    } \
} while(0)
#endif

#ifndef LOG_DEBUG
#define LOG_DEBUG(...) do { \
    if (LOG_LEVEL >= LOG_LEVEL_DEBUG) { \
        Serial.printf("[DEBUG] "); Serial.printf(__VA_ARGS__); Serial.printf("\n"); \
    } \
} while(0)
#endif


/* ---------------- Progress bar (default impl) ---------------- */

/**
 * @brief Show a simple textual progress bar.
 *
 * Users can override LOG_PROGRESS before including this header.
 * Default implementation uses printf and draws a fixed-width bar.
 *
 * @param current Current progress count
 * @param total   Total count
 * @param metric  Description string
 */
#ifndef LOG_PROGRESS
inline void LOG_PROGRESS(int current, int total, const char* metric) {
    if (total <= 0) total = 1; // avoid divide-by-zero
    int percent = (current * 100) / total;
    int barWidth = 50;  // number of characters in the bar
    int pos = (percent * barWidth) / 100;

    Serial.printf("Progress: [");
    for (int i = 0; i < barWidth; i++) {
        if (i < pos) Serial.printf("=");
        else if (i == pos) Serial.printf(">");
        else Serial.printf(" ");
    }
    Serial.printf("] %d%% (%d/%d) %s\r", percent, current, total, metric);
    // fflush(stdout); // ensure progress is flushed immediately
}
#endif
