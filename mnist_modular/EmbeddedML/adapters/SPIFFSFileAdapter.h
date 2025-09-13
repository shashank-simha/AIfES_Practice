#pragma once
#include <SPIFFS.h>
#include "../core/FileAdapter.h"

/**
 * @brief SPIFFS implementation of FileAdapter for Arduino/ESP32.
 */
class SPIFFSFileAdapter : public FileAdapter {
public:
    SPIFFSFileAdapter() : file_open(false) {}
    ~SPIFFSFileAdapter() override { close(); }

    /**
     * @brief Open a file from SPIFFS.
     */
    bool open(const std::string& path, OpenMode mode = OpenMode::READ) override {
        close(); // close any previously open file

        switch (mode) {
            case OpenMode::READ:
                file = SPIFFS.open(path.c_str(), FILE_READ);
                break;
            case OpenMode::WRITE:
                file = SPIFFS.open(path.c_str(), FILE_WRITE);
                break;
            case OpenMode::APPEND:
                // SPIFFS supports append mode directly
                file = SPIFFS.open(path.c_str(), FILE_APPEND);
                if (!file) {
                    // fallback: open write and seek to end
                    file = SPIFFS.open(path.c_str(), FILE_WRITE);
                    if (file) file.seek(file.size());
                }
                break;
            case OpenMode::READWRITE:
                // SPIFFS does not have explicit readwrite, open in write mode
                file = SPIFFS.open(path.c_str(), FILE_WRITE);
                break;
        }

        file_open = file ? true : false;
        return file_open;
    }

    size_t read(uint8_t* buffer, size_t size) override {
        if (!file_open) return 0;
        return file.read(buffer, size);
    }

    size_t write(const uint8_t* buffer, size_t size) override {
        if (!file_open) return 0;
        return file.write(buffer, size);
    }

    void close() override {
        if (file_open) {
            file.close();
            file_open = false;
        }
    }

    bool flush() override {
        if (file_open) {
            file.flush();
        }
        return true;
    }

    size_t size() const override {
        if (!file_open) return 0;
        return file.size();
    }

    bool seek(size_t offset) override {
        if (!file_open) return false;
        return file.seek(offset);
    }

    size_t tell() const override {
        if (!file_open) return 0;
        return file.position();
    }

    bool remove(const char* path) override {
        return SPIFFS.remove(path);
    }

    bool rename(const char* old_path, const char* new_path) override {
        return SPIFFS.rename(old_path, new_path);
    }

private:
    File file;
    bool file_open;
};
