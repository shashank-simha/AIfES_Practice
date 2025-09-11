#pragma once
#include "FileAdapter.h"
#include <SD_MMC.h>
#include <FS.h>

/**
 * @brief SD_MMC implementation of FileAdapter for Arduino/ESP32.
 */
class SDMMCFileAdapter : public FileAdapter {
public:
    SDMMCFileAdapter() : file_open(false) {}
    ~SDMMCFileAdapter() override { close(); }

    /**
     * @brief Open a file from SD_MMC.
     *
     * Maps FileAdapter::OpenMode to Arduino FS modes.
     */
    bool open(const std::string& path, OpenMode mode = OpenMode::READ) override {
        close(); // close any previously open file

        switch (mode) {
            case OpenMode::READ:
                file = SD_MMC.open(path.c_str(), FILE_READ);
                break;
            case OpenMode::WRITE:
                file = SD_MMC.open(path.c_str(), FILE_WRITE);
                break;
            case OpenMode::APPEND:
                // Arduino FS does not directly support append, so emulate:
                file = SD_MMC.open(path.c_str(), FILE_APPEND);
                if (!file) {
                    // fallback: open write and seek to end
                    file = SD_MMC.open(path.c_str(), FILE_WRITE);
                    if (file) file.seek(file.size());
                }
                break;
            case OpenMode::READWRITE:
                // No direct support: open in write, but allow read ops too.
                // FILE_WRITE allows both read & write on ESP32 FS.
                file = SD_MMC.open(path.c_str(), FILE_WRITE);
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

private:
    File file;
    bool file_open;
};
