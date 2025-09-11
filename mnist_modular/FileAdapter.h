#pragma once
#include <cstddef>
#include <string>
#include <cstdint>

/**
 * @brief Abstract interface for platform-agnostic file access.
 *
 * Provides APIs for opening, reading, writing, and closing files.
 * Implementations may wrap Arduino File, FatFs, POSIX FILE*, etc.
 */
class FileAdapter {
public:
    /**
     * @brief File open modes.
     */
    enum class OpenMode {
        READ,       ///< Open for reading
        WRITE,      ///< Open for writing (truncate if exists)
        APPEND,     ///< Open for appending (create if not exists)
        READWRITE   ///< Open for both reading and writing
    };

    FileAdapter() = default;                      ///< Default constructor
    virtual ~FileAdapter() = default;             ///< Virtual destructor

    FileAdapter(const FileAdapter&) = default;    ///< Copy constructor
    FileAdapter& operator=(const FileAdapter&) = default; ///< Copy assignment
    FileAdapter(FileAdapter&&) = default;         ///< Move constructor
    FileAdapter& operator=(FileAdapter&&) = default; ///< Move assignment

    /**
     * @brief Open a file at the given path.
     * @param path File path.
     * @param mode File open mode (READ, WRITE, etc.)
     * @return true if successful, false otherwise.
     */
    virtual bool open(const std::string& path, OpenMode mode = OpenMode::READ) = 0;

    /**
     * @brief Read data from the currently open file.
     * @param buffer Destination buffer.
     * @param size Number of bytes to read.
     * @return Number of bytes actually read.
     */
    virtual size_t read(uint8_t* buffer, size_t size) = 0;

    /**
     * @brief Write data to the currently open file.
     * @param buffer Source buffer.
     * @param size Number of bytes to write.
     * @return Number of bytes actually written.
     */
    virtual size_t write(const uint8_t* buffer, size_t size) = 0;

    /**
     * @brief Close the currently open file.
     */
    virtual void close() = 0;

    /**
     * @brief Flush buffered writes to disk (if applicable).
     * @return true if flush successful, false otherwise.
     *
     * Default implementation may return false if not supported.
     */
    virtual bool flush() { return false; }

    /**
     * @brief Get size of the currently open file.
     * @return File size in bytes, or 0 if not supported or file not open.
     */
    virtual size_t size() const { return 0; }

    /**
     * @brief Seek to a specific position in the file.
     * @param offset Position to seek to.
     * @return true if seek successful, false otherwise.
     *
     * Default implementation may return false if not supported.
     */
    virtual bool seek(size_t offset) { return false; }

    /**
     * @brief Get current position in the file.
     * @return Current position in bytes, or 0 if not supported.
     */
    virtual size_t tell() const { return 0; }

    /**
     * @brief Remove (delete) a file from the filesystem.
     * @param path Path of the file to remove.
     * @return true if the file was successfully removed, false otherwise.
     *
     * Default implementation may return false if the underlying filesystem
     * does not support file removal.
     */
    virtual bool remove(const char* path) { return false; }

    /**
     * @brief Rename or move a file within the filesystem.
     * @param old_path Current path of the file.
     * @param new_path New desired path of the file.
     * @return true if the file was successfully renamed, false otherwise.
     *
     * Default implementation may return false if the underlying filesystem
     * does not support renaming.
     */
    virtual bool rename(const char* old_path, const char* new_path) { return false; }
};
