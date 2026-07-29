/**
 * @file sd_card_manager.h
 * @brief SD Card Manager for OpenLog Artemis
 * 
 * This class provides a high-level interface for managing SD card operations
 * including initialization, file operations, and power management.
 */

#ifndef SD_CARD_MANAGER_H
#define SD_CARD_MANAGER_H

#include "firmware_configuration.h"
#include <SPI.h>
#include "SdFat.h"
#include "RingBuf.h"
#include "time_manager.h"
#include "boot_counter.h"
#include "watchdog_manager.h"

/// RingBuf size for multi-block writes
static constexpr size_t SD_RINGBUF_SIZE = 2 * 32 * 512;

/// Enable RingBuf for buffered writes; just comment out to disable and write directly to SD card
#define USE_RINGBUFF

/**
 * @class SD_Card_Manager
 * @brief Manages SD card initialization, power, and file operations
 * 
 * Provides methods to control SD card power, initialize the card,
 * open/close files with preallocation support, and perform write operations.
 */
class SD_Card_Manager {
public:
    /**
     * @brief Initialize and start the SD card
     * 
     * Powers on the SD card, initializes SPI communication, and reads card info.
     * If initialization fails, the SD card power is turned off.
     * 
     * @return true if initialization successful, false otherwise
     * @note Can be called multiple times; returns true immediately if already initialized
     */
    bool start();
    
    /**
     * @brief Stop the SD card and turn off power
     * 
     * Closes any open files, ends SD card communication, and turns off power.
     * Safe to call multiple times or before start() is called.
     */
    void stop();
    
    /**
     * @brief Open a file and optionally preallocate space
     * 
     * Removes existing file if present, opens a new file, truncates to zero,
     * and preallocates the specified size to improve write performance.
     * 
     * @param size_bytes Number of bytes to preallocate (0 = no preallocation)
     * @param use_folders If true, organize files into BOOT_XXXXXX folders; if false, put all files in root
     * @return true if file opened successfully, false otherwise
     * @note Closes any previously open file before opening new one
     */
    bool preallocate_and_open_file(uint32_t size_bytes, bool use_folders = true);

    void generate_filename(bool use_folders = true);
    
    /**
     * @brief Sync and close the currently open file
     * 
     * Flushes all pending writes to SD card and closes the file.
     * Safe to call even if no file is open.
     */
    void close_and_sync_file();
    
    /**
     * @brief Write a buffer to the currently open file
     * 
     * Writes data from buffer to file. For best performance, use 512-byte buffers
     * (SD card sector size) and ensure file is preallocated.
     * 
     * @param buffer Pointer to data to write (must not be NULL)
     * @param size Number of bytes to write
     * @return true if all bytes written successfully, false otherwise
     * @note Returns false if no file is open or if write fails
     */
    bool write_buffer(const uint8_t* buffer, size_t size);
    
    /**
     * @brief Get direct access to SD card object
     * 
     * For advanced operations not covered by this API.
     * Use with caution as it bypasses encapsulation.
     * 
     * @return Pointer to internal SdFat object
     */
    SdFat* get_sd() { return &sd_card; }
    
    /**
     * @brief Get direct access to file object
     * 
     * For advanced operations not covered by this API.
     * Use with caution as it bypasses encapsulation.
     * 
     * @return Pointer to internal FsFile object
     */
    FsFile* get_file() { return &sd_file; }

    // Overflow accounting: records dropped WHOLE because the ring buffer was
    // full (atomic-record guard in write_buffer — partial records are never
    // written, they would byte-shift the rest of the file). Reported in the
    // periodic logging stats; reset them there if per-interval counts are
    // wanted.
    uint32_t dropped_records {0};
    uint32_t dropped_bytes {0};

private:
    SdFat sd_card;              ///< SD card filesystem object
    FsFile sd_file;             ///< Currently open file object
#ifdef USE_RINGBUFF
    RingBuf<FsFile, SD_RINGBUF_SIZE> ring_buf;  ///< Ring buffer for multi-block writes
#endif
    bool sd_initialized = false; ///< True if SD card successfully initialized
    bool file_open = false;     ///< True if a file is currently open
    char filename_buffer[128];  ///< filename is BOOT_XXXXXX/DATA_BOOT_XXXXXX_TIME_YYYYMMDDTHHMMSS.dat
};

/// Global SD card manager instance
extern SD_Card_Manager sd_card_manager;

#endif

