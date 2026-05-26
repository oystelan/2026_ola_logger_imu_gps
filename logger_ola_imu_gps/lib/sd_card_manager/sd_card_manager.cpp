/**
 * @file sd_card_manager.cpp
 * @brief Implementation of SD Card Manager
 */

#include "sd_card_manager.h"

// RAII helper: while in scope, mask the IMU CTIMER interrupt (CTIMER_IRQn).
// The IMU ISR talks to the IMU over the same SPI bus we use for the SD card,
// so any SD SPI transaction must run uninterrupted. After this guard goes
// out of scope, the CTIMER interrupt is re-enabled and any pending IMU tick
// gets serviced.
struct SDIrqGuard {
    SDIrqGuard()  { NVIC_DisableIRQ(CTIMER_IRQn); }
    ~SDIrqGuard() { NVIC_EnableIRQ(CTIMER_IRQn);  }
};

bool toogle_state_led = false;

// Global instance
SD_Card_Manager sd_card_manager;

/**
 * @brief Turn on SD card power
 * 
 * Sets the SD_PWR pin LOW (active low) to enable SD card power,
 * then waits for the card to stabilize.
 */
void microSDPowerOn()
{
    delay(10);
    SERIAL_USB->println(F("Turning SD card power ON..."));
    pinMode(SD_PWR, OUTPUT);
    digitalWrite(SD_PWR, LOW);
    delay(1000);  // Wait for SD card to power up
    wdt.restart();
}

/**
 * @brief Turn off SD card power
 * 
 * Sets the SD_PWR pin HIGH to disable SD card power.
 */
void microSDPowerOff()
{
    delay(10);
    SERIAL_USB->println(F("Turning SD card power OFF..."));
    pinMode(SD_PWR, OUTPUT);
    digitalWrite(SD_PWR, HIGH);
    delay(10);
}

void SD_Card_Manager::generate_filename(bool use_folders) {
    uint32_t boot_count = boot_counter_instance.get_boot_number();
    common_working_posix_timestamp = board_time_manager.get_posix_timestamp();
    common_working_struct_YMDHMS = YMDHMS_from_posix_timestamp(common_working_posix_timestamp);
    
    if (use_folders) {
        // Generate filename in format: BOOT_XXXXXX/DATA_BOOT_XXXXXX_TIME_YYYYMMDDTHHMMSS.dat
        snprintf(filename_buffer, sizeof(filename_buffer),
                 "BOOT_%06u/DATA_BOOT_%06u_TIME_%04u%02u%02uT%02u%02u%02u.dat",
                 boot_count,
                 boot_count,
                 common_working_struct_YMDHMS.year,
                 common_working_struct_YMDHMS.month,
                 common_working_struct_YMDHMS.day,
                 common_working_struct_YMDHMS.hour,
                 common_working_struct_YMDHMS.minute,
                 common_working_struct_YMDHMS.second);
    } else {
        // Generate filename in format: DATA_BOOT_XXXXXX_TIME_YYYYMMDDTHHMMSS.dat (root folder)
        snprintf(filename_buffer, sizeof(filename_buffer),
                 "DATA_BOOT_%06u_TIME_%04u%02u%02uT%02u%02u%02u.dat",
                 boot_count,
                 common_working_struct_YMDHMS.year,
                 common_working_struct_YMDHMS.month,
                 common_working_struct_YMDHMS.day,
                 common_working_struct_YMDHMS.hour,
                 common_working_struct_YMDHMS.minute,
                 common_working_struct_YMDHMS.second);
    }
}

bool SD_Card_Manager::start() {
    if (sd_initialized) {
        return true;
    }

    SDIrqGuard _guard;  // pause IMU sample ISR for the duration of SD init

    microSDPowerOn();

    SERIAL_USB->println(F("Initializing SD card..."));
    
    // SHARED_SPI (not DEDICATED_SPI) because the OLA's IMU lives on the same
    // SPI bus. With DEDICATED_SPI, SdFat assumes exclusive access and skips
    // beginTransaction/endTransaction, which leaves the IOM configured for
    // the previous device (the IMU at 4 MHz) and SD CMD8 fails.
    SdSpiConfig sd_config{SD_CS_PIN, SHARED_SPI, SD_SCK_MHZ(SD_SPI_MHZ)};
    
    if (!sd_card.begin(sd_config)) {
        SERIAL_USB->println(F("ERROR: SD card initialization failed!"));
        SERIAL_USB->println(F("Check that SD card is inserted properly"));
        SERIAL_USB->print(F("Error code: "));
        SERIAL_USB->println(sd_card.card()->errorCode(), HEX);
        SERIAL_USB->print(F("Error data: "));
        SERIAL_USB->println(sd_card.card()->errorData(), HEX);
        
        // Turn off power on failure
        microSDPowerOff();
        return false;
    }
    
    sd_initialized = true;
    SERIAL_USB->println(F("SD card initialized successfully"));
    
    // Print card info
    SERIAL_USB->print(F("Card type: "));
    switch (sd_card.card()->type()) {
        case SD_CARD_TYPE_SD1:
            SERIAL_USB->println(F("SD1"));
            break;
        case SD_CARD_TYPE_SD2:
            SERIAL_USB->println(F("SD2"));
            break;
        case SD_CARD_TYPE_SDHC:
            SERIAL_USB->println(F("SDHC"));
            break;
        default:
            SERIAL_USB->println(F("Unknown"));
    }
    
    SERIAL_USB->print(F("Volume type: FAT"));
    SERIAL_USB->println(sd_card.vol()->fatType());
    
    return true;
}

void SD_Card_Manager::stop() {
    if (file_open) {
        close_and_sync_file();
    }
    
    if (sd_initialized) {
        sd_card.end();
        sd_initialized = false;
        SERIAL_USB->println(F("SD card stopped"));
    }
    
    microSDPowerOff();
}

bool SD_Card_Manager::preallocate_and_open_file(uint32_t size_bytes, bool use_folders) {
    if (!sd_initialized) {
        SERIAL_USB->println(F("ERROR: SD card not initialized"));
        return false;
    }

    SDIrqGuard _guard;  // SD opens/preallocate do lots of SPI traffic — pause IMU ISR

    if (file_open) {
        close_and_sync_file();
    }

    generate_filename(use_folders);
    
    // Extract folder name from filename_buffer if using folders (format: BOOT_XXXXXX/DATA_BOOT_...)
    if (use_folders) {
        char folder_name[32];  // "BOOT_XXXXXX" + null terminator
        char* slash_pos = strchr(filename_buffer, '/');
        if (slash_pos != nullptr) {
            size_t folder_len = slash_pos - filename_buffer;
            strncpy(folder_name, filename_buffer, folder_len);
            folder_name[folder_len] = '\0';
            
            // Check if folder exists, create if it doesn't
            if (!sd_card.exists(folder_name)) {
                SERIAL_USB->print(F("Creating folder: "));
                SERIAL_USB->println(folder_name);
                if (!sd_card.mkdir(folder_name)) {
                    SERIAL_USB->println(F("ERROR: Failed to create folder!"));
                    return false;
                }
                SERIAL_USB->println(F("Folder created successfully"));
            }
        }
    }
    
    SERIAL_USB->print(F("Opening file: "));
    SERIAL_USB->println(filename_buffer);
    
    // Try to remove old file first
    if (sd_card.exists(filename_buffer)) {
        SERIAL_USB->println(F("WARNING: Removing old file..."));
        sd_card.remove(filename_buffer);
    }
    delay(500);
    wdt.restart();
    
    if (!sd_file.open(filename_buffer, O_RDWR | O_CREAT)) {
        SERIAL_USB->println(F("ERROR: Failed to open new file!"));
        SERIAL_USB->print(F("Error code: "));
        SERIAL_USB->println(sd_card.card()->errorCode(), HEX);
        SERIAL_USB->print(F("Error data: "));
        SERIAL_USB->println(sd_card.card()->errorData(), HEX);
        return false;
    }
    delay(500);
    wdt.restart();
    
    file_open = true;
    SERIAL_USB->println(F("File opened successfully"));
    
    // Truncate file to zero before preallocation
    if (!sd_file.truncate(0)) {
        SERIAL_USB->println(F("WARNING: Failed to truncate file"));
    }
    delay(500);
    wdt.restart();
    
    // Preallocate if size specified
    if (size_bytes > 0) {
        SERIAL_USB->print(F("Preallocating "));
        SERIAL_USB->print(size_bytes);
        SERIAL_USB->println(F(" bytes..."));
        
        // Restart watchdog before starting long preallocation
        wdt.restart();
        
        if (!sd_file.preAllocate(size_bytes)) {
            SERIAL_USB->println(F("WARNING: Failed to preallocate file!"));
            SERIAL_USB->println(F("Continuing without preallocation..."));
        } else {
            SERIAL_USB->println(F("File preallocated successfully"));
            wdt.restart();  // Restart after preallocation completes
            sd_file.sync();  // Ensure FAT is updated
        }
    }
    delay(500);
    wdt.restart();
    
    // Initialize RingBuf with the opened file
#ifdef USE_RINGBUFF
    ring_buf.begin(&sd_file);
    delay(500);
    wdt.restart();
#endif
    
    return true;
}

void SD_Card_Manager::close_and_sync_file() {
    if (!file_open) {
        return;
    }

    SDIrqGuard _guard;  // sync/close are long-running SD SPI ops; pause IMU ISR

    SERIAL_USB->println(F("Syncing and closing file..."));
    
#ifdef USE_RINGBUFF
    // Flush all data from RingBuf to file
    ring_buf.sync();
    delay(500);
    wdt.restart();
#endif
    
    sd_file.sync();
    delay(500);
    wdt.restart();

    sd_file.close();
    file_open = false;
    SERIAL_USB->println(F("File closed"));
    wdt.restart();
    delay(500);
    wdt.restart();
}

bool SD_Card_Manager::write_buffer(const uint8_t* buffer, size_t size) {
    if (!file_open) {
        SERIAL_USB->println(F("ERROR: No file open for writing"));
        return false;
    }

    SDIrqGuard _guard;  // protect SD SPI writes from being preempted by IMU ISR

#ifdef USE_RINGBUFF
    digitalWrite(PIN_STAT_LED, HIGH);
    // Check if we need to flush before writing new data
    // This ensures we don't overflow the buffer
    // Do this BEFORE writing to avoid blocking after the write
    if (ring_buf.bytesFree() < size + SD_RINGBUF_SIZE / 4) {
        // Write out half the buffer to make room
        // This performs SD writes but keeps ISR blocking minimal
        // Restart watchdog as this can take time
        wdt.restart();

        ring_buf.writeOut(SD_RINGBUF_SIZE / 2);

        wdt.restart();
    }
    
    // Write to RingBuf - the write() itself only briefly disables interrupts
    // during the counter update (a few CPU cycles)
    size_t written = ring_buf.write(buffer, size);
    digitalWrite(PIN_STAT_LED, LOW);
#else
    // Direct write to SD card file
    digitalWrite(PIN_STAT_LED, HIGH);
    size_t written = sd_file.write(buffer, size);
    digitalWrite(PIN_STAT_LED, LOW);
#endif
    
    return (written == size);
}
