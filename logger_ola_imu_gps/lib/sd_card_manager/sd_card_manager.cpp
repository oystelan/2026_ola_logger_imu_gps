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

    if (file_open) {
        close_and_sync_file();
    }

    generate_filename(use_folders);

    // Extract folder name from filename_buffer if using folders.
    if (use_folders) {
        char folder_name[32];
        char* slash_pos = strchr(filename_buffer, '/');
        if (slash_pos != nullptr) {
            size_t folder_len = slash_pos - filename_buffer;
            strncpy(folder_name, filename_buffer, folder_len);
            folder_name[folder_len] = '\0';

            bool folder_exists;
            { SDIrqGuard _guard; folder_exists = sd_card.exists(folder_name); }
            if (!folder_exists) {
                SERIAL_USB->print(F("Creating folder: "));
                SERIAL_USB->println(folder_name);
                bool ok;
                { SDIrqGuard _guard; ok = sd_card.mkdir(folder_name); }
                if (!ok) {
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
    bool existed;
    { SDIrqGuard _guard; existed = sd_card.exists(filename_buffer); }
    if (existed) {
        SERIAL_USB->println(F("WARNING: Removing old file..."));
        SDIrqGuard _guard;
        sd_card.remove(filename_buffer);
    }
    wdt.restart();

    bool opened;
    { SDIrqGuard _guard; opened = sd_file.open(filename_buffer, O_RDWR | O_CREAT); }
    if (!opened) {
        SERIAL_USB->println(F("ERROR: Failed to open new file!"));
        SERIAL_USB->print(F("Error code: "));
        SERIAL_USB->println(sd_card.card()->errorCode(), HEX);
        SERIAL_USB->print(F("Error data: "));
        SERIAL_USB->println(sd_card.card()->errorData(), HEX);
        return false;
    }
    wdt.restart();

    file_open = true;
    SERIAL_USB->println(F("File opened successfully"));

    // Truncate file to zero (clean slate if the file already existed).
    bool truncated;
    { SDIrqGuard _guard; truncated = sd_file.truncate(0); }
    if (!truncated) {
        SERIAL_USB->println(F("WARNING: Failed to truncate file"));
    }
    wdt.restart();

    // Pre-allocate enough contiguous space for a full 15-min file at the
    // dmp-fifo-100hz branch's lower data rate. SdFat's RingBuf requires the
    // file to be fully pre-allocated for its expected lifetime — past the
    // pre-allocated boundary, RingBuf's block-aligned writeOut path stops
    // extending the file (BOOT_000106 stopped writing at ~150 s = exactly
    // 1 MB / 7 KB/s with a previous 1 MB cap).
    //
    // Data rate budget at 100 Hz, accel+gyro only (no mag) in DMP-FIFO:
    //   100 Hz × 28 B/IMU_record  = 2800 B/s   (\nIMU marker + 24 B struct)
    //   10 Hz × 44 B/GNSS_record  =  440 B/s
    //   ~1 Hz × 8 B/PPS_record    =    8 B/s
    //   + header text ~500 B at file open
    //   total ~3.25 KB/s, so 15 min ≈ 2.93 MB.
    // 3 MB pre-alloc just barely covers it; we round up to 4 MB so the file
    // never runs out, even with a 15-min recording that goes slightly long
    // because of crystal drift. 4 MB at 32 MHz ≈ ~500-800 ms — fits inside
    // the chip's 1.86 s FIFO headroom with margin to spare.
    //
    // The `size_bytes` parameter is honored only as an upper bound; we
    // never allocate more than PREALLOCATE_CHUNK_BYTES per call.
    static constexpr uint32_t PREALLOCATE_CHUNK_BYTES = 4u * 1024 * 1024;
    uint32_t alloc_size = size_bytes > 0
        ? (size_bytes < PREALLOCATE_CHUNK_BYTES ? size_bytes : PREALLOCATE_CHUNK_BYTES)
        : PREALLOCATE_CHUNK_BYTES;

    SERIAL_USB->print(F("Preallocating "));
    SERIAL_USB->print(alloc_size);
    SERIAL_USB->println(F(" bytes..."));
    wdt.restart();
    bool prealloc_ok;
    { SDIrqGuard _guard; prealloc_ok = sd_file.preAllocate(alloc_size); }
    if (!prealloc_ok) {
        SERIAL_USB->println(F("WARNING: Failed to preallocate file; "
                              "continuing — writes may be slower."));
    } else {
        SERIAL_USB->println(F("File preallocated successfully"));
    }
    wdt.restart();
    { SDIrqGuard _guard; sd_file.sync(); }
    wdt.restart();

    // Initialize RingBuf with the opened file (pure pointer assignment, no SPI)
#ifdef USE_RINGBUFF
    ring_buf.begin(&sd_file);
    wdt.restart();
#endif

    return true;
}

void SD_Card_Manager::close_and_sync_file() {
    if (!file_open) {
        return;
    }

    SERIAL_USB->println(F("Syncing and closing file..."));

    // Each of these is a separate SD SPI op. Wrap each individually so the
    // IMU ISR can fire during inter-op bookkeeping / watchdog restarts.
#ifdef USE_RINGBUFF
    { SDIrqGuard _guard; ring_buf.sync(); }
    wdt.restart();
#endif

    { SDIrqGuard _guard; sd_file.sync(); }
    wdt.restart();

    // Shrink the file from its preallocated size down to the bytes actually
    // written: the preallocation only matters for write performance WHILE
    // logging; leaving the zero tail on closed files wasted ~4x the SD space
    // (12 MiB files carrying ~3 MB of data).
    { SDIrqGuard _guard; sd_file.truncate(sd_file.curPosition()); }
    wdt.restart();

    { SDIrqGuard _guard; sd_file.close(); }
    file_open = false;
    SERIAL_USB->println(F("File closed"));
    wdt.restart();
}

bool SD_Card_Manager::write_buffer(const uint8_t* buffer, size_t size) {
    if (!file_open) {
        SERIAL_USB->println(F("ERROR: No file open for writing"));
        return false;
    }

#ifdef USE_RINGBUFF
    digitalWrite(PIN_STAT_LED, HIGH);
    // Two distinct phases here, only ONE of which actually touches SPI:
    //   (a) ring_buf.writeOut(): flushes RAM ring buffer to the SD card
    //       over SPI. THIS is the operation that contends with the IMU
    //       SPI traffic — wrap it in SDIrqGuard so the IMU sample ISR
    //       cannot preempt it mid-block.
    //   (b) ring_buf.write(): a pure RAM copy from `buffer` into the ring
    //       buffer. No SPI involved. The IMU ISR is welcome to fire during
    //       this so gyro+mag stay fresh.
    //
    // Pre-refactor we wrapped the whole function in SDIrqGuard, which
    // silenced the IMU ISR for ~200 ms per RingBuf drain. That windowed
    // the gyro+mag flat-spots we kept seeing. Now the guard is just the
    // ~3 ms of actual SD write time.
    // Drain policy: KEEP THE RING NEARLY EMPTY. The previous trigger
    // (drain only when < 25% free) deliberately let the ring sit ~75% full
    // in steady state, leaving only ~8 KB of headroom to absorb an SD stall —
    // which is how even 100 Hz recordings overflowed during long card-GC
    // episodes. Now we drain a small chunk as soon as one is available:
    // writes are more frequent but shorter (2 KB = 4 SD sectors, a few ms of
    // SDIrqGuard each), and the ring's steady state is < 2 KB used — so
    // nearly the FULL 32 KB (~ 9 s of data) is available as stall headroom.
    static constexpr size_t WRITEOUT_BYTES = 2048;
    if (ring_buf.bytesUsed() >= WRITEOUT_BYTES) {
        wdt.restart();
        {
            SDIrqGuard _guard;
            ring_buf.writeOut(WRITEOUT_BYTES);
        }
        wdt.restart();
    }

    // RAM-only copy — IMU ISR may run freely here.
    //
    // ATOMIC RECORD GUARD: RingBuf::write copies only as many bytes as fit.
    // If the ring fills during a long SD stall, a PARTIAL record gets written
    // and every subsequent byte in the file is misaligned — the decoder then
    // sees junk bytes, byte-shifted frames (gravity on the wrong axis) and
    // 0x4000-pattern garbage values until it resyncs. Never write a partial
    // record: if the whole record does not fit, drop it and count it. A
    // dropped record is a clean, decoder-friendly gap; a partial one poisons
    // the rest of the file.
    size_t written;
    if (ring_buf.bytesFree() < size) {
        dropped_records += 1;
        dropped_bytes += size;
        written = 0;
    } else {
        written = ring_buf.write(buffer, size);
    }
    digitalWrite(PIN_STAT_LED, LOW);
#else
    // Direct write to SD card file — needs the IRQ guard for the whole op.
    digitalWrite(PIN_STAT_LED, HIGH);
    SDIrqGuard _guard;
    size_t written = sd_file.write(buffer, size);
    digitalWrite(PIN_STAT_LED, LOW);
#endif

    return (written == size);
}
