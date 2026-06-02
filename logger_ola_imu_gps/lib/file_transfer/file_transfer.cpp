#include "file_transfer.h"

#include <Arduino.h>
#include <SdFat.h>
#include <math.h>
#include <string.h>

#include "firmware_configuration.h"
#include "sd_card_manager.h"
#include "watchdog_manager.h"

// ============================================================================
// Mode-signature LED: cosine pulse 30%..70% intensity, ~7 s period.
// Drives PIN_STAT_LED via analogWrite() so the file-transfer mode is
// unambiguously distinguishable from the heartbeat-blink of normal logging
// and the fixed-rate blinks of gyro/mag cal.
// ============================================================================
static unsigned long s_pulse_start_ms = 0;

static void pulse_led_update(void)
{
    unsigned long const now_ms = millis();
    if (s_pulse_start_ms == 0) {
        s_pulse_start_ms = now_ms;
    }
    constexpr float PERIOD_MS = 7000.0f;
    constexpr float MIN_INTENSITY = 0.00f;
    constexpr float MAX_INTENSITY = 0.60f;
    float const phase = (float)((now_ms - s_pulse_start_ms) % (unsigned long)PERIOD_MS) / PERIOD_MS;
    float const mid = 0.5f * (MIN_INTENSITY + MAX_INTENSITY);
    float const amp = 0.5f * (MAX_INTENSITY - MIN_INTENSITY);
    float const intensity = mid + amp * cosf(2.0f * (float)M_PI * phase);
    int const pwm = (int)(255.0f * intensity);
    analogWrite(PIN_STAT_LED, pwm);
}

// ============================================================================
// CRC32 (zlib/PNG polynomial 0xEDB88320, reversed). Computed once at first
// use and cached in SRAM (1 KB table; no PROGMEM hassle on Apollo3).
// ============================================================================
static uint32_t crc32_table[256];
static bool crc32_inited = false;

static void crc32_init(void)
{
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) {
            c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        }
        crc32_table[i] = c;
    }
    crc32_inited = true;
}

static uint32_t crc32_update(uint32_t crc, const uint8_t *buf, size_t len)
{
    crc = ~crc;
    for (size_t i = 0; i < len; i++) {
        crc = (crc >> 8) ^ crc32_table[(crc ^ buf[i]) & 0xFFu];
    }
    return ~crc;
}

// ============================================================================
// Tiny string + IO helpers
// ============================================================================
static bool str_endswith(const char *s, const char *suffix)
{
    size_t ls = strlen(s);
    size_t lsx = strlen(suffix);
    if (lsx > ls) return false;
    return strcmp(s + ls - lsx, suffix) == 0;
}

static bool str_startswith(const char *s, const char *prefix)
{
    return strncmp(s, prefix, strlen(prefix)) == 0;
}

// Read a CR/LF-terminated line from SERIAL_USB into `buf` (NUL-terminated).
// Blocks (with wdt resets) until a complete line is received or buf is full.
// Returns the length excluding terminator. Empty lines return 0.
static size_t read_line(char *buf, size_t bufsize)
{
    size_t n = 0;
    while (n + 1 < bufsize) {
        wdt.restart();
        if (SERIAL_USB->available() == 0) {
            // Update the cosine pulse LED while idle. The 10 ms tick keeps
            // the PWM smooth without burning the CPU.
            pulse_led_update();
            delay(10);
            continue;
        }
        int c = SERIAL_USB->read();
        if (c == '\r') continue;            // ignore CR
        if (c == '\n') break;               // line terminator
        if (c == 0x08 || c == 0x7F) {       // BS/DEL for interactive use
            if (n > 0) n--;
            continue;
        }
        buf[n++] = static_cast<char>(c);
    }
    buf[n] = '\0';
    return n;
}

// ============================================================================
// File enumeration
//
// We do two-pass operation: list prints as it walks; get walks again, looking
// for the requested index. The walk order is therefore the source of truth
// for indices and MUST be deterministic — SdFat's FsFile::openNext() iterates
// in FAT directory order which is stable as long as the filesystem isn't
// being modified, which it isn't in transfer mode.
//
// Visitor: returns true to continue, false to abort. Visit numbering starts
// at 0 and increments for every .dat file emitted.
//
// Path layout: BOOT_NNNNNN/DATA_BOOT_NNNNNN_TIME_*.dat is the firmware's
// USE_FOLDERS=true output. We also collect any top-level .dat files (for
// USE_FOLDERS=false recordings or any leftover files in root) for robustness.
// ============================================================================
typedef bool (*FileVisitor)(int idx, const char *path, uint32_t size_bytes, void *user);

static int walk_dat_files(FileVisitor visit, void *user)
{
    SdFat *sd = sd_card_manager.get_sd();
    FsFile root;
    if (!root.open(sd, "/", O_RDONLY)) {
        return -1;
    }
    int idx = 0;
    bool keep_going = true;
    FsFile entry;
    char name[80];
    char fullpath[160];
    while (keep_going && entry.openNext(&root, O_RDONLY)) {
        entry.getName(name, sizeof(name));
        if (entry.isDir() && str_startswith(name, "BOOT_")) {
            FsFile inner;
            char inner_name[80];
            while (keep_going && inner.openNext(&entry, O_RDONLY)) {
                inner.getName(inner_name, sizeof(inner_name));
                if (!inner.isDir() && str_endswith(inner_name, ".dat")) {
                    snprintf(fullpath, sizeof(fullpath), "%s/%s", name, inner_name);
                    uint32_t sz = static_cast<uint32_t>(inner.fileSize());
                    keep_going = visit(idx, fullpath, sz, user);
                    idx++;
                }
                inner.close();
            }
        } else if (!entry.isDir() && str_endswith(name, ".dat")) {
            uint32_t sz = static_cast<uint32_t>(entry.fileSize());
            keep_going = visit(idx, name, sz, user);
            idx++;
        }
        entry.close();
    }
    root.close();
    return idx;
}

// ============================================================================
// Commands
// ============================================================================
static bool visit_print(int idx, const char *path, uint32_t size_bytes, void * /*user*/)
{
    SERIAL_USB->print(idx);
    SERIAL_USB->print(' ');
    SERIAL_USB->print(size_bytes);
    SERIAL_USB->print(' ');
    SERIAL_USB->println(path);
    return true;
}

static void cmd_list(void)
{
    SERIAL_USB->println(F("LIST_BEGIN"));
    int count = walk_dat_files(visit_print, nullptr);
    if (count < 0) {
        SERIAL_USB->println(F("ERROR sd_open_failed"));
        return;
    }
    SERIAL_USB->print(F("LIST_END "));
    SERIAL_USB->println(count);
}

struct GetTarget {
    int target_idx;
    bool found;
    char path[160];
    uint32_t size_bytes;
};

static bool visit_find(int idx, const char *path, uint32_t size_bytes, void *user)
{
    GetTarget *t = static_cast<GetTarget *>(user);
    if (idx == t->target_idx) {
        strncpy(t->path, path, sizeof(t->path) - 1);
        t->path[sizeof(t->path) - 1] = '\0';
        t->size_bytes = size_bytes;
        t->found = true;
        return false;  // abort walk
    }
    return true;
}

static void cmd_get(int target_idx)
{
    GetTarget t = {target_idx, false, {0}, 0};
    walk_dat_files(visit_find, &t);
    if (!t.found) {
        SERIAL_USB->println(F("ERROR index_not_found"));
        return;
    }

    SdFat *sd = sd_card_manager.get_sd();
    FsFile f;
    if (!f.open(sd, t.path, O_RDONLY)) {
        SERIAL_USB->println(F("ERROR open_failed"));
        return;
    }

    // Header line: "GET_BEGIN <idx> <bytes>\n"
    SERIAL_USB->print(F("GET_BEGIN "));
    SERIAL_USB->print(target_idx);
    SERIAL_USB->print(' ');
    SERIAL_USB->println(t.size_bytes);

    if (!crc32_inited) crc32_init();
    uint32_t crc = 0;
    uint8_t buf[1024];
    uint32_t remaining = t.size_bytes;
    while (remaining > 0) {
        wdt.restart();
        pulse_led_update();  // keep the cosine pulse animating during transfer
        size_t to_read = (remaining > sizeof(buf)) ? sizeof(buf) : remaining;
        int r = f.read(buf, to_read);
        if (r <= 0) {
            // SD read failure mid-stream. The host will see a truncated
            // payload (received bytes < advertised). Nothing useful we can
            // emit on the same line since binary is in flight; just break
            // out and let the trailer go anyway.
            break;
        }
        SERIAL_USB->write(buf, r);
        crc = crc32_update(crc, buf, r);
        remaining -= r;
    }
    f.close();

    // Trailer on its own line. Leading \n in case the last byte of the file
    // wasn't a newline (it usually isn't — these are binary files).
    SERIAL_USB->println();
    SERIAL_USB->print(F("GET_END "));
    SERIAL_USB->println(crc, HEX);
}

static void cmd_info(void)
{
    SERIAL_USB->print(F("FIRMWARE_COMMIT "));
    SERIAL_USB->println(commit_id);
    SERIAL_USB->print(F("FIRMWARE_BRANCH "));
    SERIAL_USB->println(git_branch);

    SdFat *sd = sd_card_manager.get_sd();
    if (sd && sd->card()) {
        uint32_t sectors = sd->card()->sectorCount();
        SERIAL_USB->print(F("SD_SECTORS "));
        SERIAL_USB->println(sectors);
        // 512 bytes/sector; print MB for human consumption.
        SERIAL_USB->print(F("SD_SIZE_MB "));
        SERIAL_USB->println(static_cast<uint32_t>(sectors / 2048));
    }
}

static void cmd_help(void)
{
    SERIAL_USB->println(F("Commands:"));
    SERIAL_USB->println(F("  list           - enumerate .dat files (returns index N for each)"));
    SERIAL_USB->println(F("  get <N>        - stream binary content of file at index N"));
    SERIAL_USB->println(F("  info           - firmware + SD info"));
    SERIAL_USB->println(F("  help           - this message"));
    SERIAL_USB->println(F("  exit | reboot  - leave transfer mode, reboot to normal mode"));
}

// ============================================================================
// Entry point
// ============================================================================
void enter_file_transfer_mode(void)
{
    SERIAL_USB->println();
    SERIAL_USB->println(F("=== OLA FILE TRANSFER MODE ==="));
    SERIAL_USB->print(F("Firmware: "));
    SERIAL_USB->println(commit_id);
    SERIAL_USB->println(F("Type 'help' for the command list."));
    SERIAL_USB->println(F("READY"));

    char buf[128];
    while (true) {
        wdt.restart();
        size_t n = read_line(buf, sizeof(buf));
        if (n == 0) {
            // Empty line — emit READY again so the host stays in sync.
            SERIAL_USB->println(F("READY"));
            continue;
        }

        // Split first token (command) and rest (arg).
        char *arg = strchr(buf, ' ');
        if (arg != nullptr) {
            *arg = '\0';
            arg++;
        }

        if (strcmp(buf, "list") == 0) {
            cmd_list();
        } else if (strcmp(buf, "get") == 0 && arg != nullptr) {
            int idx = atoi(arg);
            cmd_get(idx);
        } else if (strcmp(buf, "info") == 0) {
            cmd_info();
        } else if (strcmp(buf, "help") == 0) {
            cmd_help();
        } else if (strcmp(buf, "exit") == 0 || strcmp(buf, "reboot") == 0) {
            SERIAL_USB->println(F("BYE"));
            delay(100);
            NVIC_SystemReset();
        } else {
            SERIAL_USB->print(F("ERROR unknown_command: '"));
            SERIAL_USB->print(buf);
            SERIAL_USB->println('\'');
        }
        SERIAL_USB->println(F("READY"));
    }
}
