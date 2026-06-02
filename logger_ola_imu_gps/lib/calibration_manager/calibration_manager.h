#ifndef CALIBRATION_MANAGER_H
#define CALIBRATION_MANAGER_H

#include <Arduino.h>
#include <EEPROM.h>

// Calibration manager for the OLA logger.
//
// Responsibilities:
//   1. Multi-press detection on the hardware RESET button. The button is wired
//      to NRST so each press reboots the chip; we detect "2 presses" / "3
//      presses" / "4 presses" across reboots using a decision-pending flag in
//      EEPROM (see detect_boot_action()).
//   2. Persistent storage of sensor biases in EEPROM (survives power loss).
//   3. Running the calibration procedures (Tier 1: gyro bias; Tier 2: mag
//      hard-iron — reserved, not yet implemented).
//   4. Applying the stored biases (the firmware ISR subtracts them before
//      writing samples, so the data on the SD card is already calibrated).
//
// Biases are stored as int16 LSB values in the BODY frame, i.e. AFTER the
// chip→PCB axis remap that the ISR applies — so the ISR can subtract them
// directly from the remapped gyro values.
class CalibrationManager {
public:
    enum class BootAction : uint8_t {
        NORMAL        = 0,
        GYRO_CAL      = 1,
        MAG_CAL       = 2,
        FILE_TRANSFER = 3,
    };

    // Detect how many times RESET was pressed in quick succession.
    //   1 press  → NORMAL          (ordinary reboot)
    //   2 presses → FILE_TRANSFER   (USB-serial download mode)
    //   3 presses → MAG_CAL          (magnetometer hard-iron calibration)
    //   4 presses → GYRO_CAL         (gyro bias calibration)
    // Must be called EARLY in setup(), after the STAT LED pin is configured
    // and millis() is valid (i.e. after burst mode / RTC setup). Blocks for
    // `window_ms` while the decision window is open, blinking the STAT LED so
    // the user knows they may press RESET again. The STAT LED pin is taken as
    // an argument to avoid a dependency on firmware_configuration here.
    BootAction detect_boot_action(int stat_led_pin, unsigned long window_ms = 2500);

    // Load stored biases from EEPROM into the cached members. Call once at
    // boot before the ISR starts using gyro_bias_*().
    void load(void);

    // True if a valid gyro calibration has been stored.
    bool has_gyro_calibration(void) const { return gyro_valid_; }

    // Cached gyro biases (body frame, int16 LSB). Zero if not calibrated.
    int16_t gyro_bias_x(void) const { return gyro_bias_x_; }
    int16_t gyro_bias_y(void) const { return gyro_bias_y_; }
    int16_t gyro_bias_z(void) const { return gyro_bias_z_; }

    // Store gyro biases to EEPROM and update the cache.
    void save_gyro_bias(int16_t bx, int16_t by, int16_t bz);

    // --- Tier 2: magnetometer hard-iron offset ---
    //
    // A hard-iron offset is a CONSTANT vector added to every mag reading by
    // ferrous mass that is fixed relative to the chip (USB connector, SD-card
    // metal shield, regulators, etc on the OLA itself). Subtracting it
    // produces a calibrated mag measurement suitable for yaw computation.
    bool has_mag_calibration(void) const { return mag_valid_; }
    int16_t mag_bias_x(void) const { return mag_bias_x_; }
    int16_t mag_bias_y(void) const { return mag_bias_y_; }
    int16_t mag_bias_z(void) const { return mag_bias_z_; }
    void save_mag_hardiron(int16_t hx, int16_t hy, int16_t hz);

    // Last-known UTC fallback. Used only when ENABLE_TIME_EEPROM_FALLBACK is
    // set in firmware_configuration.h. Saves the most recent POSIX timestamp
    // (e.g. at file rotation) so that a board WITHOUT a coin cell on VBAT can
    // still recover an approximate UTC across power cycles. Less accurate than
    // the H/W RTC (only as fresh as the last save), but better than the 1970
    // default. Returns 0 from load_last_known_posix() when no valid value is
    // stored (uninitialised EEPROM or never saved).
    void save_last_known_posix(uint32_t posix_seconds);
    uint32_t load_last_known_posix(void) const;
    bool has_last_known_posix(void) const;

    // Print the current calibration state to serial (for boot diagnostics).
    void print_state(Stream &out) const;

private:
    // EEPROM layout. Bytes 0-1 are owned by Boot_Counter; start at 2.
    static constexpr int ADDR_MULTIPRESS_FLAG  = 2;   // 1 byte
    static constexpr int ADDR_MULTIPRESS_COUNT = 3;   // 1 byte
    static constexpr int ADDR_GYRO_VALID       = 4;   // 1 byte (magic when valid)
    static constexpr int ADDR_GYRO_BIAS_X      = 5;   // 2 bytes (int16)
    static constexpr int ADDR_GYRO_BIAS_Y      = 7;   // 2 bytes
    static constexpr int ADDR_GYRO_BIAS_Z      = 9;   // 2 bytes
    static constexpr int ADDR_MAG_VALID        = 11;  // 1 byte (magic when valid)
    static constexpr int ADDR_MAG_BIAS_X       = 12;  // 2 bytes (int16, raw chip LSB)
    static constexpr int ADDR_MAG_BIAS_Y       = 14;  // 2 bytes
    static constexpr int ADDR_MAG_BIAS_Z       = 16;  // 2 bytes
    static constexpr int ADDR_TIME_VALID       = 18;  // 1 byte (magic when valid)
    static constexpr int ADDR_TIME_POSIX       = 19;  // 4 bytes (uint32 POSIX seconds)

    static constexpr uint8_t FLAG_DECISION_PENDING = 0xA5;
    static constexpr uint8_t FLAG_IDLE             = 0x00;
    static constexpr uint8_t GYRO_VALID_MAGIC      = 0x5A;
    static constexpr uint8_t MAG_VALID_MAGIC       = 0x96;
    static constexpr uint8_t TIME_VALID_MAGIC      = 0xC3;
    static constexpr uint8_t MAX_PRESS_COUNT       = 4;

    int16_t gyro_bias_x_ {0};
    int16_t gyro_bias_y_ {0};
    int16_t gyro_bias_z_ {0};
    bool gyro_valid_ {false};

    int16_t mag_bias_x_ {0};
    int16_t mag_bias_y_ {0};
    int16_t mag_bias_z_ {0};
    bool mag_valid_ {false};
};

extern CalibrationManager calibration_manager;

#endif
