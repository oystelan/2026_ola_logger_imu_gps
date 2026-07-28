#include "calibration_manager.h"

CalibrationManager calibration_manager;

// Out-of-line definitions for the static constexpr members. Required because
// they are passed by const-reference to EEPROM.put()/get() (ODR-use) and the
// project compiles as C++14 (pre-C++17 inline-variable semantics).
constexpr uint8_t CalibrationManager::FLAG_DECISION_PENDING;
constexpr uint8_t CalibrationManager::FLAG_IDLE;
constexpr uint8_t CalibrationManager::GYRO_VALID_MAGIC;
constexpr uint8_t CalibrationManager::MAG_VALID_MAGIC;
constexpr uint8_t CalibrationManager::TIME_VALID_MAGIC;
constexpr uint8_t CalibrationManager::MAX_PRESS_COUNT;

CalibrationManager::BootAction
CalibrationManager::detect_boot_action(int stat_led_pin, unsigned long window_ms)
{
    // Reset-cause gate. Only a reset from the external RESET pin may open or
    // advance the multi-press sequence. Power-on / brown-out resets (battery
    // contact bounce, USB plug-in) and watchdog reboots previously registered
    // as phantom "presses": powering up commonly produces two boots in quick
    // succession (power-on, then a CH340 DTR reset pulse when the host opens
    // the serial port), which landed inside the decision window and triggered
    // FILE TRANSFER mode without any button being touched.
    //
    // The RSTGEN status bits are sticky (they accumulate across resets), so
    // read-then-clear every boot: the next boot then sees only its own cause.
    am_hal_reset_status_t rst;
    am_hal_reset_status_get(&rst);
    am_hal_reset_control(AM_HAL_RESET_CONTROL_STATUSCLEAR, 0);
    const bool button_reset = rst.bEXTStat && !rst.bPORStat && !rst.bBODStat;

    if (!button_reset) {
        // Power-up / brown-out / watchdog boot: never part of a press
        // sequence. Clear any stale pending flag (e.g. power lost mid-window)
        // and boot normally, WITHOUT the 2.5 s decision window — which also
        // makes cold boots and watchdog recoveries faster.
        EEPROM.put(ADDR_MULTIPRESS_FLAG, FLAG_IDLE);
        EEPROM.put(ADDR_MULTIPRESS_COUNT, (uint8_t)0);
        return BootAction::NORMAL;
    }

    // Read the decision-pending flag and current press count from EEPROM.
    uint8_t flag = 0;
    uint8_t count = 0;
    EEPROM.get(ADDR_MULTIPRESS_FLAG, flag);
    EEPROM.get(ADDR_MULTIPRESS_COUNT, count);

    if (flag == FLAG_DECISION_PENDING) {
        // The previous boot set the pending flag and was still inside its
        // decision window when RESET was pressed again — so this boot is a
        // rapid re-press. Increment the count.
        count = (count < MAX_PRESS_COUNT) ? (uint8_t)(count + 1) : MAX_PRESS_COUNT;
    } else {
        // Fresh start (last boot completed its window and cleared the flag,
        // or this is the very first boot). Begin a new sequence at 1.
        count = 1;
    }

    // Mark the decision window as open and persist the new count BEFORE we
    // start waiting, so that if RESET is pressed during the window the next
    // boot sees FLAG_DECISION_PENDING and increments correctly.
    EEPROM.put(ADDR_MULTIPRESS_FLAG, FLAG_DECISION_PENDING);
    EEPROM.put(ADDR_MULTIPRESS_COUNT, count);

    // Decision window. Blink the STAT LED to signal "you may press RESET
    // again now". Blink rate scales with the count so the user gets feedback
    // on how many presses have registered (faster = higher count).
    pinMode(stat_led_pin, OUTPUT);
    const unsigned long blink_period = (count >= 3) ? 80 : (count == 2 ? 150 : 300);
    unsigned long start = millis();
    while (millis() - start < window_ms) {
        digitalWrite(stat_led_pin, ((millis() / blink_period) % 2) ? HIGH : LOW);
    }
    digitalWrite(stat_led_pin, LOW);

    // Window survived without a re-press → commit. Clear the pending flag so
    // the next ordinary reboot starts a fresh sequence.
    EEPROM.put(ADDR_MULTIPRESS_FLAG, FLAG_IDLE);

    switch (count) {
        case 2:  return BootAction::FILE_TRANSFER;
        case 3:  return BootAction::MAG_CAL;
        case 4:  return BootAction::GYRO_CAL;
        default: return BootAction::NORMAL;
    }
}

void CalibrationManager::load(void)
{
    uint8_t magic = 0;
    EEPROM.get(ADDR_GYRO_VALID, magic);
    if (magic == GYRO_VALID_MAGIC) {
        EEPROM.get(ADDR_GYRO_BIAS_X, gyro_bias_x_);
        EEPROM.get(ADDR_GYRO_BIAS_Y, gyro_bias_y_);
        EEPROM.get(ADDR_GYRO_BIAS_Z, gyro_bias_z_);
        gyro_valid_ = true;
    } else {
        gyro_bias_x_ = gyro_bias_y_ = gyro_bias_z_ = 0;
        gyro_valid_ = false;
    }

    magic = 0;
    EEPROM.get(ADDR_MAG_VALID, magic);
    if (magic == MAG_VALID_MAGIC) {
        EEPROM.get(ADDR_MAG_BIAS_X, mag_bias_x_);
        EEPROM.get(ADDR_MAG_BIAS_Y, mag_bias_y_);
        EEPROM.get(ADDR_MAG_BIAS_Z, mag_bias_z_);
        mag_valid_ = true;
    } else {
        mag_bias_x_ = mag_bias_y_ = mag_bias_z_ = 0;
        mag_valid_ = false;
    }
}

void CalibrationManager::save_gyro_bias(int16_t bx, int16_t by, int16_t bz)
{
    EEPROM.put(ADDR_GYRO_BIAS_X, bx);
    EEPROM.put(ADDR_GYRO_BIAS_Y, by);
    EEPROM.put(ADDR_GYRO_BIAS_Z, bz);
    EEPROM.put(ADDR_GYRO_VALID, GYRO_VALID_MAGIC);
    gyro_bias_x_ = bx;
    gyro_bias_y_ = by;
    gyro_bias_z_ = bz;
    gyro_valid_ = true;
}

void CalibrationManager::save_mag_hardiron(int16_t hx, int16_t hy, int16_t hz)
{
    EEPROM.put(ADDR_MAG_BIAS_X, hx);
    EEPROM.put(ADDR_MAG_BIAS_Y, hy);
    EEPROM.put(ADDR_MAG_BIAS_Z, hz);
    EEPROM.put(ADDR_MAG_VALID, MAG_VALID_MAGIC);
    mag_bias_x_ = hx;
    mag_bias_y_ = hy;
    mag_bias_z_ = hz;
    mag_valid_ = true;
}

void CalibrationManager::save_last_known_posix(uint32_t posix_seconds)
{
    EEPROM.put(ADDR_TIME_POSIX, posix_seconds);
    EEPROM.put(ADDR_TIME_VALID, TIME_VALID_MAGIC);
}

uint32_t CalibrationManager::load_last_known_posix(void) const
{
    if (!has_last_known_posix()) {
        return 0;
    }
    uint32_t posix_seconds = 0;
    EEPROM.get(ADDR_TIME_POSIX, posix_seconds);
    return posix_seconds;
}

bool CalibrationManager::has_last_known_posix(void) const
{
    uint8_t magic = 0;
    EEPROM.get(ADDR_TIME_VALID, magic);
    return magic == TIME_VALID_MAGIC;
}

void CalibrationManager::print_state(Stream &out) const
{
    out.print(F("Gyro calibration: "));
    if (gyro_valid_) {
        out.print(F("bias (body LSB) x="));
        out.print(gyro_bias_x_);
        out.print(F(" y="));
        out.print(gyro_bias_y_);
        out.print(F(" z="));
        out.println(gyro_bias_z_);
    } else {
        out.println(F("none stored (run quadruple-tap RESET to calibrate)"));
    }

    out.print(F("Mag  calibration: "));
    if (mag_valid_) {
        out.print(F("hard-iron (chip LSB) x="));
        out.print(mag_bias_x_);
        out.print(F(" y="));
        out.print(mag_bias_y_);
        out.print(F(" z="));
        out.println(mag_bias_z_);
    } else {
        out.println(F("none stored (run triple-tap RESET to calibrate)"));
    }
}
