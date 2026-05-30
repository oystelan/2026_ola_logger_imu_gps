#ifndef FIRMWARE_CONFIGURATION
#define FIRMWARE_CONFIGURATION

//////////////////////////////////////////////////////////////////////////////////////////
// the firmware configuration that the user should not touch
// this includes:
//   - hardware imposed choices (pins, ports numbers, etc)
//   - important conventions (baudrate, I2C frequencies, etc)
//////////////////////////////////////////////////////////////////////////////////////////

#include "Arduino.h"
#include "macro_utils.h"
#include "print_utils.h"

//////////////////////////////////////////////////////////////////////////////////////////
// serial over USB related

extern Uart * SERIAL_USB;
static constexpr int BAUD_RATE_USB {1000000};

//////////////////////////////////////////////////////////////////////////////////////////
// I2C over qwiic

// TODO: check that correct
static constexpr int PORT_I2C_QWIIC_NUMBER {1};
static constexpr int PIN_QWIIC_PWR {18};

//TODO: check that correct
static constexpr byte PIN_QWIIC_SCL {8};
static constexpr byte PIN_QWIIC_SDA {9};

//////////////////////////////////////////////////////////////////////////////////////////
// pins on the PCB

// LEDs
static constexpr int PIN_PWR_LED {29};
static constexpr int PIN_STAT_LED {19};

//////////////////////////////////////////////////////////////////////////////////////////
// SD card pins

static constexpr int SD_CS_PIN {23};
static constexpr int SD_SPI_MHZ {50};
static constexpr int SD_PWR {15};

//////////////////////////////////////////////////////////////////////////////////////////
// Built-in 9DoF IMU (ICM-20948) on SparkFun OLA — SPI

static constexpr int PIN_IMU_CHIP_SELECT {44};
static constexpr int PIN_IMU_POWER {27};
static constexpr int PIN_IMU_INT {37};
static constexpr int IMU_SPI_MHZ {4};

//////////////////////////////////////////////////////////////////////////////////////////
// Time-persistence behaviour
//
// Primary mechanism: the Apollo3 hardware RTC date/time registers, which are
// kept alive across power-off by the OLA's VBAT coin-cell. The boot path reads
// them, sanity-checks the year, and seeds the software POSIX counter — so any
// sample we log after boot is timestamped from "where the H/W RTC left off"
// rather than from the 1970 default.
//
// Fallback (this flag): if the H/W RTC value is implausible (no coin cell, or
// the cell is dead), recover the LAST UTC we wrote to EEPROM at file rotation.
// Less accurate than the H/W RTC (only as fresh as the last 15-minute save)
// but still far better than 1970. Enabled by default — costs one EEPROM write
// per 15-minute file rotation (~3 years to wear out an Apollo3 EEPROM cell).
static constexpr bool ENABLE_TIME_EEPROM_FALLBACK {true};

//////////////////////////////////////////////////////////////////////////////////////////
// misc

static constexpr char commit_id[] {STRINGIFY_CONTENT(REPO_COMMIT_ID)};
static constexpr char git_branch[] {STRINGIFY_CONTENT(REPO_GIT_BRANCH)};

//////////////////////////////////////////////////////////////////////////////////////////
// functions

void print_firmware_config(void);

uint64_t read_chip_id(void);

void blink_pwr_led(int num_blinks, int millis_on=200, int millis_off=200);
void blink_stat_led(int num_blinks, int millis_on=200, int millis_off=200);

#endif