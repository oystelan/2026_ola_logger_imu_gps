#include <Arduino.h>

#include "firmware_configuration.h"
#include "watchdog_manager.h"
#include "boot_counter.h"
#include "time_manager.h"
#include "gnss_manager.h"
#include "sleep_manager.h"
#include "sd_card_manager.h"

#include "Wire.h"
#include <SPI.h>
#include <SparkFun_u-blox_GNSS_v3.h>

#include <ICM_20948.h>

#include "Embedded_Template_Library.h"
#include "etl/deque.h"


float acc_sensitivity = 0.0f;
float gyr_sensitivity = 0.0f;

char working_buffer[1024];

static constexpr bool USE_FOLDERS = true; // If true, organize files into BOOT_XXXXXX folders; if false, put all files in root

SFE_UBLOX_GNSS log_GNSS;
// static constexpr uint32_t GNSS_FREQUENCY_HZ = 1;
static constexpr uint32_t GNSS_FREQUENCY_HZ = 10;

// ICM-20948 internal sample-rate is 1125 / (1 + SMPLRT_DIV); with div=4 the ODR is exactly 225 Hz on both
// accelerometer and gyroscope. Pick this constant together with IMU_SMPLRT_DIV below.
static constexpr float IMU_ODR_HZ = 225.0f;
static constexpr uint16_t IMU_SMPLRT_DIV = 4;

// Timer configuration
static constexpr int TIMER_NUM = 2;
static constexpr uint32_t TIMER_FREQ_HZ = static_cast<uint32_t>(2 * IMU_ODR_HZ);
// static constexpr uint32_t TIMER_FREQ_HZ = 1000;
static constexpr uint32_t TIMER_DIVIDER_GNSS = static_cast<uint32_t>(TIMER_FREQ_HZ / GNSS_FREQUENCY_HZ / 1.5);

static constexpr uint32_t SERIAL_TIMEOUT_MS = 5000;      ///< Max wait for serial connection

static constexpr bool ENABLE_BLINK_PWR_LED = false;          ///< Enable power LED blinking on startup
static constexpr bool ENABLE_BOOT_COUNTER = true;          ///< Enable boot counter functionality
static constexpr bool ENABLE_GNSS = true;                    ///< Master switch for GNSS module (begin, ISR read, PPS, deque write)
static constexpr bool ENABLE_GNSS_START = true;              ///< Wait for an initial GNSS fix at boot to set the RTC; STAT LED blinks 5x once UTC is synced
static constexpr bool ENABLE_DEBUG_FASTPRINT = false;

static constexpr bool USE_BURSTMODE = true;

TwoWire * I2C_QWIIC = &Wire1;

static constexpr int PIN_LOG_PPS = 11; // Pin to log PPS signal from GNSS

ICM_20948_SPI imu;

static constexpr uint32_t seconds_in_15_minutes = 15 * 60;

constexpr uint32_t PREALLOCATE_LOGFILE_SIZE_BYTES = 12 * 1024 * 1024; // Preallocate a file large enough for logging

static constexpr char str_start_logging[] = "Log start OLA ICM-20948 logger\n\n";
static constexpr char str_stop_logging[] = "\n\nLog stop OLA ICM-20948 logger\n";

static constexpr size_t SIZE_DEQUE_IMU {20*( (int)IMU_ODR_HZ)};
static constexpr size_t SIZE_DEQUE_GNSS {20*GNSS_FREQUENCY_HZ};
static constexpr size_t SIZE_DEQUE_PPS {20*1};

size_t working_deque_size {0};
size_t max_deque_size_imu {0};
size_t max_deque_size_gnss {0};
size_t max_deque_size_pps {0};

static constexpr unsigned long time_between_stats_millis {10 * 1000};
unsigned long accumulated_sd_time_millis {0};
unsigned long last_stats_time_millis {0};
unsigned long working_millis {0};
volatile unsigned long number_imu_samples_logged {0};
volatile unsigned long number_gnss_fixes_logged {0};
volatile unsigned long number_pps_fixes_logged {0};
float effective_imu_logging_rate_hz {0.0f};
float effective_gnss_logging_rate_hz {0.0f};
float effective_pps_logging_rate_hz {0.0f};

struct PPS_fix {
  unsigned long micros_reading;
};

struct GNSS_reading {
  unsigned long micros_reading;
  int32_t latitude;
  int32_t longitude;
  uint32_t posix_timestamp;
  uint32_t microseconds;
  int32_t NED_vel_north;
  int32_t NED_vel_east;
  int32_t NED_vel_down;
  uint8_t fix_type;
};

struct IMU_reading{
  unsigned long micros_reading;
  uint16_t counter;
  int16_t acc_x;
  int16_t acc_y;
  int16_t acc_z;
  int16_t gyr_x;
  int16_t gyr_y;
  int16_t gyr_z;
};

char entry_kind[4];
bool should_log_data = false;

PPS_fix common_isr_pps_fix;
GNSS_reading common_isr_gnss_reading;
IMU_reading common_isr_imu_reading;

etl::deque<PPS_fix, SIZE_DEQUE_PPS> deque_PPS_fixes;
etl::deque<GNSS_reading, SIZE_DEQUE_GNSS> deque_GNSS_readings;
etl::deque<IMU_reading, SIZE_DEQUE_IMU> deque_IMU_readings;

volatile uint32_t ctimer_isr_count {0};
volatile uint16_t imu_isr_count {0};

// ISR handler for CTIMER interrupts
// we use teh CTIMER to generate periodic interrupts for the data logging tasks
extern "C" void am_ctimer_isr(void)
{
  // Get interrupt status and clear
  uint32_t ui32Status = am_hal_ctimer_int_status_get(true);
  am_hal_ctimer_int_clear(ui32Status);

  // Check if it's timer 2A interrupt (reload/overflow)
  if (ui32Status & AM_HAL_CTIMER_INT_TIMERA2)
  {
    // do our work here

    if (ENABLE_DEBUG_FASTPRINT){
      SERIAL_USB->println();
      SERIAL_USB->print(F("|ISR:"));
      SERIAL_USB->print(millis());
      SERIAL_USB->print(F(";"));
    }

    // Poll the ICM-20948 for a fresh accel+gyro sample. The timer fires at 2*ODR so we will
    // typically see dataReady() true on every other ISR tick. No FIFO is configured: getAGMT()
    // reads the latest sample directly.
    if (imu.dataReady()){
      imu.getAGMT();

      common_isr_imu_reading.micros_reading = micros();
      common_isr_imu_reading.counter = imu_isr_count;
      imu_isr_count++;
      common_isr_imu_reading.acc_x = imu.agmt.acc.axes.x;
      common_isr_imu_reading.acc_y = imu.agmt.acc.axes.y;
      common_isr_imu_reading.acc_z = imu.agmt.acc.axes.z;
      common_isr_imu_reading.gyr_x = imu.agmt.gyr.axes.x;
      common_isr_imu_reading.gyr_y = imu.agmt.gyr.axes.y;
      common_isr_imu_reading.gyr_z = imu.agmt.gyr.axes.z;

      if (deque_IMU_readings.full()){
        deque_IMU_readings.pop_front();
      }
      deque_IMU_readings.push_back(common_isr_imu_reading);

      number_imu_samples_logged++;

      if (ENABLE_DEBUG_FASTPRINT){
        SERIAL_USB->print(F("DI;"));
      }
    }

    // if time to read GNSS data, do it and store in deque
    if (ENABLE_GNSS && ctimer_isr_count % (TIMER_DIVIDER_GNSS) == 0){
      // check if we have a new GNSS reading; if yes, push fix to deque
      if (log_GNSS.getPVT()){
        common_isr_gnss_reading.micros_reading = micros();
        common_isr_gnss_reading.latitude = log_GNSS.getLatitude();
        common_isr_gnss_reading.longitude = log_GNSS.getLongitude();
        common_isr_gnss_reading.posix_timestamp = log_GNSS.getUnixEpoch(common_isr_gnss_reading.microseconds);
        common_isr_gnss_reading.NED_vel_north = log_GNSS.getNedNorthVel();
        common_isr_gnss_reading.NED_vel_east = log_GNSS.getNedEastVel();
        common_isr_gnss_reading.NED_vel_down = log_GNSS.getNedDownVel();
        common_isr_gnss_reading.fix_type = log_GNSS.getFixType();

        if (deque_GNSS_readings.full()){
          deque_GNSS_readings.pop_front();
        }

        deque_GNSS_readings.push_back(common_isr_gnss_reading);

        number_gnss_fixes_logged++;

        if (ENABLE_DEBUG_FASTPRINT){
          SERIAL_USB->print(F("DG;"));
        }
    }
  }

  ctimer_isr_count += 1;
  }
}

volatile unsigned long last_pps_micros {0};

void isr_PPS() {
  unsigned long current_micros = micros();

  if (current_micros - last_pps_micros < 500000){
    // debounce: ignore if within 500 ms of last PPS
    // as this may mean we have a bouncing signal
    last_pps_micros = current_micros;
    return;
  }

  last_pps_micros = current_micros;

  common_isr_pps_fix.micros_reading = current_micros;

  if (deque_PPS_fixes.full()){
    deque_PPS_fixes.pop_front();
  }
  deque_PPS_fixes.push_back(common_isr_pps_fix);

  number_pps_fixes_logged++;

  if (ENABLE_DEBUG_FASTPRINT){
    SERIAL_USB->print(F("DP;"));
  }
}

class ConditionChecker {
  public:
    void start(int number_in_a_row);
    bool check(bool condition_met);
  private:
    int number_in_a_row_;
    int count_bad_in_a_row_ {0};
};

void ConditionChecker::start(int number_in_a_row){
  count_bad_in_a_row_ = 0;
  number_in_a_row_ = number_in_a_row;
}

bool ConditionChecker::check(bool condition_met){
  if (!condition_met){
    count_bad_in_a_row_++;
    if (count_bad_in_a_row_ >= number_in_a_row_){
      return false;
    }
  }
  else {
    count_bad_in_a_row_ = 0;
  }

  return true;
}

ConditionChecker sd_not_saturated_checker;

class FrequencyChecker {
  public:
    void start(float expected_frequency_hz, int numbers_in_a_row, float tolerance_percent=30.0f);
    bool check(float current_frequency_hz);

  private:
    float expected_frequency_hz_;
    float tolerance_percent_;
    int numbers_in_a_row_;
    int count_bad_in_a_row_ {0};
};

void FrequencyChecker::start(float expected_frequency_hz, int numbers_in_a_row, float tolerance_percent){
  expected_frequency_hz_ = expected_frequency_hz;
  numbers_in_a_row_ = numbers_in_a_row;
  tolerance_percent_ = tolerance_percent;
}

bool FrequencyChecker::check(float current_frequency_hz){
  if (current_frequency_hz < expected_frequency_hz_ * (1.0 - tolerance_percent_ / 100.0) || current_frequency_hz > expected_frequency_hz_ * (1.0 + tolerance_percent_ / 100.0)){
    SERIAL_USB->print(F("WARNING: frequency out of range! Current: "));
    SERIAL_USB->print(current_frequency_hz);
    SERIAL_USB->print(F(" Hz; expected: "));
    SERIAL_USB->print(expected_frequency_hz_);
    SERIAL_USB->print(F(" Hz; tolerance: +/-"));
    SERIAL_USB->print(tolerance_percent_);
    SERIAL_USB->println(F("%"));

    count_bad_in_a_row_++;

    if (count_bad_in_a_row_ >= numbers_in_a_row_){
      return false;
    }
  }
  else {
    count_bad_in_a_row_ = 0;
  }

  return true;
}

FrequencyChecker gnss_frequency_checker;
FrequencyChecker pps_frequency_checker;
FrequencyChecker imu_frequency_checker;

void setup() {

  gnss_frequency_checker.start(GNSS_FREQUENCY_HZ, 45, 50.0f);
  pps_frequency_checker.start(1.0f, 45, 50.0f);
  imu_frequency_checker.start(IMU_ODR_HZ, 5, 30.0f);
  sd_not_saturated_checker.start(6);

  /////////////////////////////////////////////////////////////////////////////////
  // Initialize watchdog timer
  // Set WDT to 1 Hz, interrupt at 64 ticks, reset at 128 ticks; slow as pre allocate can take quite a while it seems
  wdt.configure(WDT_1HZ, 64, 64);
  wdt.start();

  if (USE_BURSTMODE){
    enableBurstMode();
  }
  
  // Initialize RTC AFTER burst mode is enabled
  // This ensures timing is correctly configured for the 96MHz clock
  board_time_manager.setup_RTC();

  if (ENABLE_BLINK_PWR_LED){
    blink_pwr_led(3);
  }

  pinMode(PIN_STAT_LED, OUTPUT);

  /////////////////////////////////////////////////////////////////////////////////
  // Initialize serial over USB
  SERIAL_USB->begin(BAUD_RATE_USB);
  while (!(*SERIAL_USB) && millis() < SERIAL_TIMEOUT_MS);

  /////////////////////////////////////////////////////////////////////////////////
  // Startup delay to allow time for serial monitor connection, uploading new firmware, etc.
  SERIAL_USB->println();
  SERIAL_USB->println(F("startup delay..."));
  wdt.restart();
  delay(500);
  wdt.restart();
  SERIAL_USB->println(F("... done"));
  SERIAL_USB->println();

  /////////////////////////////////////////////////////////////////////////////////
  // Print firmware configuration
  print_firmware_config();
  SERIAL_USB->println();

  boot_counter_instance.increment_boot_number();
  SERIAL_USB->print(F("Boot count: "));
  SERIAL_USB->println(boot_counter_instance.get_boot_number());
  delay(100);
  wdt.restart();

  /////////////////////////////////////////////////////////////////////////////////
  // Print boot count and offer to reset it
  // If the user presses 'y' within 5 seconds, reset the boot count
  // Otherwise, keep the current boot count
  if (ENABLE_BOOT_COUNTER){
    SERIAL_USB->println(F("Press y to reset boot count... "));
    wdt.restart();
    unsigned long startTime = millis();
    bool resetRequested = false;
    while (millis() - startTime < 5000) {
      wdt.restart();
      if (SERIAL_USB->available()) {
        char c = SERIAL_USB->read();
        if (c == 'y' || c == 'Y') {
          resetRequested = true;
          break;
        }
      }
    }
    if (resetRequested) {
      boot_counter_instance.set_boot_number(0);
      SERIAL_USB->println(F("Boot count reset."));
    } else {
      SERIAL_USB->println(F("No reset requested."));
      wdt.restart();
    }
    SERIAL_USB->println();
  }

  if (ENABLE_BLINK_PWR_LED){
    blink_pwr_led(5);
  }

  /////////////////////////////////////////////////////////////////////////////////
  // Initialize time manager from GNSS
  // Keep trying to get a valid GNSS fix until successful
  // As long as we do not have a valid fix, sleep for a while between attempts
  board_time_manager.set_posix_timestamp(0);
  board_time_manager.print_status();
  SERIAL_USB->println();
  if (ENABLE_GNSS_START){
    bool got_valid_fix = false;
    while (!got_valid_fix) {
      SERIAL_USB->println(F("Attempting to get initial GNSS fix..."));
      wdt.restart();
      got_valid_fix = gnss_manager.get_a_fix(timeout_initial_fix_gnss_seconds, false, true, false);
      if (!got_valid_fix) {
        SERIAL_USB->println(F("Failed to get GNSS fix. Sleep and retry..."));
        turn_gnss_off();
        wdt.restart();
        sleep_for_seconds(sleep_no_initial_gnss_fix_seconds);
        blink_stat_led(3);
      }
      else {
        SERIAL_USB->println(F("Successfully obtained GNSS fix."));
        SERIAL_USB->println(F("Get a few fixes to make sure the quality is good before setting clock..."));
        for (int i=0; i<10; i++) {
          wdt.restart();
          gnss_manager.get_a_fix(10, false, false, false);
        }
        gnss_manager.get_a_fix(10, true, false, false);
        // Visual confirmation: UTC has been synced from GNSS. 5 STAT-LED blinks
        // before we proceed to IMU/SD setup and start logging. Useful when the
        // device is outside and you can't see the serial monitor.
        SERIAL_USB->println(F("UTC synced from GNSS — blinking STAT LED 5x"));
        blink_stat_led(5);
      }
    }
    board_time_manager.print_status();
    SERIAL_USB->println();
  }

  if (ENABLE_BLINK_PWR_LED){
    blink_pwr_led(7);
  }

  /////////////////////////////////////////////////////////////////////////////////

  int start_attempt {1};
  bool setup_successful = false;

  while (start_attempt<=5){
    SERIAL_USB->print(F("Setup attempt #: "));
    SERIAL_USB->println(start_attempt);
    start_attempt += 1;

    wdt.restart();
    delay(10);

    ////////////////////////////////////////////////////
    // start I2C port

    SERIAL_USB->println(F("Starting I2C QWIIC..."));
    pinMode(PIN_QWIIC_PWR, OUTPUT);
    digitalWrite(PIN_QWIIC_PWR, LOW); // Ensure power is off before starting
    delay(1000); // Wait for power to stabilize
    wdt.restart();
    digitalWrite(PIN_QWIIC_PWR, HIGH); 
    delay(100);
    wdt.restart();

    I2C_QWIIC->begin();
    delay(100);
    wdt.restart();
    I2C_QWIIC->setClock(400000);
    delay(100);
    wdt.restart();
    SERIAL_USB->println(F("I2C QWIIC started"));

    ////////////////////////////////////////////////////
    // start and set up GNSS itself

    if (ENABLE_GNSS){
      if (!log_GNSS.begin(*I2C_QWIIC)){
          SERIAL_USB->println(F("problem starting GNSS"));

          I2C_QWIIC->end();
          delay(500);
          continue;
      }
      SERIAL_USB->println(F("success starting GNSS"));

      log_GNSS.setI2COutput(COM_TYPE_UBX);
      delay(100);
      wdt.restart();
      SERIAL_USB->println(F("GNSS set to UBX output"));
      delay(100);

      log_GNSS.setAutoPVT(true);
      log_GNSS.setNavigationFrequency(GNSS_FREQUENCY_HZ);
      delay(100);
      wdt.restart();
      uint8_t rate = log_GNSS.getNavigationFrequency();
      SERIAL_USB->print("Current update rate: ");
      SERIAL_USB->println(rate);

      // wait until we get a fix
      if (ENABLE_GNSS_START){
        bool fix_obtained {false};
        static constexpr unsigned long GNSS_FIX_WAIT_TIMEOUT_MS = 1000 * 60 * 2;
        unsigned long start_wait_ms = millis();
        SERIAL_USB->println(F("Waiting for GNSS fix..."));
        while (millis() - start_wait_ms < GNSS_FIX_WAIT_TIMEOUT_MS){
          if (log_GNSS.getFixType() >= 3){
            fix_obtained = true;
            SERIAL_USB->println(F("GNSS fix acquired."));
            break;
          }
          delay(500);
          wdt.restart();
          SERIAL_USB->print(F("."));
        }

        if (!fix_obtained){
          SERIAL_USB->println();
          SERIAL_USB->println(F("Failed to obtain GNSS fix in time."));
          continue;
        }
      }

      SERIAL_USB->println(F("GNSS setup complete."));
      wdt.restart();

      ////////////////////////////////////////////////////
      // I would prefer a PULLDOWN but for some reason it does not work
      // this should not matter: PULLUP should protect us anyways if floating,
      // and the PULLUP is not harmful as it is weak enough that this gets fully driven
      // by the GNSS PPS output
      pinMode(PIN_LOG_PPS, INPUT_PULLUP);
      attachInterrupt(PIN_LOG_PPS, isr_PPS, RISING);
    } else {
      SERIAL_USB->println(F("GNSS disabled at compile time, skipping GNSS setup."));
    }

    ////////////////////////////////////////////////////
    // start and set up the built-in ICM-20948 over SPI

    // ----------------------------------------------------------------
    // CRITICAL ORDER on the OLA: power on the microSD card module BEFORE
    // the IMU init. The SD card module provides a hardware pull-up on
    // MISO (shared with the IMU). Without SD powered, MISO rise time is
    // too slow at 4 MHz and the ICM-20948 WHO_AM_I read comes back
    // corrupted as 0xE0 instead of 0xEA. SparkFun's reference OLA
    // firmware always calls beginSD() before beginIMU() for this reason.
    // ----------------------------------------------------------------
    SERIAL_USB->println(F("Powering up microSD card (for shared SPI MISO pull-up)..."));
    pinMode(SD_PWR, OUTPUT);
    am_hal_gpio_pinconfig(SD_PWR, g_AM_HAL_GPIO_OUTPUT);
    digitalWrite(SD_PWR, LOW);   // SD power is active-LOW (LOW = ON)
    pinMode(SD_CS_PIN, OUTPUT);
    digitalWrite(SD_CS_PIN, HIGH);  // deselect SD so it tri-states its MISO
    delay(50);                      // SD card needs time to power up
    wdt.restart();

    SERIAL_USB->println(F("Powering up ICM-20948..."));
    // SparkFun's OLA reference forcibly resets pad funcsel to GPIO for these two pads.
    // Without this, pinMode() alone may not override alternate functions on some pads.
    pinMode(PIN_IMU_CHIP_SELECT, OUTPUT);
    am_hal_gpio_pinconfig(PIN_IMU_CHIP_SELECT, g_AM_HAL_GPIO_OUTPUT);
    digitalWrite(PIN_IMU_CHIP_SELECT, HIGH); // deselect before powering
    pinMode(PIN_IMU_POWER, OUTPUT);
    am_hal_gpio_pinconfig(PIN_IMU_POWER, g_AM_HAL_GPIO_OUTPUT);
    digitalWrite(PIN_IMU_POWER, LOW);  // ensure power is off
    delay(10);
    digitalWrite(PIN_IMU_POWER, HIGH); // power on
    delay(100);                        // SparkFun reference firmware waits 100ms before talking SPI
    wdt.restart();

    SPI.begin();

    // Enable Apollo3 internal 1.5KΩ pull-up on MISO (pad 6). Combined with
    // the SD card module's MISO pull-up (above), this gives MISO enough
    // drive strength to rise cleanly at 4 MHz SPI.
    {
      am_hal_gpio_pincfg_t cipoPinCfg = g_AM_BSP_GPIO_IOM0_MISO;
      cipoPinCfg.ePullup = AM_HAL_GPIO_PIN_PULLUP_1_5K;
      am_hal_gpio_pinconfig(6 /* MISO pad */, cipoPinCfg);
    }

    SERIAL_USB->println(F("Starting ICM-20948..."));
    for (int imu_try = 0; imu_try < 3; imu_try++) {
      imu.begin(PIN_IMU_CHIP_SELECT, SPI, IMU_SPI_MHZ * 1000000UL);
      if (imu.status == ICM_20948_Stat_Ok) break;
      delay(10);
      wdt.restart();
    }
    if (imu.status != ICM_20948_Stat_Ok){
      SERIAL_USB->print(F("problem starting ICM-20948: "));
      SERIAL_USB->println(imu.statusString());

      digitalWrite(PIN_IMU_POWER, LOW);
      detachInterrupt(PIN_LOG_PPS);
      log_GNSS.end();
      I2C_QWIIC->end();
      delay(500);
      continue;
    }
    SERIAL_USB->println(F("success starting ICM-20948"));
    delay(50);
    wdt.restart();

    // Make sure we are in a known state, then wake up
    imu.swReset();
    delay(50);
    imu.sleep(false);
    imu.lowPower(false);

    // Continuous sampling for both accelerometer and gyroscope
    imu.setSampleMode((ICM_20948_Internal_Acc | ICM_20948_Internal_Gyr), ICM_20948_Sample_Mode_Continuous);

    // Full-scale ranges: keep close to the original ISM330DHCX 2g / 125dps configuration.
    // ICM-20948 minimum gyro range is 250dps, so we pick that.
    ICM_20948_fss_t fss;
    fss.a = gpm2;
    fss.g = dps250;
    imu.setFullScale((ICM_20948_Internal_Acc | ICM_20948_Internal_Gyr), fss);

    // Enable the digital low-pass filter at moderate bandwidth (chosen to be well above IMU_ODR_HZ / 2
    // so it does not attenuate the signal at the chosen ODR).
    ICM_20948_dlpcfg_t dlpcfg;
    dlpcfg.a = acc_d111bw4_n136bw;
    dlpcfg.g = gyr_d119bw5_n154bw3;
    imu.setDLPFcfg((ICM_20948_Internal_Acc | ICM_20948_Internal_Gyr), dlpcfg);
    imu.enableDLPF(ICM_20948_Internal_Acc, true);
    imu.enableDLPF(ICM_20948_Internal_Gyr, true);

    // Sample rate divider — ICM-20948 internal clock is 1125 Hz, so div=4 gives 225 Hz
    ICM_20948_smplrt_t smplrt;
    smplrt.a = IMU_SMPLRT_DIV;
    smplrt.g = IMU_SMPLRT_DIV;
    imu.setSampleRate((ICM_20948_Internal_Acc | ICM_20948_Internal_Gyr), smplrt);
    wdt.restart();

    if (imu.status != ICM_20948_Stat_Ok){
      SERIAL_USB->print(F("ICM-20948 configuration error: "));
      SERIAL_USB->println(imu.statusString());
    }

    // Sensitivity values matching the chosen full-scale ranges. ICM-20948 datasheet:
    //   accel  +/-2g    -> 16384 LSB/g  -> 0.061035 mg/LSB
    //   gyro +/-250dps -> 131 LSB/dps -> 7.633588 mdps/LSB
    acc_sensitivity = 1000.0f / 16384.0f;
    gyr_sensitivity = 1000.0f / 131.0f;
    SERIAL_USB->print(F("ICM-20948 Acc sensitivity (mg/LSB): "));
    SERIAL_USB->println(acc_sensitivity, 6);
    SERIAL_USB->print(F("ICM-20948 Gyr sensitivity (mdps/LSB): "));
    SERIAL_USB->println(gyr_sensitivity, 6);

    SERIAL_USB->println(F("ICM-20948 setup complete."));

    ////////////////////////////////////////////////////
    SERIAL_USB->println(F("Configuring (but NOT yet starting) IMU sample timer..."));

    // Power up the clock
    am_hal_clkgen_control(AM_HAL_CLKGEN_CONTROL_SYSCLK_MAX, 0);

    // Stop timer
    am_hal_ctimer_stop(TIMER_NUM, AM_HAL_CTIMER_TIMERA);

    // Clear timer
    am_hal_ctimer_clear(TIMER_NUM, AM_HAL_CTIMER_TIMERA);

    // Configure timer in REPEAT mode with 187.5 kHz source clock
    am_hal_ctimer_config_single(TIMER_NUM, AM_HAL_CTIMER_TIMERA,
                                (AM_HAL_CTIMER_FN_REPEAT |
                                  AM_HAL_CTIMER_HFRC_187_5KHZ |
                                  AM_HAL_CTIMER_INT_ENABLE));

    // Set the period for the timer
    static constexpr uint32_t period = 187500 / TIMER_FREQ_HZ;
    static_assert(period < 0xFFFF, "Timer period too large for 16-bit timer");
    static_assert(period > 1, "Timer period must be greater than one");
    am_hal_ctimer_period_set(TIMER_NUM, AM_HAL_CTIMER_TIMERA, period, 0);

    // Clear any pending interrupts
    am_hal_ctimer_int_clear(AM_HAL_CTIMER_INT_TIMERA2);

    // NOTE: timer interrupts and timer start are DEFERRED until after SD card
    // is initialized. The IMU ISR does SPI transactions; if it fires during
    // SdFat init it corrupts SD's command/response sequence and SD init fails.
    wdt.restart();

    setup_successful = true;
    break;
  }

  // Check if setup succeeded
  if (!setup_successful) {
    SERIAL_USB->println(F("FATAL ERROR: Hardware setup failed after all retry attempts!"));
    SERIAL_USB->println(F("Entering infinite loop to trigger watchdog reset..."));
    while (true) {
      delay(1000);
      NVIC_SystemReset();
      // Watchdog will reset the board
    }
  }

  /////////////////////////////////////////////////////////////////////////////////
  // log forever

  deque_PPS_fixes.clear();
  deque_GNSS_readings.clear();
  deque_IMU_readings.clear();

  // Start doing the logging to the SD card
  if (!sd_card_manager.start()) {
    SERIAL_USB->println(F("FATAL ERROR: Failed to start SD card manager!"));
    SERIAL_USB->println(F("Entering infinite loop to trigger watchdog reset..."));
    while (true) {
      delay(1000);
      NVIC_SystemReset();
      // Watchdog will reset the board
    }
  }
  wdt.restart();

  // SD is up. NOW it's safe to enable the IMU sample timer and start it.
  // (Doing this earlier means the IMU ISR's SPI traffic collides with SdFat's
  // SD init traffic on the same shared SPI bus, breaking SD init.)
  SERIAL_USB->println(F("Starting IMU sample timer..."));
  am_hal_ctimer_int_enable(AM_HAL_CTIMER_INT_TIMERA2);
  NVIC_EnableIRQ(CTIMER_IRQn);
  am_hal_ctimer_start(TIMER_NUM, AM_HAL_CTIMER_TIMERA);
  am_hal_interrupt_master_enable();
  SERIAL_USB->println(F("Timer started!"));
  wdt.restart();

  uint32_t posix_timestamp;
  uint32_t posix_timestamp_next_file;

  // Local stack variables for main loop to avoid race conditions with ISRs
  PPS_fix local_pps_fix;
  GNSS_reading local_gnss_reading;
  IMU_reading local_imu_reading;

  while (true){
    // create a new file
    SERIAL_USB->println();
    SERIAL_USB->println(F("Preparing to start new log file..."));

    if (sd_card_manager.preallocate_and_open_file(PREALLOCATE_LOGFILE_SIZE_BYTES, USE_FOLDERS)) {
      SERIAL_USB->println(F("Log file opened and preallocated successfully."));
    } else {
      SERIAL_USB->println(F("ERROR: Failed to open and preallocate log file on SD card."));
      delay(1000);
      sd_card_manager.close_and_sync_file();
      delay(1000);
      sd_card_manager.stop();
      delay(1000);
      while (true) {
        delay(1000);
        NVIC_SystemReset();
      }
      break;
    }
    wdt.restart();

    // we do a new file every time UTC times hits 0 minutes modulo 15 minutes
    posix_timestamp = board_time_manager.get_posix_timestamp();
    SERIAL_USB->print(F("Current posix timestamp: "));
    SERIAL_USB->println(posix_timestamp);
    posix_timestamp_next_file = posix_timestamp - (posix_timestamp % seconds_in_15_minutes) + seconds_in_15_minutes;
    SERIAL_USB->print(F("Next log file posix timestamp: "));
    SERIAL_USB->println(posix_timestamp_next_file);
    wdt.restart();
    SERIAL_USB->println(F("Logging..."));
    SERIAL_USB->println();

    SERIAL_USB->print(str_start_logging);
    sd_card_manager.write_buffer(reinterpret_cast<const uint8_t*>(str_start_logging), sizeof(str_start_logging)-1);

    // write header with configuration info

    // firmware version info: commit ID used
    for (size_t i=0; i<sizeof(working_buffer); i++){
      working_buffer[i] = '\0';
    }
    snprintf(working_buffer, sizeof(working_buffer),
             "Firmware commit ID: %s\n", commit_id);
    SERIAL_USB->print(working_buffer);
    sd_card_manager.write_buffer(reinterpret_cast<const uint8_t*>(working_buffer), strlen(working_buffer));

    // sensitivity IMU ACC
    for (size_t i=0; i<sizeof(working_buffer); i++){
      working_buffer[i] = '\0';
    }
    snprintf(working_buffer, sizeof(working_buffer),
             "ICM-20948 Acc sensitivity (mg/LSB): %.6f\n", acc_sensitivity);
    SERIAL_USB->print(working_buffer);
    sd_card_manager.write_buffer(reinterpret_cast<const uint8_t*>(working_buffer), strlen(working_buffer));

    // sensitivity IMU GYR
    for (size_t i=0; i<sizeof(working_buffer); i++){
      working_buffer[i] = '\0';
    }
    snprintf(working_buffer, sizeof(working_buffer),
             "ICM-20948 Gyr sensitivity (mdps/LSB): %.6f\n", gyr_sensitivity);
    SERIAL_USB->print(working_buffer);
    sd_card_manager.write_buffer(reinterpret_cast<const uint8_t*>(working_buffer), strlen(working_buffer));

    // sample rate IMU
    for (size_t i=0; i<sizeof(working_buffer); i++){
      working_buffer[i] = '\0';
    }
    snprintf(working_buffer, sizeof(working_buffer),
             "ICM-20948 ODR (Hz): %.2f\n", IMU_ODR_HZ);
    SERIAL_USB->print(working_buffer);
    sd_card_manager.write_buffer(reinterpret_cast<const uint8_t*>(working_buffer), strlen(working_buffer));

    // sample rate GNSS
    for (size_t i=0; i<sizeof(working_buffer); i++){
      working_buffer[i] = '\0';
    }
    snprintf(working_buffer, sizeof(working_buffer),
             "GNSS update rate (Hz): %d\n", GNSS_FREQUENCY_HZ);
    SERIAL_USB->print(working_buffer);
    sd_card_manager.write_buffer(reinterpret_cast<const uint8_t*>(working_buffer), strlen(working_buffer));

    wdt.restart();

    last_stats_time_millis = millis();
    accumulated_sd_time_millis = 0;
    
    // Reset counters atomically
    am_hal_interrupt_master_disable();
    number_imu_samples_logged = 0;
    number_gnss_fixes_logged = 0;
    number_pps_fixes_logged = 0;
    am_hal_interrupt_master_enable();

    while (board_time_manager.get_posix_timestamp() < posix_timestamp_next_file){
      should_log_data = false;

      // log
      // the logging from sensors to dequeues buffers is taken care of by the ISR driven routines

      // if there are data on the dequeue buffers, write them to SD card
      // for each of the deques:
      //   - turn off interrupts as the deques are shared with ISRs
      //   - pop from the deques into the local buffer to make ready to write
      //   - turn on interrupts
      //   - write the local buffer to SD card

      if (millis() - last_stats_time_millis >= time_between_stats_millis){
        last_stats_time_millis = millis();

        // Read counters atomically
        am_hal_interrupt_master_disable();
        unsigned long imu_samples = number_imu_samples_logged;
        unsigned long gnss_fixes = number_gnss_fixes_logged;
        unsigned long pps_fixes = number_pps_fixes_logged;
        number_imu_samples_logged = 0;
        number_gnss_fixes_logged = 0;
        number_pps_fixes_logged = 0;
        am_hal_interrupt_master_enable();

        // compute effective logging rates
        effective_imu_logging_rate_hz = (imu_samples * 1000.0f) / (time_between_stats_millis);
        effective_gnss_logging_rate_hz = (gnss_fixes * 1000.0f) / (time_between_stats_millis);
        effective_pps_logging_rate_hz = (pps_fixes * 1000.0f) / (time_between_stats_millis);

        SERIAL_USB->println();

        board_time_manager.print_status();
        SERIAL_USB->print(F("millis(): "));
        SERIAL_USB->print(millis());
        SERIAL_USB->print(F("; seconds since boot: "));
        SERIAL_USB->println(millis() / 1000);

        SERIAL_USB->print(F("Samples logged in last interval: "));
        SERIAL_USB->print(F("IMU: "));
        SERIAL_USB->print(imu_samples);
        SERIAL_USB->print(F("; GNSS: "));
        SERIAL_USB->print(gnss_fixes);
        SERIAL_USB->print(F("; PPS: "));
        SERIAL_USB->println(pps_fixes);

        SERIAL_USB->print(F("Max deque sizes reached: "));
        SERIAL_USB->print(F("IMU: "));
        SERIAL_USB->print(max_deque_size_imu);
        SERIAL_USB->print(F(" over "));
        SERIAL_USB->print(SIZE_DEQUE_IMU);
        SERIAL_USB->print(F("; GNSS: "));
        SERIAL_USB->print(max_deque_size_gnss);
        SERIAL_USB->print(F(" over "));
        SERIAL_USB->print(SIZE_DEQUE_GNSS);
        SERIAL_USB->print(F("; PPS: "));
        SERIAL_USB->print(max_deque_size_pps);
        SERIAL_USB->print(F(" over "));
        SERIAL_USB->println(SIZE_DEQUE_PPS);

        max_deque_size_imu = 0;
        max_deque_size_gnss = 0;
        max_deque_size_pps = 0;
 
        SERIAL_USB->print(F("Effective logging rates (Hz): "));
        SERIAL_USB->print(F("IMU (Hz): "));
        SERIAL_USB->print(effective_imu_logging_rate_hz, 2);
        SERIAL_USB->print(F("; GNSS (Hz): "));
        SERIAL_USB->print(effective_gnss_logging_rate_hz, 2);
        SERIAL_USB->print(F("; PPS (Hz): "));
        SERIAL_USB->println(effective_pps_logging_rate_hz, 2);
 
        SERIAL_USB->print(F("Accumulated SD time (ms): "));
        SERIAL_USB->print(accumulated_sd_time_millis);
        SERIAL_USB->print(F(" ms over "));
        SERIAL_USB->print(time_between_stats_millis);
        SERIAL_USB->println(F(" ms interval"));

        if (!imu_frequency_checker.check(effective_imu_logging_rate_hz)){
          SERIAL_USB->println(F("ERROR: Effective IMU logging frequency is out of expected range!"));
          sd_card_manager.close_and_sync_file();
          delay(2000);
          NVIC_SystemReset();
        }
        if (ENABLE_GNSS && !gnss_frequency_checker.check(effective_gnss_logging_rate_hz)){
          SERIAL_USB->println(F("ERROR: Effective GNSS logging frequency is out of expected range!"));
          sd_card_manager.close_and_sync_file();
          delay(2000);
          NVIC_SystemReset();
        }
        if (ENABLE_GNSS && !pps_frequency_checker.check(effective_pps_logging_rate_hz)){
          SERIAL_USB->println(F("ERROR: Effective PPS logging frequency is out of expected range!"));
          sd_card_manager.close_and_sync_file();
          delay(2000);
          NVIC_SystemReset();
        }

        bool sd_not_saturated = accumulated_sd_time_millis < 8000;
        if (!sd_not_saturated_checker.check(sd_not_saturated)){
          SERIAL_USB->println(F("ERROR: SD card logging is taking too much time!"));
          sd_card_manager.close_and_sync_file();
          delay(2000);
          NVIC_SystemReset();
        }

        accumulated_sd_time_millis = 0;

      }

      if (ENABLE_GNSS){
        // with the GNSS PPS deque
        am_hal_interrupt_master_disable();

        working_deque_size = deque_PPS_fixes.size();
        if (working_deque_size > 0){
          should_log_data = true;
          local_pps_fix = deque_PPS_fixes.front();
          deque_PPS_fixes.pop_front();
        }

        am_hal_interrupt_master_enable();

        if (should_log_data){
          entry_kind[0] = '\n';
          entry_kind[1] = 'P';
          entry_kind[2] = 'P';
          entry_kind[3] = 'S';
          working_millis = millis();
          sd_card_manager.write_buffer(reinterpret_cast<const uint8_t*>(entry_kind), sizeof(entry_kind));
          sd_card_manager.write_buffer(reinterpret_cast<const uint8_t*>(&local_pps_fix), sizeof(local_pps_fix));
          accumulated_sd_time_millis += (millis() - working_millis);
          should_log_data = false;
        }

        if (working_deque_size > max_deque_size_pps){
          max_deque_size_pps = working_deque_size;
        }

        // with the GNSS fixes deque
        am_hal_interrupt_master_disable();

        working_deque_size = deque_GNSS_readings.size();
        if (working_deque_size > 0){
          should_log_data = true;
          local_gnss_reading = deque_GNSS_readings.front();
          deque_GNSS_readings.pop_front();
        }

        am_hal_interrupt_master_enable();

        if (should_log_data){
          entry_kind[0] = '\n';
          entry_kind[1] = 'G';
          entry_kind[2] = 'P';
          entry_kind[3] = 'S';
          working_millis = millis();
          sd_card_manager.write_buffer(reinterpret_cast<const uint8_t*>(entry_kind), sizeof(entry_kind));
          sd_card_manager.write_buffer(reinterpret_cast<const uint8_t*>(&local_gnss_reading), sizeof(local_gnss_reading));
          accumulated_sd_time_millis += (millis() - working_millis);
          should_log_data = false;
        }

        if (working_deque_size > max_deque_size_gnss){
          max_deque_size_gnss = working_deque_size;
        }
      }

      // with the IMU deque
      am_hal_interrupt_master_disable();

      working_deque_size = deque_IMU_readings.size();
      if (working_deque_size > 0){
        should_log_data = true;
        local_imu_reading = deque_IMU_readings.front();
        deque_IMU_readings.pop_front();
      }

      am_hal_interrupt_master_enable();

      if (should_log_data){
        entry_kind[0] = '\n';
        entry_kind[1] = 'I';
        entry_kind[2] = 'M';
        entry_kind[3] = 'U';
        working_millis = millis();
        sd_card_manager.write_buffer(reinterpret_cast<const uint8_t*>(entry_kind), sizeof(entry_kind));
        sd_card_manager.write_buffer(reinterpret_cast<const uint8_t*>(&local_imu_reading), sizeof(local_imu_reading));
        accumulated_sd_time_millis += (millis() - working_millis);
        should_log_data = false;
      }

      if (working_deque_size > max_deque_size_imu){
        max_deque_size_imu = working_deque_size;
      }

      wdt.restart();
    }

    sd_card_manager.write_buffer(reinterpret_cast<const uint8_t*>(str_stop_logging), sizeof(str_stop_logging)-1);
    wdt.restart();
    // time to close the file and start logging a new one
    SERIAL_USB->println();
    SERIAL_USB->println(F("Time to start new log file"));

    uint64_t file_size = sd_card_manager.get_file()->size();
    SERIAL_USB->print(F("Final log file size (KBytes): "));
    SERIAL_USB->println((uint32_t) (file_size / 1024));

    uint64_t available_size = sd_card_manager.get_file()->available();
    SERIAL_USB->print(F("Final log file available size remaining (KBytes): "));
    SERIAL_USB->println((uint32_t) (available_size / 1024));
    
    wdt.restart();

    sd_card_manager.close_and_sync_file();
    wdt.restart();

    delay(10);
  }

  /////////////////////////////////////////////////////////////////////////////////
  // if we reach here, we have an issue:
  // stop SD card manager and let watchdog reset the board

  SERIAL_USB->println(F("Stopping logging due to error..."));
  sd_card_manager.close_and_sync_file();
  delay(5000);
  sd_card_manager.stop();
  wdt.restart();

  pinMode(PIN_QWIIC_PWR, OUTPUT);
  digitalWrite(PIN_QWIIC_PWR, LOW);

  delay(1000);

  SERIAL_USB->println(F("Entering infinite loop to trigger watchdog reset..."));

  while (true)
  {
    // reboot
    delay(1000);
    NVIC_SystemReset();
  }

}

void loop() {
  // we should never get here
  // if we get here, the watchdog will eventually restart the board
  delay(1000);
  NVIC_SystemReset();
}
