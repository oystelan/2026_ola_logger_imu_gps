#include <Arduino.h>

#include "firmware_configuration.h"
#include "watchdog_manager.h"
#include "boot_counter.h"
#include "calibration_manager.h"
#include "time_manager.h"
#include "gnss_manager.h"
#include "sleep_manager.h"
#include "sd_card_manager.h"
#include "file_transfer.h"

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
static constexpr bool ENABLE_GNSS_START = true;              ///< If true AND a GNSS module is detected at boot, wait for a valid fix before starting sampling
static constexpr bool ENABLE_DEBUG_FASTPRINT = false;

// --- Runtime state populated during boot ---
// True iff log_GNSS.begin() succeeded at boot. Drives whether we wait for a
// GNSS fix and whether we treat GNSS UTC as authoritative for the RTC sync.
static bool g_gnss_present = false;

// True iff the boot path successfully seeded the SW posix_timestamp counter
// from the H/W RTC (i.e. a coin cell preserved the value across power-off).
// Used to know whether a drift check vs GNSS is meaningful at first fix.
static bool g_posix_seeded_from_hw_rtc = false;

// Drift marker: when GNSS first locks AND the H/W-RTC-seeded SW counter
// disagrees with GNSS by more than 1 second, set this to 2. The GNSS-write
// path in the ISR then forces posix_timestamp=0 on that many subsequent
// entries, producing a visible 1970 spike in the data file at the moment of
// RTC re-sync — so any plot of posix_timestamp shows exactly where the
// pre-sync timestamps stopped being trustworthy.
static volatile uint8_t g_drift_marker_remaining = 0;

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
  int32_t altitude_msl_mm;
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
  // Magnetometer (AK09916, inside the ICM-20948) — runs at its own internal
  // ~100 Hz, so at our 225 Hz IMU rate ~half the records have repeated mag
  // values (just the latest cached read). The library's getAGMT() already
  // fetches mag, so adding these fields is essentially free.
  int16_t mag_x;
  int16_t mag_y;
  int16_t mag_z;
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

// Gyro bias (body frame, int16 LSB) subtracted in the ISR before logging.
// Populated from EEPROM via calibration_manager at boot; 0 if uncalibrated.
volatile int16_t g_gyro_bias_x {0};
volatile int16_t g_gyro_bias_y {0};
volatile int16_t g_gyro_bias_z {0};

// Mag hard-iron offset (raw chip LSB) subtracted in the ISR before logging.
// Same convention as gyro: populated from EEPROM at boot; 0 if uncalibrated.
volatile int16_t g_mag_bias_x {0};
volatile int16_t g_mag_bias_y {0};
volatile int16_t g_mag_bias_z {0};

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

    // Drain the ICM-20948 FIFO. The chip buffers ACCEL samples in its 4 KB
    // FIFO at the configured ODR (225 Hz) while the MCU is busy (e.g. during
    // SD writes when the CTIMER ISR is masked by SDIrqGuard). When this ISR
    // resumes, we pull every accumulated sample out in one batch — no data
    // loss across SD gaps.
    //
    // ACCEL ONLY in FIFO — 6 bytes per sample:
    //   ACCEL_X_H, ACCEL_X_L, ACCEL_Y_H, ACCEL_Y_L, ACCEL_Z_H, ACCEL_Z_L
    // (signed 16-bit big-endian, chip frame; remapped to body below.)
    // Why not also gyro+mag? Because their ODRs are slightly different
    // (gyro 1100/(1+div), accel 1125/(1+div)) — when both are in FIFO the
    // chip writes each sensor's bytes independently as they become ready
    // and they desync over time, corrupting the byte→sensor mapping.
    // See BOOT_000292 ~28-37 s where accel data appeared in the gyro slot.
    //
    // Gyro and mag are read via getAGMT() once at the START of each ISR
    // drain — i.e. their effective sample rate is the ISR rate. For the
    // basin / vertical-AHRS workflow this is more than enough (slow
    // motion). All N accel samples in a batch share the same gyro+mag.
    //
    // Timing: monotonic counter (next_sample_us += dt per sample) with a
    // ±100 ms resync against micros() to bound drift, anchored on the
    // first batch.
    {
      static constexpr uint8_t FIFO_SAMPLE_BYTES = 6;
      static constexpr unsigned long FIFO_SAMPLE_DT_US =
          (unsigned long)(1000000.0f / IMU_ODR_HZ + 0.5f);

      uint16_t fifo_bytes = 0;
      imu.getFIFOcount(&fifo_bytes);
      uint16_t n_samples = fifo_bytes / FIFO_SAMPLE_BYTES;
      // Cap at the physical FIFO size (4096 B / 6 B-per-sample ≈ 682). If
      // getFIFOcount() ever returns an impossibly large value (observed in
      // BOOT_000307: ~43k bytes → n_samples ~7200 → 32-second backward
      // resync jump), clamp it so the timestamp logic stays sane.
      static constexpr uint16_t FIFO_MAX_SAMPLES = 4096 / FIFO_SAMPLE_BYTES;
      if (n_samples > FIFO_MAX_SAMPLES) n_samples = FIFO_MAX_SAMPLES;
      if (n_samples > 0){
        // Snapshot gyro + mag direct registers ONCE for this whole batch.
        // getAGMT() reads accel too — wasted SPI, but harmless and keeps
        // the API simple. We ignore the accel values from agmt and use the
        // FIFO bytes for accel below.
        imu.getAGMT();
        int16_t gx_chip = imu.agmt.gyr.axes.x;
        int16_t gy_chip = imu.agmt.gyr.axes.y;
        int16_t gz_chip = imu.agmt.gyr.axes.z;
        int16_t mx_chip = imu.agmt.mag.axes.x;
        int16_t my_chip = imu.agmt.mag.axes.y;
        int16_t mz_chip = imu.agmt.mag.axes.z;

        // ---- chip → PCB body GYRO remap ----
        // GYRO die orientation = PCB silkscreen orientation, all three axes
        // (verified by rotation test). Accel die is in a DIFFERENT orientation
        // and gets the cyclic remap further down.
        //   PCB_gx = +chip_gx
        //   PCB_gy = +chip_gy
        //   PCB_gz = +chip_gz
        // Bias is stored in PCB body frame (= raw chip frame here); re-run gyro
        // cal (quadruple-tap RESET) after flashing — any previously stored bias
        // was in the wrong frame.
        int16_t gx = (int16_t)(gx_chip - g_gyro_bias_x);
        int16_t gy = (int16_t)(gy_chip - g_gyro_bias_y);
        int16_t gz = (int16_t)(gz_chip - g_gyro_bias_z);

        // Mag axis mapping is chip-frame pass-through — verified by the
        // orientation test (BOOT_000362): chip-mx/my/mz already match the
        // PCB silkscreen mag arrows in both direction and sign. Only the
        // hard-iron offset is subtracted; the full y/z flip relative to the
        // accel/gyro silkscreen cross is handled downstream.
        int16_t mx = (int16_t)(mx_chip - g_mag_bias_x);
        int16_t my = (int16_t)(my_chip - g_mag_bias_y);
        int16_t mz = (int16_t)(mz_chip - g_mag_bias_z);

        unsigned long now_us = micros();
        uint8_t buf[FIFO_SAMPLE_BYTES];

        // Monotonic timestamp counter. Guarantees:
        //   (a) micros_reading strictly increases sample-to-sample (no
        //       backward jumps even if getFIFOcount() returns garbage)
        //   (b) drift vs micros() is bounded (snap forward on lag)
        // First batch: anchor at now_us. On lag > 100 ms: snap forward to
        // now_us, NEVER backward. The trade-off: when the chip has
        // accumulated samples during a long SD-block, their timestamps get
        // clustered near the recovery moment rather than spread back over
        // when they were taken — but they stay monotonic, and the AHRS
        // resampler tolerates this small inaccuracy fine.
        static unsigned long next_sample_us = 0;
        static bool first_batch = true;
        const unsigned long lag_threshold_us = 100000;  // 100 ms

        if (first_batch || (next_sample_us + lag_threshold_us < now_us)){
          next_sample_us = now_us;
          first_batch = false;
        }

        for (uint16_t i = 0; i < n_samples; i++){
          imu.readFIFO(buf, FIFO_SAMPLE_BYTES);

          // ACCEL bytes from FIFO (chip frame, big-endian int16)
          int16_t ax_chip = (int16_t)((buf[0] << 8) | buf[1]);
          int16_t ay_chip = (int16_t)((buf[2] << 8) | buf[3]);
          int16_t az_chip = (int16_t)((buf[4] << 8) | buf[5]);

          // chip → PCB body remap: PCB_x = chip_z, PCB_y = chip_x, PCB_z = chip_y
          int16_t ax = az_chip;
          int16_t ay = ax_chip;
          int16_t az = ay_chip;

          common_isr_imu_reading.micros_reading = next_sample_us;
          next_sample_us += FIFO_SAMPLE_DT_US;
          common_isr_imu_reading.counter = imu_isr_count;
          imu_isr_count++;
          common_isr_imu_reading.acc_x = ax;
          common_isr_imu_reading.acc_y = ay;
          common_isr_imu_reading.acc_z = az;
          common_isr_imu_reading.gyr_x = gx;
          common_isr_imu_reading.gyr_y = gy;
          common_isr_imu_reading.gyr_z = gz;
          common_isr_imu_reading.mag_x = mx;
          common_isr_imu_reading.mag_y = my;
          common_isr_imu_reading.mag_z = mz;

          if (deque_IMU_readings.full()){
            deque_IMU_readings.pop_front();
          }
          deque_IMU_readings.push_back(common_isr_imu_reading);
          number_imu_samples_logged++;
        }

        if (ENABLE_DEBUG_FASTPRINT){
          SERIAL_USB->print(F("DI"));
          SERIAL_USB->print(n_samples);
          SERIAL_USB->print(F(";"));
        }
      }
    }

    // if time to read GNSS data, do it and store in deque
    // (skip entirely if no GNSS was detected at boot — calling getPVT on a
    // non-existent I2C device would just time out per ISR tick)
    if (ENABLE_GNSS && g_gnss_present && ctimer_isr_count % (TIMER_DIVIDER_GNSS) == 0){
      // check if we have a new GNSS reading; if yes, push fix to deque
      if (log_GNSS.getPVT()){
        common_isr_gnss_reading.micros_reading = micros();
        common_isr_gnss_reading.latitude = log_GNSS.getLatitude();
        common_isr_gnss_reading.longitude = log_GNSS.getLongitude();
        common_isr_gnss_reading.posix_timestamp = log_GNSS.getUnixEpoch(common_isr_gnss_reading.microseconds);
        // Drift marker: force posix_timestamp=0 on the next g_drift_marker_remaining
        // GNSS entries so that any plot of UTC vs sample index shows a clear 1970
        // spike at the moment the GNSS-sync re-corrected the RTC.
        if (g_drift_marker_remaining > 0){
          common_isr_gnss_reading.posix_timestamp = 0;
          g_drift_marker_remaining--;
        }
        common_isr_gnss_reading.NED_vel_north = log_GNSS.getNedNorthVel();
        common_isr_gnss_reading.NED_vel_east = log_GNSS.getNedEastVel();
        common_isr_gnss_reading.NED_vel_down = log_GNSS.getNedDownVel();
        common_isr_gnss_reading.altitude_msl_mm = log_GNSS.getAltitudeMSL();
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
  // RESET-button multi-press detection. Runs EARLY so the decision window
  // opens close to boot — the user presses RESET 2× for gyro cal, 3× for mag
  // cal. Each press reboots the chip; consecutive presses are detected across
  // reboots via a decision-pending flag in EEPROM. Blocks ~2.5 s (LED blinks).
  CalibrationManager::BootAction boot_action =
      calibration_manager.detect_boot_action(PIN_STAT_LED);

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
  // Report RESET multi-press result and load any stored calibration.
  switch (boot_action){
    case CalibrationManager::BootAction::FILE_TRANSFER:
      SERIAL_USB->println(F("RESET double-press detected -> FILE TRANSFER mode requested"));
      break;
    case CalibrationManager::BootAction::MAG_CAL:
      SERIAL_USB->println(F("RESET triple-press detected -> MAG CALIBRATION requested"));
      break;
    case CalibrationManager::BootAction::GYRO_CAL:
      SERIAL_USB->println(F("RESET quadruple-press detected -> GYRO CALIBRATION requested"));
      break;
    default:
      break;
  }
  calibration_manager.load();
  calibration_manager.print_state(*SERIAL_USB);
  // Publish the loaded gyro bias to the ISR-visible globals.
  g_gyro_bias_x = calibration_manager.gyro_bias_x();
  g_gyro_bias_y = calibration_manager.gyro_bias_y();
  g_gyro_bias_z = calibration_manager.gyro_bias_z();
  // Same for the mag hard-iron offset (raw chip LSB).
  g_mag_bias_x = calibration_manager.mag_bias_x();
  g_mag_bias_y = calibration_manager.mag_bias_y();
  g_mag_bias_z = calibration_manager.mag_bias_z();
  wdt.restart();

  /////////////////////////////////////////////////////////////////////////////////
  // Host-handshake fallback: if the user is connecting via serial_transfer.py,
  // the script has been writing "OLA_ENTER_TRANSFER\n" continuously since it
  // opened the port. Those bytes accumulated in our USB-CDC RX buffer during
  // boot + multi-press-window. Scan the buffer once for the magic token and,
  // if present, promote the boot action to FILE_TRANSFER without requiring a
  // physical RESET double-press. Only overrides NORMAL — explicit calibration
  // requests via multi-press are preserved.
  if (boot_action == CalibrationManager::BootAction::NORMAL){
    // We may have boot-banner / RTC-seed output already in the TX buffer; flush
    // it so the host knows we are alive and sees nothing weird interspersed.
    SERIAL_USB->flush();
    // Listen for up to 500 ms so a script that connects DURING our boot still
    // has a chance to be heard (CDC buffer may not be ready until ~1.5 s post-
    // reset on Apollo3; this catches the tail of the script's spam burst).
    char rxbuf[256];
    size_t n = 0;
    unsigned long const probe_start = millis();
    while (millis() - probe_start < 500){
      while (SERIAL_USB->available() && n < sizeof(rxbuf) - 1){
        rxbuf[n++] = (char)SERIAL_USB->read();
      }
      rxbuf[n] = '\0';
      if (strstr(rxbuf, "OLA_ENTER_TRANSFER") != nullptr){
        boot_action = CalibrationManager::BootAction::FILE_TRANSFER;
        SERIAL_USB->println();
        SERIAL_USB->println(F("=== HOST HANDSHAKE ACCEPTED -> FILE TRANSFER mode ==="));
        break;
      }
      delay(10);
      wdt.restart();
    }
  }
  wdt.restart();

  /////////////////////////////////////////////////////////////////////////////////
  // File-transfer mode short-circuit. Bring up only the SD card (we don't need
  // IMU or GNSS) and hand control to the file-transfer command loop on the USB
  // serial. enter_file_transfer_mode() never returns — it reboots on `exit`.
  if (boot_action == CalibrationManager::BootAction::FILE_TRANSFER){
    SERIAL_USB->println(F("Starting SD card for file-transfer mode..."));
    if (!sd_card_manager.start()){
      SERIAL_USB->println(F("ERROR sd_init_failed — cannot enter transfer mode. Rebooting in 5s."));
      delay(5000);
      NVIC_SystemReset();
    }
    enter_file_transfer_mode();   // NEVER RETURNS
  }

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
  // Seed the software posix_timestamp counter from a persistent source.
  // Priority:
  //   1. H/W RTC (battery-backed via VBAT coin cell) — most accurate, drift on
  //      the 32.768 kHz XT is ~20 ppm so even after weeks the seed is within
  //      seconds of true UTC.
  //   2. EEPROM-stored last-known UTC (only if ENABLE_TIME_EEPROM_FALLBACK) —
  //      as fresh as the last save (15 min for a normally-running device);
  //      worst case stale by however long the device was powered off PLUS
  //      one file rotation.
  //   3. Fall through to 0 — same as before (1970 default), still gets
  //      overridden by GNSS once a fix arrives.
  // g_posix_seeded_from_hw_rtc is set only on path (1) so the GNSS-sync drift
  // check knows whether the seed was authoritative enough to compare against.
  {
    kiss_time_t const hw_rtc_posix = board_time_manager.read_hw_rtc_posix();
    if (hw_rtc_posix != 0){
      board_time_manager.set_posix_timestamp(hw_rtc_posix);
      g_posix_seeded_from_hw_rtc = true;
      SERIAL_USB->print(F("RTC seeded from H/W RTC (battery-backed): "));
      SERIAL_USB->println(hw_rtc_posix);
    } else if (ENABLE_TIME_EEPROM_FALLBACK && calibration_manager.has_last_known_posix()){
      uint32_t const eeprom_posix = calibration_manager.load_last_known_posix();
      board_time_manager.set_posix_timestamp(eeprom_posix);
      SERIAL_USB->print(F("RTC seeded from EEPROM fallback (last known UTC): "));
      SERIAL_USB->println(eeprom_posix);
    } else {
      board_time_manager.set_posix_timestamp(0);
      SERIAL_USB->println(F("RTC seed: no persistent source available, starting at 1970"));
    }
  }
  board_time_manager.print_status();
  SERIAL_USB->println();

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
    //
    // Probe behaviour: try `log_GNSS.begin()` once per setup attempt. If the
    // GNSS module isn't present on QWIIC (no response, e.g. user didn't plug
    // one in), set g_gnss_present=false and proceed WITHOUT GNSS — relying on
    // the H/W RTC / EEPROM time seed for timestamps. This is the path the user
    // wants when running the IMU without GNSS for indoor / quick-test work.

    g_gnss_present = false;
    if (ENABLE_GNSS){
      if (!log_GNSS.begin(*I2C_QWIIC)){
          SERIAL_USB->println(F("No GNSS module detected on QWIIC — proceeding "
                                "without GNSS; timestamps will use the persistent "
                                "RTC seed (H/W RTC or EEPROM)."));
          // Don't `continue` (= retry setup) — assume the user intentionally
          // didn't attach a GNSS, and carry on.
      } else {
        g_gnss_present = true;
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

        // wait until we get a fix (only if both compile-time flag and runtime
        // probe agree there's a GNSS to wait on)
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

          // We have a GNSS fix. Compare against the previously-seeded SW
          // counter (if it came from the battery-backed H/W RTC) — any large
          // disagreement means the H/W RTC drifted or the cell was replaced
          // while powered off. Embed a 1970-spike marker in the next two
          // GNSS-deque writes so the data file makes the re-sync visible.
          uint32_t const gnss_utc = log_GNSS.getUnixEpoch();
          if (g_posix_seeded_from_hw_rtc){
            uint32_t const rtc_utc = board_time_manager.get_posix_timestamp();
            int64_t const delta = (int64_t)gnss_utc - (int64_t)rtc_utc;
            if (delta > 1 || delta < -1){
              g_drift_marker_remaining = 2;
              SERIAL_USB->print(F("GNSS vs H/W-RTC drift = "));
              SERIAL_USB->print((long)delta);
              SERIAL_USB->println(F(" s — arming 1970-spike marker for next 2 GNSS entries"));
            } else {
              SERIAL_USB->print(F("GNSS vs H/W-RTC drift = "));
              SERIAL_USB->print((long)delta);
              SERIAL_USB->println(F(" s — within tolerance, no marker"));
            }
          }
          // Adopt the GNSS UTC as the authoritative time for both the SW
          // counter AND the battery-backed H/W RTC.
          board_time_manager.set_posix_timestamp(gnss_utc);
          board_time_manager.write_hw_rtc_posix(gnss_utc);
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
      }
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

    // Bring up the AK09916 magnetometer. begin() already did this once, but
    // the swReset() above wiped the AK09916 startup with the rest of the
    // chip state, so we have to redo it here. Without this, imu.agmt.mag.*
    // reads zero forever.
    {
      ICM_20948_Status_e mag_status = imu.startupMagnetometer();
      SERIAL_USB->print(F("ICM-20948 startupMagnetometer: "));
      SERIAL_USB->println(imu.statusString(mag_status));
      if (mag_status != ICM_20948_Stat_Ok){
        SERIAL_USB->println(F("WARNING: magnetometer init failed; mag values will be zero"));
      }
    }
    wdt.restart();

    // ---------- Enable the on-chip FIFO for accel + gyro ----------
    // Why: when the main MCU is busy writing to the SD card, the CTIMER ISR
    // is masked (see SDIrqGuard) and we used to lose every IMU sample for
    // the entire SD-flush window (≥700 ms in practice). With the FIFO on,
    // the ICM-20948 keeps buffering samples in its 4 KB FIFO at the
    // configured ODR while the MCU is "deaf". When the CTIMER ISR resumes
    // it drains everything in one go, so the recorded time series has no
    // gaps even across long SD pauses. Per-sample size below is 12 bytes
    // (6 accel + 6 gyro, no temp, no mag), so 4 KB holds ≈ 341 samples,
    // i.e. ≈ 1.5 s of buffer headroom at 225 Hz.
    imu.enableFIFO(false);                              // disable while reconfiguring
    // ACCEL ONLY in FIFO. With BOTH accel (1125/5=225Hz) and gyro (1100/5=220Hz)
    // in FIFO, the chip writes each sensor's 6 bytes INDEPENDENTLY at its own
    // rate — they don't form synchronised 12-byte frames. Over time the byte
    // positions of accel vs gyro drift within each 12-byte read, eventually
    // putting accel bytes into the gyro slot (verified in BOOT_000292 ~30s).
    // Solution: only accel in FIFO; gyro+mag read via direct registers once
    // per ISR drain and applied to the whole batch. Per-sample size = 6 B.
    imu.setFIFOdataAccelGyroTemp(true, false, false);   // accel only
    imu.setFIFOmode(false);                             // stream mode
    imu.resetFIFO();
    imu.enableFIFO(true);
    SERIAL_USB->print(F("ICM-20948 FIFO enabled (accel only), 6 B/sample: "));
    SERIAL_USB->println(imu.statusString());

    // FIFO register sanity check (one-shot at boot)
    {
      uint8_t v;
      imu.debugReadReg(0, 0x67, &v); SERIAL_USB->print(F("[boot] FIFO_EN_2=0x")); SERIAL_USB->print(v, HEX);
      imu.debugReadReg(2, 0x14, &v); SERIAL_USB->print(F("  ACCEL_CONFIG=0x")); SERIAL_USB->print(v, HEX);
      imu.debugReadReg(2, 0x01, &v); SERIAL_USB->print(F("  GYRO_CFG_1=0x")); SERIAL_USB->println(v, HEX);
    }

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
    // Tier-1 gyro bias calibration (triggered by RESET quadruple-press).
    // Runs here: IMU + FIFO are configured but the CTIMER ISR is not yet
    // started, so we can poll the gyro directly without contention. We read
    // via getAGMT() (direct registers) and store the bias in CHIP frame —
    // same frame the ISR now publishes — so it subtracts cleanly. Gyro bias
    // is a DC offset, so it's identical whether sampled via FIFO or direct
    // registers.
    if (boot_action == CalibrationManager::BootAction::GYRO_CAL){
      SERIAL_USB->println(F("=== GYRO CALIBRATION ==="));
      // Mode-entry LED signature: 5 short flashes.
      blink_stat_led(5, 80, 120);
      SERIAL_USB->println(F("Keep the device PERFECTLY STILL for 5 seconds..."));
      delay(500);  // brief pause so the user stops touching the board after reset
      wdt.restart();

      long sum_gx = 0, sum_gy = 0, sum_gz = 0;
      uint32_t n_cal = 0;
      unsigned long cal_start = millis();
      while (millis() - cal_start < 5000){
        if (imu.dataReady()){
          imu.getAGMT();
          int16_t gx_chip = imu.agmt.gyr.axes.x;
          int16_t gy_chip = imu.agmt.gyr.axes.y;
          int16_t gz_chip = imu.agmt.gyr.axes.z;
          // PCB-frame accumulation — gyro is pass-through (= raw chip frame):
          //   PCB_gx = +chip_gx, PCB_gy = +chip_gy, PCB_gz = +chip_gz
          sum_gx += (long)gx_chip;
          sum_gy += (long)gy_chip;
          sum_gz += (long)gz_chip;
          n_cal++;
        }
        // fast blink to indicate "hold still, calibrating"
        digitalWrite(PIN_STAT_LED, ((millis() / 100) % 2) ? HIGH : LOW);
        wdt.restart();
      }
      digitalWrite(PIN_STAT_LED, LOW);

      if (n_cal > 50){
        int16_t bx = (int16_t)(sum_gx / (long)n_cal);
        int16_t by = (int16_t)(sum_gy / (long)n_cal);
        int16_t bz = (int16_t)(sum_gz / (long)n_cal);
        calibration_manager.save_gyro_bias(bx, by, bz);
        g_gyro_bias_x = bx;
        g_gyro_bias_y = by;
        g_gyro_bias_z = bz;
        SERIAL_USB->print(F("Gyro bias stored (body LSB): x="));
        SERIAL_USB->print(bx); SERIAL_USB->print(F(" y="));
        SERIAL_USB->print(by); SERIAL_USB->print(F(" z="));
        SERIAL_USB->print(bz);
        SERIAL_USB->print(F("  (= "));
        SERIAL_USB->print(bx * gyr_sensitivity / 1000.0f, 2); SERIAL_USB->print(F(", "));
        SERIAL_USB->print(by * gyr_sensitivity / 1000.0f, 2); SERIAL_USB->print(F(", "));
        SERIAL_USB->print(bz * gyr_sensitivity / 1000.0f, 2);
        SERIAL_USB->print(F(" deg/s over "));
        SERIAL_USB->print(n_cal); SERIAL_USB->println(F(" samples)"));
        // 3 slow confirm blinks
        blink_stat_led(3, 250, 250);
      } else {
        SERIAL_USB->print(F("Gyro calibration FAILED: only "));
        SERIAL_USB->print(n_cal);
        SERIAL_USB->println(F(" samples collected; keeping previous calibration"));
      }
      // Clear samples accumulated in the FIFO during calibration so logging
      // starts clean.
      imu.resetFIFO();
      wdt.restart();
    }

    ////////////////////////////////////////////////////
    // Tier-2 magnetometer hard-iron calibration (triggered by RESET triple-press).
    // The user rotates the device through as many orientations as possible for
    // 30 seconds. Each (mx, my, mz) sample is a point on a sphere centred at
    // the hard-iron offset (h_x, h_y, h_z) with radius equal to the true |B|.
    // We online-accumulate normal-equation sums for the sphere fit and solve a
    // 4x4 linear system at the end:
    //
    //   2 h_x mx_i + 2 h_y my_i + 2 h_z mz_i + K = mx_i² + my_i² + mz_i²
    //   where K = R² − (h_x² + h_y² + h_z²)
    //
    // No need to buffer samples: 11 scalar accumulators capture everything.
    // The bias is stored in chip-LSB units (raw mag values), matching the ISR
    // which subtracts it before logging.
    if (boot_action == CalibrationManager::BootAction::MAG_CAL){
      SERIAL_USB->println(F("=== MAGNETOMETER CALIBRATION ==="));
      // Mode-entry LED signature: 3 short flashes + 2 long flashes.
      blink_stat_led(3, 80, 150);
      delay(250);
      blink_stat_led(2, 400, 250);
      SERIAL_USB->println(F("Rotate the device through as many 3D orientations"));
      SERIAL_USB->println(F("as you can (figure-8 motion in all axes) for 30 s."));
      SERIAL_USB->println(F("Stay >1 m clear of metal furniture / electronics."));
      SERIAL_USB->println(F("STAT LED blinks slowly during sampling, fast on success."));
      delay(2000);  // give user time to start moving the device
      wdt.restart();

      // Sphere-fit accumulators (doubles for headroom — Apollo3 has an FPU).
      double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
      double sum_xx = 0.0, sum_yy = 0.0, sum_zz = 0.0;
      double sum_xy = 0.0, sum_xz = 0.0, sum_yz = 0.0;
      double sum_x_r2 = 0.0, sum_y_r2 = 0.0, sum_z_r2 = 0.0;
      double sum_r2 = 0.0;
      uint32_t n_cal = 0;

      unsigned long const cal_start = millis();
      unsigned long last_progress_ms = 0;
      static constexpr unsigned long MAG_CAL_DURATION_MS = 30000UL;
      while (millis() - cal_start < MAG_CAL_DURATION_MS){
        if (imu.dataReady()){
          imu.getAGMT();
          double const mx = (double)imu.agmt.mag.axes.x;
          double const my = (double)imu.agmt.mag.axes.y;
          double const mz = (double)imu.agmt.mag.axes.z;
          double const r2 = mx*mx + my*my + mz*mz;
          sum_x  += mx;     sum_y  += my;     sum_z  += mz;
          sum_xx += mx*mx;  sum_yy += my*my;  sum_zz += mz*mz;
          sum_xy += mx*my;  sum_xz += mx*mz;  sum_yz += my*mz;
          sum_x_r2 += mx*r2; sum_y_r2 += my*r2; sum_z_r2 += mz*r2;
          sum_r2 += r2;
          n_cal++;
        }
        // slow blink (500 ms period) so the user can see the firmware is alive
        // and visually distinguish from the gyro cal's faster blink
        digitalWrite(PIN_STAT_LED, ((millis() / 500) % 2) ? HIGH : LOW);
        // progress print every 5 s
        unsigned long const elapsed = millis() - cal_start;
        if (elapsed / 5000UL > last_progress_ms / 5000UL){
          SERIAL_USB->print(F("  "));
          SERIAL_USB->print(elapsed / 1000UL);
          SERIAL_USB->print(F("s elapsed; "));
          SERIAL_USB->print(n_cal);
          SERIAL_USB->println(F(" samples"));
          last_progress_ms = elapsed;
        }
        wdt.restart();
      }
      digitalWrite(PIN_STAT_LED, LOW);

      bool cal_ok = false;
      if (n_cal > 500){
        // Build A^T A (4x4 symmetric) and A^T b (4x1).
        //   A row i = [2 mx_i, 2 my_i, 2 mz_i, 1]
        //   b   i   = mx_i² + my_i² + mz_i²
        double M[4][5] = {
          {4.0*sum_xx, 4.0*sum_xy, 4.0*sum_xz, 2.0*sum_x, 2.0*sum_x_r2},
          {4.0*sum_xy, 4.0*sum_yy, 4.0*sum_yz, 2.0*sum_y, 2.0*sum_y_r2},
          {4.0*sum_xz, 4.0*sum_yz, 4.0*sum_zz, 2.0*sum_z, 2.0*sum_z_r2},
          {2.0*sum_x,  2.0*sum_y,  2.0*sum_z,  (double)n_cal, sum_r2  }
        };

        // Solve via Gauss-Jordan with partial pivoting (4x4, trivial).
        bool singular = false;
        for (int p = 0; p < 4 && !singular; p++){
          // Find pivot
          int pivot = p;
          double pivot_mag = fabs(M[p][p]);
          for (int i = p+1; i < 4; i++){
            double const v = fabs(M[i][p]);
            if (v > pivot_mag){ pivot_mag = v; pivot = i; }
          }
          if (pivot_mag < 1e-9){
            singular = true; break;
          }
          if (pivot != p){
            for (int j = 0; j < 5; j++){
              double const tmp = M[p][j]; M[p][j] = M[pivot][j]; M[pivot][j] = tmp;
            }
          }
          for (int i = p+1; i < 4; i++){
            double const factor = M[i][p] / M[p][p];
            for (int j = p; j < 5; j++){ M[i][j] -= factor * M[p][j]; }
          }
        }

        if (!singular){
          // Back-substitute
          double x[4];
          for (int i = 3; i >= 0; i--){
            double s = M[i][4];
            for (int j = i+1; j < 4; j++){ s -= M[i][j] * x[j]; }
            x[i] = s / M[i][i];
          }
          double const hx = x[0];
          double const hy = x[1];
          double const hz = x[2];
          double const K  = x[3];
          double const R2 = K + hx*hx + hy*hy + hz*hz;
          double const R  = (R2 > 0.0) ? sqrt(R2) : 0.0;

          SERIAL_USB->print(F("Mag cal: hard-iron offset (chip LSB) = ("));
          SERIAL_USB->print(hx, 1); SERIAL_USB->print(F(", "));
          SERIAL_USB->print(hy, 1); SERIAL_USB->print(F(", "));
          SERIAL_USB->print(hz, 1); SERIAL_USB->println(F(")"));
          SERIAL_USB->print(F("Mag cal: fitted field radius (chip LSB) = "));
          SERIAL_USB->println(R, 1);
          // AK09916 sensitivity = 0.15 µT / LSB
          SERIAL_USB->print(F("  (in uT: offset = ("));
          SERIAL_USB->print(hx * 0.15f, 2); SERIAL_USB->print(F(", "));
          SERIAL_USB->print(hy * 0.15f, 2); SERIAL_USB->print(F(", "));
          SERIAL_USB->print(hz * 0.15f, 2); SERIAL_USB->print(F("); |B| ~= "));
          SERIAL_USB->print(R * 0.15f, 1); SERIAL_USB->println(F(" uT)"));
          SERIAL_USB->print(F("  collected over "));
          SERIAL_USB->print(n_cal); SERIAL_USB->println(F(" samples"));

          // Saturate to int16 and store
          auto clamp_i16 = [](double v) -> int16_t {
            if (v >  32767.0) return  32767;
            if (v < -32768.0) return -32768;
            return (int16_t)lround(v);
          };
          int16_t const bx = clamp_i16(hx);
          int16_t const by = clamp_i16(hy);
          int16_t const bz = clamp_i16(hz);
          calibration_manager.save_mag_hardiron(bx, by, bz);
          g_mag_bias_x = bx;
          g_mag_bias_y = by;
          g_mag_bias_z = bz;
          SERIAL_USB->println(F("Mag hard-iron stored in EEPROM."));
          // Fast confirm-blink pattern
          blink_stat_led(5, 80, 80);
          cal_ok = true;
        } else {
          SERIAL_USB->println(F("Mag cal FAILED: normal-equation matrix is "
                                "singular — not enough rotation diversity. "
                                "Make sure to rotate through orientations on "
                                "all three axes (figure-8 motion)."));
        }
      } else {
        SERIAL_USB->print(F("Mag cal FAILED: only "));
        SERIAL_USB->print(n_cal);
        SERIAL_USB->println(F(" samples (need >500); keeping previous calibration"));
      }
      if (!cal_ok){
        // Three slow blinks to indicate failure (vs. the fast confirm pattern)
        blink_stat_led(3, 600, 200);
      }
      imu.resetFIFO();
      wdt.restart();
    }

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

    // Persist the current UTC to EEPROM as a fallback for boards without a
    // working coin cell. Only writes if (a) the timestamp is plausibly real
    // (year >= 2025) so we never overwrite a good cached value with the 1970
    // default, AND (b) the previous EEPROM write happened more than
    // EEPROM_TIME_WRITE_INTERVAL_S ago. Each save_last_known_posix() call is
    // two flash erase+write cycles (~100-200 ms total during which the
    // Apollo3 EEPROM lib internally masks all interrupts — including the
    // IMU ISR), so calling it at every 15-min file rotation adds ~100-200 ms
    // to the per-file gyro/mag freeze window. A 1-hour-old RTC seed is still
    // perfectly useful as a next-boot fallback, so we batch the writes.
    static constexpr uint32_t EEPROM_TIME_WRITE_INTERVAL_S = 3600;  // 1 h
    static uint32_t last_eeprom_time_save = 0;
    if (ENABLE_TIME_EEPROM_FALLBACK
        && posix_timestamp >= 1735689600UL /* 2025-01-01 */
        && (last_eeprom_time_save == 0
            || posix_timestamp - last_eeprom_time_save >= EEPROM_TIME_WRITE_INTERVAL_S))
    {
      calibration_manager.save_last_known_posix(posix_timestamp);
      last_eeprom_time_save = posix_timestamp;
    }
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

    // sensitivity IMU MAG (AK09916 fixed sensitivity per ICM-20948 datasheet: 0.15 uT/LSB)
    for (size_t i=0; i<sizeof(working_buffer); i++){
      working_buffer[i] = '\0';
    }
    snprintf(working_buffer, sizeof(working_buffer),
             "ICM-20948 Mag sensitivity (uT/LSB): %.6f (AK09916 fixed)\n", 0.15f);
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

    // GNSS struct includes altitude_msl (int32 mm, MAX-M10S getAltitudeMSL)
    for (size_t i=0; i<sizeof(working_buffer); i++){
      working_buffer[i] = '\0';
    }
    snprintf(working_buffer, sizeof(working_buffer),
             "GNSS includes altitude_msl (mm)\n");
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

    // Heartbeat LED. While in normal logging mode, give the STAT LED a short
    // 50 ms ON pulse every 3 seconds so the user can distinguish "logging" at
    // a glance from the other modes (file-transfer pulse, mag-cal slow blink,
    // gyro-cal fast blink). We only write the LED on the *edges* of the
    // pulse — during the off-phase between pulses the SD-write toggle in
    // sd_card_manager.cpp is free to drive the LED itself (so SD-busy bursts
    // still flash visibly between heartbeats).
    static constexpr unsigned long HEARTBEAT_PERIOD_MS = 3000;
    static constexpr unsigned long HEARTBEAT_ON_MS     = 50;
    static unsigned long heartbeat_start_ms = millis();
    static bool heartbeat_was_high = false;

    while (board_time_manager.get_posix_timestamp() < posix_timestamp_next_file){
      should_log_data = false;

      {
        unsigned long const now_ms = millis();
        unsigned long const phase = (now_ms - heartbeat_start_ms) % HEARTBEAT_PERIOD_MS;
        bool const in_pulse = (phase < HEARTBEAT_ON_MS);
        if (in_pulse && !heartbeat_was_high) {
          digitalWrite(PIN_STAT_LED, HIGH);
          heartbeat_was_high = true;
        } else if (!in_pulse && heartbeat_was_high) {
          digitalWrite(PIN_STAT_LED, LOW);
          heartbeat_was_high = false;
        }
      }

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

        // Frequency-checker auto-reset removed: a momentary dip in IMU/GNSS/PPS
        // rate (e.g. from SD-write blocking) used to trigger NVIC_SystemReset,
        // which silently rebooted the MCU mid-recording and corrupted the
        // micros timeline. Any sample-rate anomaly is now visible in the
        // recorded data itself, no firmware action needed.

        accumulated_sd_time_millis = 0;

      }

      if (ENABLE_GNSS && g_gnss_present){
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
