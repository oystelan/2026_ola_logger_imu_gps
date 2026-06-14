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

// ICM-20948 internal sample-rate is 1125 / (1 + SMPLRT_DIV). With div=10 the
// accel ODR is 1125/11 = 102.27 Hz; we call it "100 Hz" everywhere for
// brevity. Gyro ODR with the same div is 1100/11 = 100 Hz exactly. Picked at
// this rate because the chip's 4 KB FIFO has to absorb the full 8 MB
// preAllocate stall (~500-1000 ms on the user's SD card) at every file
// rotation. At 100 Hz with DMP packets carrying accel+gyro+bias only (22 B
// each), the FIFO holds ~1.86 s of data — comfortable margin for the stall.
static constexpr float IMU_ODR_HZ = 100.0f;
static constexpr uint16_t IMU_SMPLRT_DIV = 10;

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

    // Drain the ICM-20948 DMP FIFO. The DMP firmware writes HEADERED packets
    // (header bitmap + per-sensor data + footer) so byte alignment is
    // unambiguous even with the slightly different accel/gyro sample clocks.
    // At 100 Hz with our config (Accel + Gyro, no Compass) each packet is
    // exactly Header(2) + Accel(6) + Gyro+bias(12) + Footer(2) = 22 bytes.
    //
    // The 4 KB on-chip FIFO buffers 4096/22 ≈ 186 packets = ~1.86 s of data,
    // enough to ride out the worst-case SD preAllocate stall (~500-1000 ms)
    // at every file rotation without losing a sample. The DMP firmware
    // keeps writing into the FIFO even while CTIMER_IRQn is masked by
    // SDIrqGuard, so SD-stall windows produce zero gyro/mag freeze.
    //
    // BUG FIXES carried over from the first DMP attempt:
    //   * resetDMP() is NEVER called at runtime — it nukes DMP firmware
    //     state and we can't recover without re-init (BOOT_70 dead-stop
    //     at 2.5s).
    //   * Pre-check getFIFOcount() >= DMP_MAX_PACKET_BYTES before each
    //     readDMPdataFromFIFO() — the lib's first read consumes the 2-byte
    //     header before checking the body fits; pre-checking prevents that
    //     header from being lost (BOOT_61 multi-second desyncs).
    //   * Discard POST_RESET_DISCARD_PACKETS = 32 packets after every
    //     reset (boot included) — the DMP firmware emits warmup garbage
    //     during its internal resync (BOOT_78 zero-sample spike clusters).
    //   * Drop packets where Raw_Accel = (0,0,0) — physically impossible
    //     and the known DMP "glitch" signature (BOOT_75: 11 packets in
    //     11106 had this pattern).
    //   * On any non-Ok status, resetFIFO() (NOT resetDMP) and discard 32
    //     packets to clear potential wraparound corruption.
    {
      static constexpr unsigned long FIFO_SAMPLE_DT_US =
          (unsigned long)(1000000.0f / IMU_ODR_HZ + 0.5f);
      // Max packet size when all our enabled sensors fire in the same
      // DMP cycle. We pre-check FIFO has at least this many bytes before
      // calling readDMPdataFromFIFO so the lib's header-then-body read
      // path can't desync the FIFO on a partial packet.
      static constexpr uint16_t DMP_MAX_PACKET_BYTES = 22;
      static constexpr uint16_t POST_RESET_DISCARD_PACKETS = 32;
      static constexpr uint16_t MAX_PACKETS_PER_ISR = 256;

      // Per-ISR static state
      static int16_t last_gx_chip {0};
      static int16_t last_gy_chip {0};
      static int16_t last_gz_chip {0};
      static unsigned long last_emitted_us = 0;
      static bool first_emit = true;
      static uint16_t skip_packets_after_reset = POST_RESET_DISCARD_PACKETS;  // discard boot warmup

      // Local per-ISR buffer for the parsed samples drained THIS ISR. Static
      // (not on the ISR stack) to keep stack use low; not re-entrant, which is
      // fine — the CTIMER ISR cannot preempt itself. 6 int16 per sample.
      static int16_t b_ax[MAX_PACKETS_PER_ISR], b_ay[MAX_PACKETS_PER_ISR], b_az[MAX_PACKETS_PER_ISR];
      static int16_t b_gx[MAX_PACKETS_PER_ISR], b_gy[MAX_PACKETS_PER_ISR], b_gz[MAX_PACKETS_PER_ISR];

      // --- Timestamping: anchor each batch to the real hardware micros() ---
      // The DMP emits accel at 1125/(1+SMPLRT_DIV) = 102.27 Hz; FIFO_SAMPLE_DT_US
      // (1e6/IMU_ODR_HZ = 100 Hz) is only used for INTRA-batch spacing of the
      // rare multi-sample post-stall drains. The CTIMER ISR fires at 2x ODR
      // (~200 Hz) so a normal ISR drains 0 or 1 samples; the single sample is
      // stamped with now_us = micros() directly, locking the timeline to the
      // GPS-disciplined hardware clock with ZERO rate bias. (The earlier free-
      // running 100 Hz counter drifted +2.27%; a PLL replacement still left
      // +0.7% because its monotonicity clamp fought the pull-back. Direct
      // per-batch micros() anchoring has neither problem.)
      unsigned long now_us = micros();

      // ---- Phase 1: drain all available packets into the local buffer ----
      uint16_t n_batch = 0;
      while (n_batch < MAX_PACKETS_PER_ISR){
        // Pre-check: only call readDMPdataFromFIFO when a full packet is
        // guaranteed to be in FIFO. The lib's first SPI read consumes the
        // 2-byte header even if the body bytes aren't ready — pre-checking
        // makes that impossible.
        uint16_t fifo_avail = 0;
        imu.getFIFOcount(&fifo_avail);
        if (fifo_avail < DMP_MAX_PACKET_BYTES) break;

        icm_20948_DMP_data_t pkt;
        imu.readDMPdataFromFIFO(&pkt);
        const ICM_20948_Status_e s = imu.status;

        // Any unexpected status — overflow / corrupt header / etc. — means
        // we'd best dump the FIFO and re-sync. resetFIFO() is safe at
        // runtime; resetDMP() is not (it kills the firmware state).
        if (s != ICM_20948_Stat_Ok && s != ICM_20948_Stat_FIFOMoreDataAvail){
          imu.resetFIFO();
          skip_packets_after_reset = POST_RESET_DISCARD_PACKETS;
          break;
        }

        // Discard the first packets after a reset — they tend to be DMP
        // warmup garbage (acc zero, gyro huge spikes) while the chip
        // re-populates internal state.
        if (skip_packets_after_reset > 0){
          skip_packets_after_reset--;
          if (s != ICM_20948_Stat_FIFOMoreDataAvail) break;
          continue;
        }

        // We only emit if the packet carries accel data — at our config
        // every packet should — but guard.
        if (!(pkt.header & DMP_header_bitmap_Accel)){
          if (s != ICM_20948_Stat_FIFOMoreDataAvail) break;
          continue;
        }

        // Drop the "DMP-glitch" packet: raw accel exactly (0,0,0). Gravity
        // is always present (~16384 LSB at gpm2 for 1g) and even chip
        // noise is non-zero, so an exact-zero triple is the known DMP
        // garbage signature, observed at ~0.1% of packets.
        if (pkt.Raw_Accel.Data.X == 0
            && pkt.Raw_Accel.Data.Y == 0
            && pkt.Raw_Accel.Data.Z == 0){
          if (s != ICM_20948_Stat_FIFOMoreDataAvail) break;
          continue;
        }

        // ---- GYRO: update cached chip-frame values if present, else reuse ----
        if (pkt.header & DMP_header_bitmap_Gyro){
          last_gx_chip = pkt.Raw_Gyro.Data.X;
          last_gy_chip = pkt.Raw_Gyro.Data.Y;
          last_gz_chip = pkt.Raw_Gyro.Data.Z;
        }
        b_gx[n_batch] = (int16_t)(last_gx_chip - g_gyro_bias_x);
        b_gy[n_batch] = (int16_t)(last_gy_chip - g_gyro_bias_y);
        b_gz[n_batch] = (int16_t)(last_gz_chip - g_gyro_bias_z);

        // ---- ACCEL chip→PCB body remap: same as previous DMP path ----
        // Empirical from BOOT_63: the DMP outputs accel such that
        // pkt.Raw_Accel.Data.X carries the gravity-axis reading. Identity
        // remap puts gravity on PCB_x.
        b_ax[n_batch] = pkt.Raw_Accel.Data.X;
        b_ay[n_batch] = pkt.Raw_Accel.Data.Y;
        b_az[n_batch] = pkt.Raw_Accel.Data.Z;
        n_batch++;

        if (s != ICM_20948_Stat_FIFOMoreDataAvail) break;
      }

      // ---- Phase 2: stamp + push. Newest sample of the batch gets now_us;
      // earlier samples spread backward by FIFO_SAMPLE_DT_US. Monotonicity
      // clamp keeps timestamps strictly increasing across ISRs. ----
      for (uint16_t k = 0; k < n_batch; k++){
        unsigned long ts = now_us - (unsigned long)(n_batch - 1 - k) * FIFO_SAMPLE_DT_US;
        if (first_emit){
          first_emit = false;
        } else if ((int32_t)(ts - last_emitted_us) <= 0){
          ts = last_emitted_us + 1;
        }
        last_emitted_us = ts;

        common_isr_imu_reading.micros_reading = ts;
        common_isr_imu_reading.counter = imu_isr_count;
        imu_isr_count++;
        common_isr_imu_reading.acc_x = b_ax[k];
        common_isr_imu_reading.acc_y = b_ay[k];
        common_isr_imu_reading.acc_z = b_az[k];
        common_isr_imu_reading.gyr_x = b_gx[k];
        common_isr_imu_reading.gyr_y = b_gy[k];
        common_isr_imu_reading.gyr_z = b_gz[k];
        // Mag is intentionally not in the DMP-FIFO on this branch; zero
        // the on-disk struct fields so existing decoders still read the
        // 24-byte IMU record but show "no mag data" to the user.
        common_isr_imu_reading.mag_x = 0;
        common_isr_imu_reading.mag_y = 0;
        common_isr_imu_reading.mag_z = 0;

        if (deque_IMU_readings.full()){
          deque_IMU_readings.pop_front();
        }
        deque_IMU_readings.push_back(common_isr_imu_reading);
        number_imu_samples_logged++;
      }

      if (ENABLE_DEBUG_FASTPRINT){
        SERIAL_USB->print(F("DI"));
        SERIAL_USB->print(n_batch);
        SERIAL_USB->print(F(";"));
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
    // Diagnostic: always print the RAW H/W RTC reading so a reset-vs-power-cycle
    // test can tell whether VBAT (coin cell) is actually preserving the RTC.
    // If this is 0/garbage after a full power-off but valid after a plain
    // reset, the coin cell is not holding up the VBAT rail.
    SERIAL_USB->print(F("[diag] raw H/W RTC posix at boot: "));
    SERIAL_USB->println(hw_rtc_posix);

    if (hw_rtc_posix != 0){
      board_time_manager.set_posix_timestamp(hw_rtc_posix);
      g_posix_seeded_from_hw_rtc = true;
      SERIAL_USB->print(F("RTC seeded from H/W RTC (battery-backed): "));
      SERIAL_USB->println(hw_rtc_posix);
    } else if (ENABLE_TIME_EEPROM_FALLBACK && calibration_manager.has_last_known_posix()){
      uint32_t const eeprom_posix = calibration_manager.load_last_known_posix();
      board_time_manager.set_posix_timestamp(eeprom_posix);
      // Bootstrap the H/W RTC from the EEPROM fallback. Previously the H/W RTC
      // was only ever written on a GNSS fix, so a GPS-less boot left the RTC
      // un-seeded and every subsequent power-cycle fell through to EEPROM
      // again. Writing it here means that — IF the coin cell is good — the
      // next reset/power-cycle will read a valid RTC (path 1 above) and the
      // time will advance correctly across power-offs even without GPS.
      board_time_manager.write_hw_rtc_posix(eeprom_posix);
      SERIAL_USB->print(F("RTC seeded from EEPROM fallback (last known UTC), "
                          "and written into H/W RTC: "));
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

    // ---------- FIFO off during calibration phase ----------
    // Calibration paths below read via getAGMT() (direct registers, no
    // FIFO). DMP-mode FIFO setup happens further down, AFTER calibration —
    // initializeDMP() reconfigures I2C_SLV0 for its own mag-read path, which
    // would break getAGMT()'s mag values if DMP were brought up earlier.
    imu.enableFIFO(false);
    imu.resetFIFO();

    // Register sanity check
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
    // ---------- DMP-mode FIFO for accel + gyro at 100 Hz ----------
    //
    // We put accel and gyro into the chip's hardware FIFO via the DMP
    // firmware. The DMP writes HEADERED packets, so byte alignment is
    // unambiguous even with the slightly different sample clocks (~225 Hz
    // accel vs ~220 Hz gyro at our div). At 100 Hz with 22 B/packet (Header
    // 2 + Accel 6 + Gyro+bias 12 + Footer 2) the data rate is 2.2 KB/s and
    // the 4 KB FIFO buffers ~1.86 s — enough to ride out the worst-case SD
    // preAllocate stall (~500-1000 ms) at every file rotation without losing
    // a sample.
    //
    // Mag is intentionally NOT in the FIFO: the user disabled it to free up
    // packet bytes (and AK09916 is capped at 100 Hz internally anyway, so
    // adding it would have given us at-most-100 Hz mag values for an extra
    // 6 B/packet that we don't have the headroom for). The IMU_reading
    // struct's mag_* fields are zeroed in the ISR — the on-disk binary
    // format is unchanged and the decoder still loads without modification.
    //
    // Init order: AFTER the gyro/mag calibration paths above. initializeDMP()
    // reconfigures I2C_SLV0 for its own (InvenSense "secret-sauce") mag-read
    // path, which would have broken getAGMT()'s mag readings if DMP came up
    // before calibration.
    //
    // Bug fixes carried over from the first DMP attempt (BOOT_70-85):
    //   * resetDMP() is NEVER called at runtime — it nukes DMP firmware
    //     state and we have no clean way to re-init from inside the ISR.
    //     resetFIFO() alone is enough to clear stream-mode wraparound.
    //   * Pre-check getFIFOcount() before each readDMPdataFromFIFO() so the
    //     lib's header-consuming first read can't desync the FIFO on a
    //     partial-packet read.
    //   * Discard the first 32 packets after every reset (boot included) —
    //     the DMP firmware emits warmup garbage during its internal resync.
    //   * Drop packets where Raw_Accel = (0,0,0) — physically impossible
    //     and a known DMP glitch signature.
    SERIAL_USB->print(F("ICM-20948 initializeDMP: "));
    ICM_20948_Status_e dmp_init_status = imu.initializeDMP();
    SERIAL_USB->println(imu.statusString(dmp_init_status));

    // Override stock DMP defaults (55 Hz, gpm4/dps2000) with our values.
    {
      ICM_20948_smplrt_t dmp_smplrt;
      dmp_smplrt.a = IMU_SMPLRT_DIV;
      dmp_smplrt.g = IMU_SMPLRT_DIV;
      imu.setSampleRate((ICM_20948_Internal_Acc | ICM_20948_Internal_Gyr), dmp_smplrt);
    }
    {
      ICM_20948_fss_t dmp_fss;
      dmp_fss.a = gpm2;
      dmp_fss.g = dps250;
      imu.setFullScale((ICM_20948_Internal_Acc | ICM_20948_Internal_Gyr), dmp_fss);
    }

    // Enable RAW accel + RAW gyro. NO compass — see the comment block above.
    imu.enableDMPSensor(INV_ICM20948_SENSOR_RAW_ACCELEROMETER);
    imu.enableDMPSensor(INV_ICM20948_SENSOR_RAW_GYROSCOPE);

    // Output each enabled sensor on every DMP cycle (max rate)
    imu.setDMPODRrate(DMP_ODR_Reg_Accel, 0);
    imu.setDMPODRrate(DMP_ODR_Reg_Gyro,  0);

    imu.enableFIFO();
    imu.enableDMP();
    imu.resetDMP();      // safe at SETUP (it's part of the init sequence)
    imu.resetFIFO();

    SERIAL_USB->print(F("ICM-20948 DMP enabled (accel+gyro @100Hz, no mag): "));
    SERIAL_USB->println(imu.statusString());
    wdt.restart();

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
