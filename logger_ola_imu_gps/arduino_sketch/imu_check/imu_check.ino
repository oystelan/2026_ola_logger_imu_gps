// Minimal IMU verification sketch for the SparkFun OLA's built-in ICM-20948.
//
// Goal: prove that the Arduino IDE toolchain (SparkFun Apollo3 v2.x board
// package) can talk to the IMU on this board. If WHO_AM_I reads back as
// 0xEA and live data streams, we know the toolchain switch is solid and can
// then port the full logger.
//
// Setup:
//   1. Install board package: Arduino IDE → File → Preferences → Additional
//      board manager URLs → add
//      https://raw.githubusercontent.com/sparkfun/Arduino_Apollo3/main/package_sparkfun_apollo3_index.json
//      Then Tools → Board → Boards Manager → search "Apollo3" → install
//      "SparkFun Apollo3 Boards" v2.x (e.g. 2.2.1).
//   2. Tools → Board → SparkFun Apollo3 → "RedBoard Artemis ATP".
//      (The OLA is wired the same way as ATP for the built-in IMU pins, and
//      this variant exposes SPI by default — SparkFun_Artemis_Module does not.)
//   3. Tools → Library Manager → search and install:
//        - SparkFun 9DoF IMU Breakout - ICM 20948 - Arduino Library
//   4. Tools → Port → COM5 (or whichever COM is the OLA's CH340).
//   5. Tools → Serial Monitor baud: 115200.
//   6. Upload and watch the serial monitor.

#include <SPI.h>
#include "ICM_20948.h"

// OLA built-in IMU pins (Apollo3 pad numbers — same on all Artemis variants).
static constexpr int PIN_IMU_CHIP_SELECT = 44;
static constexpr int PIN_IMU_POWER       = 27;
static constexpr int PIN_MICROSD_POWER   = 15;  // SD power is active-LOW
static constexpr int PIN_MICROSD_CS      = 23;
static constexpr int PIN_STAT_LED        = 19;  // STAT LED on OLA
// Hardcoded SPI pad numbers (do NOT use the MISO/MOSI/SCK macros — those
// can map to different pads depending on the selected board variant). The
// OLA always uses these specific Apollo3 pads regardless of variant choice.
static constexpr int PIN_SPI_SCK         = 5;
static constexpr int PIN_SPI_CIPO        = 6;   // MISO
static constexpr int PIN_SPI_COPI        = 7;   // MOSI
// SparkFun's OLA reference firmware uses 4 MHz. Confirmed working here once
// SD power is ON (SD module's MISO pull-up needed) + Apollo3 1.5K internal
// pull-up applied + applied AGAIN after SPI.beginTransaction (which resets it).
static constexpr int IMU_SPI_HZ          = 4000000;  // 4 MHz, matches SparkFun OLA firmware

ICM_20948_SPI imu;

static void imuPowerOn()  { digitalWrite(PIN_IMU_POWER, HIGH); }
static void imuPowerOff() { digitalWrite(PIN_IMU_POWER, LOW); }

void setup() {
  // STAT LED on so you can visually confirm the sketch is running.
  pinMode(PIN_STAT_LED, OUTPUT);
  digitalWrite(PIN_STAT_LED, HIGH);

  // Match SparkFun's OLA setup order EXACTLY: SPI.begin() very early,
  // before Serial / IMU / SD pin config. This might prime the IOM/pad
  // state in a way that later steps depend on.
  SPI.begin();

  Serial.begin(115200);
  while (!Serial && millis() < 3000) {}
  Serial.println();
  Serial.println(F("=== OLA ICM-20948 minimal check ==="));

  // Power ON the SD card. The SD card module on the OLA has its own MISO
  // pull-up resistor that adds in parallel with Apollo3's internal 1.5KΩ —
  // and with the SD card inserted, the card itself adds even more bus
  // conditioning. SparkFun's working firmware always has SD powered before
  // the IMU init (their sd.begin() call requires SD power).
  pinMode(PIN_MICROSD_POWER, OUTPUT);
  pin_config(PinName(PIN_MICROSD_POWER), g_AM_HAL_GPIO_OUTPUT);
  digitalWrite(PIN_MICROSD_POWER, LOW);   // active-LOW: LOW = ON
  pinMode(PIN_MICROSD_CS, OUTPUT);
  digitalWrite(PIN_MICROSD_CS, HIGH);     // deselect SD (its MISO tri-states)
  delay(50);  // SD needs time to power up

  // Enable a 1.5 KΩ pull-up on MISO (CIPO = Apollo3 pad 6).
  {
    am_hal_gpio_pincfg_t cipoPinCfg = g_AM_BSP_GPIO_IOM0_MISO;
    cipoPinCfg.ePullup = AM_HAL_GPIO_PIN_PULLUP_1_5K;
    pin_config(PinName(PIN_SPI_CIPO), cipoPinCfg);
  }

  // Now set up IMU pins and power-cycle (SparkFun's beginIMU does this last).
  pinMode(PIN_IMU_CHIP_SELECT, OUTPUT);
  pin_config(PinName(PIN_IMU_CHIP_SELECT), g_AM_HAL_GPIO_OUTPUT);
  digitalWrite(PIN_IMU_CHIP_SELECT, HIGH);

  pinMode(PIN_IMU_POWER, OUTPUT);
  pin_config(PinName(PIN_IMU_POWER), g_AM_HAL_GPIO_OUTPUT);
  imuPowerOff();
  delay(10);
  imuPowerOn();
  delay(25);

  // SD-card-style SPI bus warmup: clock 100+ dummy bytes with BOTH CS pins
  // HIGH. SparkFun's OLA firmware calls sd.begin() before the IMU which
  // pumps ~80+ dummy clocks per SD spec; the ICM-20948 may need to see SPI
  // bus activity to engage its SPI mode (vs locking to I2C auto-detect).
  digitalWrite(PIN_IMU_CHIP_SELECT, HIGH);
  digitalWrite(PIN_MICROSD_CS, HIGH);
  {
    SPISettings warmup(400000, MSBFIRST, SPI_MODE0);
    SPI.beginTransaction(warmup);
    for (int i = 0; i < 16; i++) SPI.transfer(0xFF);
    SPI.endTransaction();
  }
  delay(10);

  // Blink STAT to signal "configuration done, about to talk to IMU".
  for (int i = 0; i < 5; i++) {
    digitalWrite(PIN_STAT_LED, LOW);  delay(100);
    digitalWrite(PIN_STAT_LED, HIGH); delay(100);
  }

  // Helper to (re)apply 1.5KΩ pull-up on MISO. We re-apply right before each
  // probe in case SPI.beginTransaction/format() reconfigured the pad.
  auto applyMisoPullup = []() {
    am_hal_gpio_pincfg_t cfg = g_AM_BSP_GPIO_IOM0_MISO;
    cfg.ePullup = AM_HAL_GPIO_PIN_PULLUP_1_5K;
    pin_config(PinName(PIN_SPI_CIPO), cfg);
  };

  // PROBE A: per-byte transfers (what the SparkFun ICM library does internally).
  Serial.println(F("Probe A: 9 separate SPI.transfer(byte) calls..."));
  {
    SPISettings probe(1000000, MSBFIRST, SPI_MODE0);
    uint8_t rxA[9] = {0};
    SPI.beginTransaction(probe);
    applyMisoPullup();  // re-apply after beginTransaction in case it was reset
    digitalWrite(PIN_IMU_CHIP_SELECT, LOW);
    delayMicroseconds(5);
    rxA[0] = SPI.transfer(0x80);
    for (int i = 1; i < 9; i++) rxA[i] = SPI.transfer(0x00);
    delayMicroseconds(5);
    digitalWrite(PIN_IMU_CHIP_SELECT, HIGH);
    SPI.endTransaction();

    Serial.print(F("  rxA="));
    for (int i = 0; i < 9; i++) {
      Serial.print("0x"); if (rxA[i] < 0x10) Serial.print('0');
      Serial.print(rxA[i], HEX); Serial.print(' ');
    }
    Serial.println();
  }

  // PROBE B: same payload but as ONE multi-byte transfer (single IOM call,
  // no inter-byte gap).
  Serial.println(F("Probe B: single SPI.transfer(buf, 9) multi-byte..."));
  {
    SPISettings probe(1000000, MSBFIRST, SPI_MODE0);
    uint8_t bufB[9] = {0x80, 0, 0, 0, 0, 0, 0, 0, 0};
    SPI.beginTransaction(probe);
    applyMisoPullup();
    digitalWrite(PIN_IMU_CHIP_SELECT, LOW);
    delayMicroseconds(5);
    SPI.transfer(bufB, 9);
    delayMicroseconds(5);
    digitalWrite(PIN_IMU_CHIP_SELECT, HIGH);
    SPI.endTransaction();

    Serial.print(F("  rxB="));
    for (int i = 0; i < 9; i++) {
      Serial.print("0x"); if (bufB[i] < 0x10) Serial.print('0');
      Serial.print(bufB[i], HEX); Serial.print(' ');
    }
    Serial.println();
  }

  // PROBE C: per-byte at SLOWER 100 kHz to see if MISO has enough time to
  // settle. If at 100 kHz we get WHO_AM_I=0xEA, the issue is purely slow
  // MISO rise time and we need either a stronger pull-up or slower clock.
  Serial.println(F("Probe C: per-byte at 100 kHz..."));
  {
    SPISettings probe(100000, MSBFIRST, SPI_MODE0);
    uint8_t rxC[9] = {0};
    SPI.beginTransaction(probe);
    applyMisoPullup();
    digitalWrite(PIN_IMU_CHIP_SELECT, LOW);
    delayMicroseconds(5);
    rxC[0] = SPI.transfer(0x80);
    for (int i = 1; i < 9; i++) rxC[i] = SPI.transfer(0x00);
    delayMicroseconds(5);
    digitalWrite(PIN_IMU_CHIP_SELECT, HIGH);
    SPI.endTransaction();

    Serial.print(F("  rxC="));
    for (int i = 0; i < 9; i++) {
      Serial.print("0x"); if (rxC[i] < 0x10) Serial.print('0');
      Serial.print(rxC[i], HEX); Serial.print(' ');
    }
    Serial.println();
    Serial.println(F("  (idx 0 = junk during addr byte; idx 1 = WHO_AM_I; idx 2..8 = reg 0x01..0x07)"));
  }

  // Enable library debug so we see what the ICM_20948 lib does internally.
  imu.enableDebugging(Serial);

  bool ok = false;
  for (int attempt = 0; attempt < 5; attempt++) {
    Serial.print(F("Init attempt ")); Serial.print(attempt);
    imu.begin(PIN_IMU_CHIP_SELECT, SPI, IMU_SPI_HZ);
    // CAPTURE status from begin() before calling getWhoAmI(), which is its
    // own SPI read that overwrites imu.status. Otherwise we'd report a
    // false success whenever any byte (even a wrong one) reads back.
    ICM_20948_Status_e begin_status = imu.status;
    uint8_t who = imu.getWhoAmI();
    Serial.print(F("  begin status="));
    Serial.print((int)begin_status);  // 0=Ok, 4=WrongID
    Serial.print(F(" ("));
    Serial.print(imu.statusString(begin_status));
    Serial.print(F(")  WHO_AM_I=0x"));
    Serial.println(who, HEX);
    if (begin_status == ICM_20948_Stat_Ok) { ok = true; break; }
    delay(50);
  }

  if (!ok) {
    Serial.println(F("FAIL: IMU never identified. Check wiring/power/SPI mode."));
    // Fast blink to indicate failure state — different from the normal loop heartbeat.
    while (true) {
      digitalWrite(PIN_STAT_LED, HIGH); delay(80);
      digitalWrite(PIN_STAT_LED, LOW);  delay(80);
    }
  }

  Serial.println(F("OK: IMU online. Streaming AGMT data at ~10 Hz..."));
}

void loop() {
  // Slow heartbeat blink on STAT LED so you can see the sketch is alive.
  static unsigned long last_blink = 0;
  static bool led_state = true;
  if (millis() - last_blink > 500) {
    last_blink = millis();
    led_state = !led_state;
    digitalWrite(PIN_STAT_LED, led_state ? HIGH : LOW);
  }

  if (imu.dataReady()) {
    imu.getAGMT();
    Serial.print(F("acc(mg) "));
    Serial.print(imu.accX()); Serial.print('\t');
    Serial.print(imu.accY()); Serial.print('\t');
    Serial.print(imu.accZ()); Serial.print(F("\tgyr(dps) "));
    Serial.print(imu.gyrX()); Serial.print('\t');
    Serial.print(imu.gyrY()); Serial.print('\t');
    Serial.print(imu.gyrZ()); Serial.print(F("\ttemp(C) "));
    Serial.println(imu.temp());
  }
  delay(100);
}
