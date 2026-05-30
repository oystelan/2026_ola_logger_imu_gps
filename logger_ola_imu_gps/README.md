# OLA Waves Logger (OWL)

A high-rate, low-jitter inertial + GNSS data logger built around the **SparkFun OpenLog Artemis (OLA)**, using its built-in ICM-20948 9-DoF IMU and a u-blox GNSS receiver on QWIIC. Designed primarily for wave-buoy / motion-sensing work but works as a general-purpose IMU+GNSS logger.

## Hardware

### Required

- **Main board:** [SparkFun OpenLog Artemis](https://www.sparkfun.com/openlog-artemis) — the *with-IMU* variant. The on-board ICM-20948 (3-axis accel + 3-axis gyro + AK09916 magnetometer) is wired to the Artemis over SPI and is the primary inertial sensor.

### Optional but recommended

- **GNSS receiver on QWIIC** — any u-blox module that speaks the standard u-blox UBX protocol over I2C. We use the **u-blox MAX-M10S** but any of the MAX-M10S / SAM-M10Q / NEO-M9N family works. Connect the module's PPS pin to **OLA pin 11** if you want sub-sample-period UTC timestamps for the IMU samples.
- **VBAT coin cell (CR1225)** — keeps the Apollo3's hardware RTC alive across power-off so UTC timestamps survive reboots even without a GNSS fix. See [Time keeping](#time-keeping-utc-across-power-cycles) below.
- **QWIIC cable** between OLA QWIIC port and GNSS module.

The firmware **detects at boot** whether a GNSS module is present. If yes, it waits for a valid fix before starting to log. If no, it starts logging immediately with the best available time source — see [Runtime GNSS detection](#runtime-gnss-detection).

### Power supply options

Two reasonable approaches:

- **Non-rechargeable batteries.** SAFT LSH20 are good in cold conditions; alkaline cells in normal conditions. Design the pack for 3.6–4.5 V, then use a [Pololu step-up/step-down regulator](https://www.pololu.com/product/2122) to feed clean 3.3 V to the OLA's 3V3 pin (and GND to GND).
- **Rechargeable LiPo.** The OLA accepts SparkFun's [LiPo cells](https://www.sparkfun.com/lithium-ion-battery-2ah.html) directly on the JST connector. Pair with a [LiPo charger](https://www.sparkfun.com/sparkfun-lipo-charger-plus.html).

To minimise standby current, cut the power LED pads on the back of any breakout boards that have them.

### SD card

Use a good-quality SD card formatted as FAT. When formatting, use the **overwrite** option so the card is filled with `\0` — this matters because the firmware pre-allocates each log file at fixed size, and `\0` makes it trivial to find the actual end-of-data later. SD card choice has a big impact on both power consumption and write throughput — worth testing a few models before committing to a batch.

## PCB

A PCB design for an OLA-based logger with capacity for 3 × LSH20 batteries is in the `pcb/` folder (production gerbers included).

## IMU axis convention

The ICM-20948 die inside the OLA package is rotated relative to the PCB silkscreen arrows, and *the accel and gyro dies inside the chip are at different orientations to each other*. The firmware applies a chip→PCB body-frame remap so the output channels match the PCB silkscreen arrows in both magnitude and sign:

| Axis | Accel mapping | Gyro mapping |
|---|---|---|
| `PCB_x` | `chip_z` | `chip_x` |
| `PCB_y` | `chip_x` | `chip_y` |
| `PCB_z` | `chip_y` | `chip_z` |

Net effect for the user:

- **Accel:** point any silkscreen arrow downward (toward gravity) → that channel reads **−1 g**. (Specific-force convention: arrow up reads +1 g.)
- **Gyro:** rotate about any silkscreen axis using the right-hand rule → that channel reads **positive**.
- **Mag:** logged as raw chip-frame output. The AK09916 die's orientation relative to the PCB has not been verified yet; **do a compass-bearing test before relying on yaw**.

If you replace the OLA with a different unit and the values look swapped, the chip-to-PCB orientation may differ — re-derive the mapping by laying the PCB flat in each of the 6 stable poses and noting which channel reads ±1 g.

## Calibration

The hardware RESET button doubles as a calibration trigger. Each press reboots the chip; consecutive presses inside a ~2.5 s window are detected via a decision-pending flag in EEPROM:

| Presses | Action |
|---|---|
| 1 (normal) | Boot normally; load stored biases from EEPROM |
| **2** | **Tier 1: gyro bias calibration** — keep the device perfectly still for 5 s after the second press. The firmware averages gyro readings over 5 s, stores the bias in EEPROM, and from then on subtracts it from every logged sample. Survives power loss. |
| **3** | Tier 2: magnetometer hard-iron calibration (reserved; not yet implemented) |

The STAT LED blinks during the decision window — faster blinks mean a higher press count has been registered, giving you live feedback that the multi-press is being detected.

## Time keeping (UTC across power-cycles)

The firmware seeds its software POSIX-time counter at boot from the **first available** of:

1. **Apollo3 hardware RTC** (battery-backed by the VBAT coin cell). The H/W RTC's date/time registers are written every time the firmware receives a GNSS fix and keep counting on their own from the 32.768 kHz XT crystal across power-off. Drift is ~±20 ppm → ~1.7 s/day. On boot the H/W RTC is read and the value is accepted if year ≥ 2025.
2. **EEPROM-stored last-known UTC** (only if `ENABLE_TIME_EEPROM_FALLBACK` is set in `firmware_configuration.h`, default on). Saved at every 15-minute file rotation. Less accurate than the H/W RTC (only as fresh as the last save) but recovers something useful on a board without a coin cell.
3. Fall through to **0** (1970 epoch) — same as before. Will get overridden once GNSS locks.

After the seed, whenever a GNSS fix arrives, the firmware:
- Writes the GNSS UTC to **both** the software counter and the H/W RTC date/time registers.
- **Compares** the GNSS UTC to the pre-sync software counter (only if the seed had come from the H/W RTC). If `|delta| > 1 s`, the next two GNSS entries written to the data file are tagged with `posix_timestamp = 0` — a **visible 1970-spike marker** in the data file that immediately shows when re-syncing happened and that the previous timestamps in this file are likely off.

The threshold and marker length are constants in `main.cpp`; change them if you want a different sensitivity or visibility.

## Runtime GNSS detection

`log_GNSS.begin()` is attempted at every setup attempt. The runtime outcome controls the rest of boot:

- **GNSS responds** → `g_gnss_present = true`. If `ENABLE_GNSS_START` is also true (default), the firmware waits up to 2 minutes for a valid fix before starting to sample. Subsequent loops capture PVT at 10 Hz and PPS via external-interrupt ISR.
- **GNSS does not respond** → `g_gnss_present = false`. The firmware proceeds **without GNSS**, using the H/W-RTC or EEPROM-seeded UTC for timestamps. The GNSS-read and PPS code paths in the ISR are gated off so no I2C cycles are wasted polling a non-existent device.

In other words: plug in a GNSS to wait for satellite time; leave it unplugged for indoor / quick-test work and the firmware uses persistent UTC instead.

## Sampling architecture

The aim is a robust, high-accuracy, high-frequency, low-jitter logger.

- **IMU at 225 Hz** over SPI (4 MHz) via a CTIMER-driven ISR. ACCEL samples are buffered in the ICM-20948's on-board FIFO (4 KB ≈ 1.5 s of headroom at 225 Hz) so that SD-card write blocking (typically up to ~700 ms per stall) doesn't drop samples. The ISR drains the FIFO into a ring-buffer deque whenever it gets a chance. Gyro and mag are read once per ISR call via `getAGMT()` and stamped onto each FIFO sample.
- **GNSS PVT at 10 Hz** over QWIIC I2C (400 kHz), captured by the same CTIMER ISR.
- **PPS rising edge** captured by an external-interrupt ISR on pin 11.
- **SD-card writes** run asynchronously in a busy loop in the main thread, draining the IMU/GNSS/PPS deques. The watchdog covers the whole sketch and will hard-reboot the board if anything stalls long enough.
- **New file every 15 minutes** of UTC. Files are pre-allocated to ~12 MB.

## Data files

Binary `.dat` files on the SD card. Each boot creates a folder `BOOT_NNNNNN/` containing `.dat` files named with the UTC start time, e.g. `BOOT_000349/DATA_BOOT_000349_TIME_20260530T134500.dat`.

See the [decoder/](decoder/) folder for:

- `decoder.py` — binary `.dat` parser, segment splitter, outlier detection
- `ahrs_vertical.py` — three vertical-motion estimators on the IMU stream (Madgwick AHRS, complementary filter, savgol-detrend) followed by FFT band-pass double integration to displacement
- `sensor_fusion.py` — 15-state error-state EKF that loosely fuses IMU + GNSS + magnetometer for full 6-DoF position/velocity/attitude
- `ahrs_example.py`, `ekf_fusion_example.py`, `plot_raw_accel.py` — usage examples and plotting utilities

## LED indicators

| LED | Meaning |
|---|---|
| STAT (blue) | Blinks during boot setup. During the calibration decision window after RESET it blinks at a rate that scales with the press count. During logging, on while writing to SD, off otherwise (so usually flickers). |
| PWR (red) | Steady when powered. Optional startup-blink pattern controlled by `ENABLE_BLINK_PWR_LED`. |
| PPS | Blinks at 1 Hz when the GNSS has a fix. |

## Compiling / Uploading

The project uses the SparkFun Artemis Arduino core v1 via PlatformIO: see [github.com/nigelb/platform-apollo3blue](https://github.com/nigelb/platform-apollo3blue). Make sure to choose **Core V1**. All dependencies are vendored in the `lib/` folder.

Build + flash:
```
pio run -t upload
```

Pre-built `.bin` files (when provided) can also be flashed directly via the [Artemis Firmware Upload GUI](https://github.com/sparkfun/Artemis-Firmware-Upload-GUI).

## Configuration knobs

Most user-relevant constants live in:

- `firmware_configuration.h` — `ENABLE_TIME_EEPROM_FALLBACK`, pin assignments, baudrates
- top of `main.cpp` — `ENABLE_GNSS`, `ENABLE_GNSS_START`, `ENABLE_BLINK_PWR_LED`, IMU sample rate (`IMU_ODR_HZ` + `IMU_SMPLRT_DIV`), GNSS update rate (`GNSS_FREQUENCY_HZ`)

## Disclaimers

This started as a clean project and ended up as a mix of old libs, new libs, custom libs, and assorted code accumulated from years of related work. There are corners that are still messy.

## Serial logs

At baudrate 1000000 over USB, the logger prints status during boot and periodic rate/deque summaries during logging. A typical session looks something like:

```
=== GYRO CALIBRATION ===  (only if RESET was double-pressed)
Keep the device PERFECTLY STILL for 5 seconds...
...

RTC seeded from H/W RTC (battery-backed): 1748640000
- TimeManager -
posix_is_set = true
posix_timestamp: 1748640000
gregorian: 2026-05-31T00:00:00Z

Setup attempt #: 1
Starting I2C QWIIC...
I2C QWIIC started
success starting GNSS
GNSS set to UBX output
Current update rate: 10
Waiting for GNSS fix...
..........
GNSS fix acquired.
GNSS vs H/W-RTC drift = 0 s — within tolerance, no marker
GNSS setup complete.
...
Preparing to start new log file...
Opening file: DATA_BOOT_000350_TIME_20260530T134500.dat
File opened successfully
Preallocating 12582912 bytes...
File preallocated successfully
Current posix timestamp: 1748641500
Next log file posix timestamp: 1748642400
Logging...

millis(): 15039; seconds since boot: 15
Samples logged in last interval: IMU: 2247; GNSS: 95; PPS: 8
Max deque sizes reached: IMU: 21 over 4500; FIFO: 12 over 4096; GNSS: 1 over 200; PPS: 1 over 20
Effective logging rates (Hz): IMU: 224.70; GNSS: 9.50; PPS: 0.80
Accumulated SD time (ms): 4810 ms over 10000 ms interval
```

If no GNSS is detected, the GNSS-related lines are replaced by:

```
No GNSS module detected on QWIIC — proceeding without GNSS;
timestamps will use the persistent RTC seed (H/W RTC or EEPROM).
```

and the firmware proceeds straight to IMU + SD setup.
