# Arduino IDE sketches

This folder hosts Arduino IDE sketches for the project. We're here because PlatformIO's `apollo3blue` platform pins to Apollo3 Arduino framework v1.2.3, which has an IOM SPI bug that prevents the built-in ICM-20948 on the OLA from being read. SparkFun's reference OLA firmware works on v2.x (mbed-os based) via Arduino IDE — same hardware, same chip, different framework — so we develop here.

## Sketches

- **`imu_check/`** — minimal sketch that just initializes the built-in ICM-20948 and streams accelerometer + gyro + temperature to the serial monitor. Use this first to confirm the Arduino IDE toolchain is working before porting the full logger.

## First-time setup (Arduino IDE 2.x)

1. **Add SparkFun board manager URL.** File → Preferences → "Additional boards manager URLs" → add:
   ```
   https://raw.githubusercontent.com/sparkfun/Arduino_Apollo3/main/package_sparkfun_apollo3_index.json
   ```

2. **Install the Apollo3 v2.x board package.** Tools → Board → Boards Manager → search "Apollo3" → install **SparkFun Apollo3 Boards** version `2.2.1` (or `2.2.2` if available).

3. **Select board.** Tools → Board → SparkFun Apollo3 → **RedBoard Artemis ATP**. (The OLA isn't its own variant in v2.x; the ATP variant exposes the SPI pins the OLA uses for the built-in IMU. The Artemis Module variant explicitly disables SPI in v2.x.)

4. **Install required libraries** (Tools → Library Manager):
   - `SparkFun 9DoF IMU Breakout - ICM 20948 - Arduino Library`
   - (More to be added when we port the full logger: SdFat, SparkFun u-blox GNSS v3, etc.)

5. **Connect** the OLA via USB. Tools → Port → pick the CH340 COM (probably `COM5`).

## Running `imu_check`

1. Open `arduino_sketch/imu_check/imu_check.ino` in Arduino IDE.
2. Sketch → Upload.
3. Tools → Serial Monitor (set baud to `115200`).

Expected output:
```
=== OLA ICM-20948 minimal check ===
Init attempt 0  status=All is well.  WHO_AM_I=0xEA
OK: IMU online. Streaming AGMT data at ~10 Hz...
acc(mg) -7.81   3.91    1010.74   gyr(dps) 0.10  -0.05   0.02   temp(C) 24.5
...
```

If `WHO_AM_I=0xEA`, the Arduino IDE toolchain is solid and we can proceed with porting the full logger.
