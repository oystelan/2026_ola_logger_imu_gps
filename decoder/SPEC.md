# Specification of the data format on the SD card

The information below is from the PlatformIO C++ project in the sibling directory: see `../logger_ola_imu_gps` for the firmware. Data files are pre-allocated to a fixed size (≈ 12 MB by default), so the tail of each file may contain trailing `\0` bytes or whatever was previously on the SD card.

## Hardware

- **IMU:** ICM-20948 (3-axis accel + 3-axis gyro on one die, AK09916 3-axis magnetometer on a second die) — wired to the Artemis over SPI on the built-in OLA IMU socket.
- **GNSS:** u-blox MAX-M10S (or any u-blox module that speaks UBX over I2C) attached on QWIIC, with the PPS pin wired to OLA pin 11.

Default logging rates:

- IMU (accel + gyro + mag): **225 Hz** (chip ODR = 1125 Hz with `SMPLRT_DIV = 4`). Accel is buffered through the ICM-20948's on-chip FIFO; gyro and mag come from a `getAGMT()` direct register read on each ISR tick.
- GNSS position (PVT): **10 Hz**.
- PPS (GNSS rising-edge): nominally **1 Hz** when a fix is available.

Files rotate every 15 minutes of UTC. A file opened mid-window ends at the next 00 / 15 / 30 / 45 minute boundary. Each boot creates a sub-folder `BOOT_NNNNNN/`.

## File name

```cpp
snprintf(filename_buffer, sizeof(filename_buffer),
         "DATA_BOOT_%04u_TIME_%02u%02u%02uT%02u%02u%02u.dat",
         boot_count,
         common_working_struct_YMDHMS.year,
         common_working_struct_YMDHMS.month,
         common_working_struct_YMDHMS.day,
         common_working_struct_YMDHMS.hour,
         common_working_struct_YMDHMS.minute,
         common_working_struct_YMDHMS.second);
```

Time in the filename is whatever the firmware's POSIX counter holds at file-open. With a working VBAT coin cell *or* the EEPROM-time fallback, this is real UTC; without either it's `19700101T...` until GNSS locks.

## File layout

### Header (ASCII, ends with a blank line)

```
Log start OLA ICM-20948 logger

Firmware commit ID: <40-char SHA>
ICM-20948 Acc sensitivity (mg/LSB): 0.061035
ICM-20948 Gyr sensitivity (mdps/LSB): 7.633588
ICM-20948 ODR (Hz): 225.00
GNSS update rate (Hz): 10
```

The sensitivities depend on the configured full-scale ranges (default `±2 g` for accel → `1000/16384 mg/LSB`; `±250 dps` for gyro → `1000/131 mdps/LSB`). The magnetometer is a fixed `0.15 µT/LSB` (AK09916 datasheet) and is not written into the header — the decoder applies the conversion directly.

### Body (binary entries)

Each entry starts with a 4-byte ASCII marker that disambiguates kind, then a raw dump of the corresponding C++ struct.

#### Markers

| Kind | First 4 bytes |
|---|---|
| PPS | `\n` `P` `P` `S` |
| GNSS | `\n` `G` `P` `S` |
| IMU | `\n` `I` `M` `U` |

#### PPS struct

```cpp
struct PPS_fix {
  unsigned long micros_reading;   // 4 bytes (Apollo3 long = 32-bit)
};
```

#### GNSS struct

```cpp
struct GNSS_reading {
  unsigned long micros_reading;
  int32_t       latitude;
  int32_t       longitude;
  uint32_t      posix_timestamp;
  uint32_t      microseconds;
  int32_t       NED_vel_north;
  int32_t       NED_vel_east;
  int32_t       NED_vel_down;
  int32_t       altitude_msl_mm;
  uint8_t       fix_type;
};
```

Units:

- `latitude`, `longitude`: degrees × 10⁷ (UBX convention)
- `posix_timestamp`: seconds since 1970, with the sub-second part in `microseconds`
- `NED_vel_*`: mm/s
- `altitude_msl_mm`: metres above mean sea level × 10³
- `fix_type`: u-blox fix type (0 = no fix, 2 = 2D, 3 = 3D, …)

The decoder also recognises a magic **`posix_timestamp == 0` marker pair** the firmware writes on the next two GNSS samples after a re-sync where the H/W RTC seed disagreed with GNSS UTC by more than 1 s — visible as a 1970 spike in any plot of UTC vs sample index.

#### IMU struct

```cpp
struct IMU_reading {
  unsigned long micros_reading;
  uint16_t      counter;
  int16_t       acc_x, acc_y, acc_z;
  int16_t       gyr_x, gyr_y, gyr_z;
  int16_t       mag_x, mag_y, mag_z;
};
```

The struct is **26 bytes** on disk (no padding needed — all fields are 2- or 4-byte aligned). On the chip the `counter` field comes from a free-running 16-bit ISR counter; the decoder unwraps it modulo 65536 to give a monotonic sequence number.

**Axis convention.** The IMU values written here are in the **PCB silkscreen body frame**, *not* the raw chip frame:

- The accel die is rotated relative to the PCB silkscreen, so the firmware applies a 3-axis cycle `PCB_x = chip_z`, `PCB_y = chip_x`, `PCB_z = chip_y`. After this remap, pointing any silkscreen accel-arrow down (toward gravity) reads **−1 g** on the same-letter output channel.
- The gyro die is mounted aligned with the silkscreen, so it's pass-through. Right-hand-rule rotation about any silkscreen arrow gives a positive reading on the same-letter channel.
- The magnetometer is pass-through. The AK09916's chip axes happen to match the silkscreen *magnetometer* triad — but note that the mag silkscreen cross on the OLA has its Y and Z arrows opposite to the accel/gyro silkscreen cross. So when fusing mag with accel/gyro, downstream code must flip `my` and `mz` to express the mag vector in the accel/gyro body frame.

**Bias subtraction.** Both gyro and mag samples are bias-corrected in the ISR before being written: gyro uses the EEPROM-stored Tier-1 calibration (`save_gyro_bias` in `CalibrationManager`), and mag uses the EEPROM-stored Tier-2 hard-iron offset (`save_mag_hardiron`). Logged values are therefore already calibrated for the device they were recorded on.

**Magnetometer caveats.** The AK09916 internal sample rate is ≈ 100 Hz, so at the 225 Hz IMU rate roughly half of consecutive samples carry repeated mag values (the latest cached read). This is normal and is harmless for any decimated or filtered downstream use. The values are in chip LSB; multiply by `0.15` for µT.

### Footer

```
\n\nLog stop OLA ICM-20948 logger\n
```

Anything beyond the footer (up to the file-end at the pre-allocated size) is leftover SD-card content and should be ignored — the decoder stops at the footer or at a recognisable corruption boundary, whichever comes first.

## Float-from-int conversion

| Field | Formula | Units |
|---|---|---|
| `acc_*` | `acc_* × acc_sensitivity (mg/LSB) / 1000` | g |
| `gyr_*` | `gyr_* × gyr_sensitivity (mdps/LSB) / 1000` | dps |
| `mag_*` | `mag_* × 0.15` | µT |
| `latitude` / `longitude` | `value × 10⁻⁷` | degrees |
| `NED_vel_*` | `value × 10⁻³` | m/s |
| `altitude_msl_mm` | `value × 10⁻³` | m |
| `posix_timestamp + microseconds × 10⁻⁶` | direct | seconds since 1970 |
