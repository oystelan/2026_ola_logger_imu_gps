# OLA ICM-20948 + u-blox GNSS Data Decoder + Post-processing

Python decoder and analysis pipeline for `.dat` files produced by the OLA (OpenLogArtemis) firmware in `../logger_ola_imu_gps`. The decoder parses the binary stream into NumPy arrays; the post-processing modules then turn those arrays into vertical motion, fused position, or other application-specific outputs.

## What this package does

1. **Decode** binary `.dat` recordings → `.npz` compressed archives → NumPy arrays.
2. **Outlier detection** + spike rejection on the IMU stream (per-axis and magnitude).
3. **Timestamp sync** between PPS and GNSS to give microsecond-accurate UTC.
4. **Vertical motion estimation** (three methods — Madgwick AHRS, complementary filter, savgol-detrend) followed by FFT band-pass double integration to displacement.
5. **15-state error-state EKF** for full 6-DoF fusion of IMU + GNSS + magnetometer, with an optional **static-position pseudo-measurement** that makes the EKF usable in GNSS-denied environments (moored buoys, basins, indoor tests).
6. **Plotting helpers** for raw inspection and standard processing-pipeline output.

## Hardware backstory

The firmware logs the SparkFun OLA's built-in **ICM-20948** (3-axis accel + 3-axis gyro + AK09916 mag) at 225 Hz over SPI, plus a u-blox GNSS module on QWIIC (PVT at 10 Hz + PPS). The IMU samples land in the file already remapped from chip to PCB silkscreen body frame, and with gyro and mag bias offsets pre-subtracted (Tier 1 & Tier 2 calibrations stored in the OLA's EEPROM — see the firmware README for the RESET-button calibration triggers).

## Project structure

```
.
├── decoder.py                 # Core binary parser, segment splitter, outlier detection
├── decoder_cli.py             # Command-line interface for quick visualisation
├── ahrs_vertical.py           # AHRS + FFT vertical-motion pipeline
├── sensor_fusion.py           # 15-state ESKF for IMU/GNSS/mag fusion
│
├── ahrs_example.py            # Madgwick / complementary / savgol vertical-motion demo
├── ekf_fusion_example.py      # EKF demo with real GNSS
├── ekf_anchored_example.py    # EKF demo with static-position anchor (GNSS-denied)
├── plot_raw_accel.py          # 4-figure raw-IMU diagnostic plot
│
├── simple_example.py          # Decode + plot in <30 lines (start here)
├── example_decode.py          # Detailed decode example (dataclass usage)
├── example_segments.py        # Segment-based processing example
├── demo_corruption_recovery.py
│
├── environment.yml            # mamba/conda environment
├── AGENT.md                   # Dev guidelines / coding practices
├── SPEC.md                    # Binary file format spec
└── test_*.py                  # Pytest suites
```

## Installation

```bash
# Create the env from the YAML (use -y for non-interactive)
mamba env create -f environment.yml -y
mamba activate ola_ism330dhcx_samm10q_decoder
```

(The env name still references the old hardware — it's a label, not a constraint, so no rename is needed unless you want one.)

## Quick start

The fastest path from `.dat` to a plot:

```bash
python simple_example.py
```

This decodes the first `.dat` in the working directory and produces 4 plots — accel, gyro, GPS track, GNSS velocities — all with outlier markings.

For a deeper inspection (gaps, sample-rate stability, all three sensors, mag-vector magnitude):

```bash
python plot_raw_accel.py path/to/file.dat
```

This emits four figures:
1. Raw accel per axis (mg) with gap shading.
2. Raw gyro per axis (deg/s).
3. Integrated gyro per axis — degrees of cumulative rotation; X/Y are also shown high-passed at 0.1 Hz to expose the slow drift (= gyro bias).
4. Raw magnetometer per axis (µT) plus `|B|` total field with the median as a dashed reference.

## Vertical-motion pipelines

For wave-buoy / basin / motion-of-interest use cases, see `ahrs_example.py`:

```python
from ahrs_vertical import (
    compute_vertical_motion,                    # Madgwick AHRS
    compute_vertical_motion_lowpass_gravity,    # Complementary filter (gyro + low-pass accel)
    compute_vertical_motion_savgol_detrend,     # Pure gyro + savgol detrend
)
```

All three return a `VerticalAHRSResult` with the same dataclass shape:

- `t`, `accel_z_up`, `velocity_z_up`, `displacement_z_up` — time series in m/s², m/s, m
- `roll_deg`, `pitch_deg`, `fs_hz`, `low_hz`, `high_hz`

The three differ in how they recover body→world attitude:

| Method | Attitude source | Best for |
|---|---|---|
| `compute_vertical_motion` (Madgwick) | Gyro propagation + motion-gated accel correction | Recordings with brief stationary moments. After axis-remap fixes, this is the default — see `METHOD = "madgwick"` in `ahrs_example.py`. |
| `compute_vertical_motion_lowpass_gravity` | Gyro for fast attitude + low-pass-accel for slow gravity anchor | Sustained rocking-while-translating motion where Madgwick's accel correction would fight the gyro. |
| `compute_vertical_motion_savgol_detrend` | Pure gyro + Savitzky-Golay detrend of Euler angles | Wave-buoy deployments where the mean attitude is approximately level; assumes any slow attitude trend is gyro drift. |

The actual displacement integration is the same FFT band-pass-and-double-integrate kernel for all three. The default band is `[0.05, 2.5]` Hz, matching typical wave / human-scale motion.

## EKF fusion

Two example scripts, one for each environment:

```bash
# Outdoor / valid GNSS fix in the recording
python ekf_fusion_example.py path/to/file.dat
```

Uses the full 15-state error-state EKF with loose-coupled GNSS PVT updates and (optional) magnetometer yaw updates. Output: position / velocity / attitude with 1-sigma bands.

```bash
# Indoor / basin / handheld test (no usable GNSS)
python ekf_anchored_example.py path/to/file.dat
```

Same EKF, but with a synthetic **static-position pseudo-measurement** at the origin injected at a configurable cadence. This makes accel-bias observable (without a position anchor of some kind, the IMU-only EKF runs away by kilometres in dead-reckoning). Tunables at the top of the script:

- `ANCHOR_SIGMA_HORIZ_M` / `_VERT_M` — how strongly to anchor (larger σ = lets real motion through, smaller σ = stronger anchor).
- `ANCHOR_CADENCE_HZ` — how often the synthetic measurement fires (default 10 Hz).
- `APPLY_MAG_UPDATES` — default `False` for anchored use because mag yaw updates can fight the gyro when no clean stationary init period is available.

## Decoded data fields

`load_data_as_arrays(npz_file)` returns a dict with:

**Header / metadata**
- `firmware_commit` — 40-char SHA
- `imu_odr` — Hz
- `gnss_rate` — Hz
- `acc_sensitivity`, `gyr_sensitivity` — mg/LSB and mdps/LSB from the header
- `number_of_segments` — integer count

**PPS**
- `pps_utc`, `pps_micros`, `pps_micros_unwrapped`

**GNSS**
- `gnss_utc`, `gnss_micros_unwrapped`, `gnss_latitude` / `_longitude` (degrees), `gnss_altitude_msl` (m)
- `gnss_vel_north` / `_east` / `_down` (mm/s)
- `gnss_fix_type` (0=no fix, 2=2D, 3=3D, …)
- `gnss_posix_timestamp` (raw POSIX seconds from receiver) + `gnss_microseconds` (sub-second part)

**IMU**
- `imu_utc`, `imu_micros_unwrapped`, `imu_counter`, `imu_counter_unwrapped`
- `imu_acc_x` / `_y` / `_z` (mg) — already in PCB-silkscreen body frame
- `imu_gyr_x` / `_y` / `_z` (mdps) — already in PCB-silkscreen body frame, gyro bias subtracted
- `imu_mag_x` / `_y` / `_z` (µT) — chip frame (= mag silkscreen drawing), hard-iron subtracted

**Outlier flags** (boolean arrays, same length as the data)
- `imu_acc_*_outlier`, `imu_gyr_*_outlier`, `imu_mag_*_outlier`

See `decoder.py:load_data_as_arrays` docstring for the full list.

## Important: mag silkscreen vs accel/gyro silkscreen

The OLA has two axis-cross drawings on the silkscreen — one near the IMU footprint for accel/gyro, and one with the mag y-axis and z-axis arrows pointing in **opposite** directions to the accel/gyro cross. The firmware logs raw chip values for mag (= mag silkscreen frame); the accel and gyro are remapped into the accel/gyro silkscreen frame as described in `SPEC.md`. **When fusing mag with accel/gyro in a common body frame**, flip `imu_mag_y` and `imu_mag_z`:

```python
mag_in_acc_gyr_frame = np.column_stack([
     data["imu_mag_x"],
    -data["imu_mag_y"],
    -data["imu_mag_z"],
])
```

This is done inside `sensor_fusion.run_fusion()` automatically when the EKF loads mag updates.

## Time synchronization

When PPS + GNSS are present, the decoder fits a linear regression between PPS rising-edge timestamps and the GNSS-reported UTC, then applies that to every IMU sample. Typical R² > 0.999999. Access:

- `imu_utc`, `gnss_utc`, `pps_utc` — UTC seconds since 1970 from the regression
- Each dataclass entry also has `datetime_timestamp_from_pps_regression` (timezone-aware)

When PPS isn't available, `decode_file(..., allow_no_pps=True)` skips the regression and uses raw `micros()` as the timeline (relative time only). All post-processing modules accept this.

## Corruption recovery

If a recording is interrupted (power cut, SD eject mid-write), the decoder will:

1. Detect the corruption at the first byte that doesn't match a valid marker.
2. Scan ahead up to 1024 bytes for the next valid marker (`\nPPS`, `\nGPS`, `\nIMU`).
3. Resume parsing from there.
4. If no marker is found, save everything up to the corruption point.

```bash
python demo_corruption_recovery.py    # walk-through of the recovery flow
```

## Running tests

```bash
pytest -v .
```

Specific suites:

- `test_decoder.py` — core decoder logic
- `test_corruption_recovery.py` — corruption handling
- `test_edge_cases.py`
- `test_outlier_detection.py`
- `test_regression_boot8.py`

## Code quality

```bash
ruff check .                          # fast lint
pylint decoder.py test_decoder.py     # deeper lint
flake8 decoder.py
complexipy decoder.py
```

## File format

See `SPEC.md`.

## Pointers for new readers

- **Just want a quick plot of your recording?** → `simple_example.py` or `plot_raw_accel.py`
- **Vertical motion (wave buoy, basin, handheld)?** → `ahrs_example.py`
- **Full 6-DoF with real GNSS?** → `ekf_fusion_example.py`
- **Indoor / basin EKF (no GNSS)?** → `ekf_anchored_example.py`
- **Wondering about a field in the binary file?** → `SPEC.md`
- **Adding a new processing step?** → see `AGENT.md` for code conventions
