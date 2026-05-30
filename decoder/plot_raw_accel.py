#!/usr/bin/env python3
"""Plot raw IMU signals (accel, gyro, integrated-gyro, magnetometer) from a .dat.

Four diagnostic figures:
  1. Accelerometer — mg per axis. Gaps in the sample stream highlighted in
     orange.
  2. Gyroscope — deg/s per axis.
  3. Integrated gyroscope — cumulative angle per axis. X/Y are high-passed at
     HIGHPASS_HZ to suppress slow gyro-bias drift; Z is left raw (no gravity
     reference to anchor it).
  4. Magnetometer — µT per axis plus |B| total. Useful for the chip→PCB
     orientation test: point each silkscreen mag axis at magnetic north for
     ~10 s and observe which output channel responds.

Usage:
    python plot_raw_accel.py [path/to/file.dat]

If no path is given, picks the most recently modified DATA_BOOT_*.dat in
the script directory.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

from decoder import decode_file, load_data_as_arrays


# X/Y integrated-angle high-pass cutoff. Removes slow gyro-bias drift
# (and any slow thermal wander on top) while keeping real rotation motion.
HIGHPASS_HZ = 0.1


def find_default_data_file() -> Path | None:
    """Pick the newest DATA_BOOT_*.dat in the script directory."""
    here = Path(__file__).parent
    candidates = list(here.glob("DATA_BOOT_*.dat"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    if len(sys.argv) > 1:
        data_file = Path(sys.argv[1])
    else:
        data_file = find_default_data_file()
        if data_file is None:
            print("No data file given and no DATA_BOOT_*.dat found.")
            print("Usage: python plot_raw_accel.py [path/to/file.dat]")
            return
        print(f"Auto-selected newest: {data_file.name}")

    if not data_file.exists():
        print(f"Not found: {data_file}")
        return

    # Decode + load
    result = decode_file(data_file, allow_no_pps=True)
    data = load_data_as_arrays(result["file"])

    if len(data["imu_micros"]) == 0:
        print("No IMU samples in this file.")
        return

    # Use raw micros (no PPS / UTC required); seconds since first sample
    t_us = np.asarray(data["imu_micros_unwrapped"], dtype=np.float64)
    t = (t_us - t_us[0]) * 1e-6

    # Highlight any gaps > 2× the median sample interval
    dts = np.diff(t)
    dt_med = float(np.median(dts))
    gap_threshold = max(2 * dt_med, 0.020)  # 20 ms floor
    gap_idx = np.where(dts > gap_threshold)[0]

    # Stats
    print(f"\nSamples:           {len(t)}")
    print(f"Duration:          {t[-1]:.1f} s")
    print(f"Median dt:         {dt_med*1000:.3f} ms (~{1/dt_med:.1f} Hz)")
    print(f"Max dt:            {dts.max()*1000:.1f} ms")
    print(f"Gaps > {gap_threshold*1000:.0f} ms:    {len(gap_idx)}")
    if len(gap_idx) > 0:
        print(f"  Worst gap: {dts[gap_idx].max()*1000:.1f} ms at t={t[gap_idx[np.argmax(dts[gap_idx])]]:.1f} s")

    labels = ("X", "Y", "Z")
    colors = ("tab:blue", "tab:green", "tab:red")
    gap_title = (
        f"{len(t)} samples, {len(gap_idx)} gaps > {gap_threshold*1000:.0f} ms"
        + (f" (max {dts.max()*1000:.0f} ms)" if len(gap_idx) else "")
    )

    # Plot 1: accelerometer (mg)
    fig_acc, axes_acc = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for k, (lab, c) in enumerate(zip(labels, colors)):
        axes_acc[k].plot(t, data[f"imu_acc_{lab.lower()}"], color=c, linewidth=0.6)
        for gi in gap_idx:
            axes_acc[k].axvspan(t[gi], t[gi + 1], color="orange", alpha=0.25)
        axes_acc[k].set_ylabel(f"Acc {lab} (mg)")
        axes_acc[k].grid(True, alpha=0.3)
    axes_acc[-1].set_xlabel("Time since first sample (s)")
    axes_acc[0].set_title(f"Raw accelerometer — {data_file.name}\n{gap_title}")
    fig_acc.tight_layout()

    # Plot 2: gyroscope. Decoder gives mdps; show in deg/s for readability.
    fig_gyr, axes_gyr = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for k, (lab, c) in enumerate(zip(labels, colors)):
        gyr_dps = np.asarray(data[f"imu_gyr_{lab.lower()}"]) / 1000.0
        axes_gyr[k].plot(t, gyr_dps, color=c, linewidth=0.6)
        for gi in gap_idx:
            axes_gyr[k].axvspan(t[gi], t[gi + 1], color="orange", alpha=0.25)
        axes_gyr[k].axhline(0, color="k", linewidth=0.5, alpha=0.3)
        axes_gyr[k].set_ylabel(f"Gyr {lab} (deg/s)")
        axes_gyr[k].grid(True, alpha=0.3)
    axes_gyr[-1].set_xlabel("Time since first sample (s)")
    axes_gyr[0].set_title(f"Raw gyroscope — {data_file.name}\n{gap_title}")
    fig_gyr.tight_layout()

    # Plot 3: integrated gyroscope (per-axis cumulative angle). Trapezoidal
    # integration of raw deg/s gives degrees; with no bias correction this is
    # the unfiltered "where did the gyro think it pointed". Slow drift here
    # is gyro bias; sharp ramps are real rotations.
    # X and Y are high-passed (cutoff HIGHPASS_HZ) to remove not just a
    # constant bias but also slowly varying bias drift. Z is left raw —
    # yaw has no gravity reference to anchor it.
    fig_int, axes_int = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    dt = np.diff(t)
    fs_int = 1.0 / max(float(np.median(dt)), 1e-6)
    sos_hp = butter(4, HIGHPASS_HZ, btype="highpass", fs=fs_int, output="sos")
    for k, (lab, c) in enumerate(zip(labels, colors)):
        gyr_dps = np.asarray(data[f"imu_gyr_{lab.lower()}"]) / 1000.0
        angle = np.concatenate((
            [0.0],
            np.cumsum(0.5 * (gyr_dps[:-1] + gyr_dps[1:]) * dt),
        ))
        if lab in ("X", "Y"):
            angle_hp = sosfiltfilt(sos_hp, angle)
            axes_int[k].plot(t, angle, color="0.7", linewidth=0.5, alpha=0.6,
                             label="raw integrated")
            axes_int[k].plot(t, angle_hp, color=c, linewidth=0.6,
                             label=f"high-pass > {HIGHPASS_HZ} Hz")
            axes_int[k].legend(loc="upper right", fontsize=8)
        else:
            axes_int[k].plot(t, angle, color=c, linewidth=0.6)
        for gi in gap_idx:
            axes_int[k].axvspan(t[gi], t[gi + 1], color="orange", alpha=0.25)
        axes_int[k].axhline(0, color="k", linewidth=0.5, alpha=0.3)
        axes_int[k].set_ylabel(f"Integrated Gyr {lab} (deg)")
        axes_int[k].grid(True, alpha=0.3)
    axes_int[-1].set_xlabel("Time since first sample (s)")
    axes_int[0].set_title(
        f"Integrated gyroscope (X/Y high-pass > {HIGHPASS_HZ} Hz) — {data_file.name}\n{gap_title}"
    )
    fig_int.tight_layout()

    # Plot 4: magnetometer (raw chip-frame µT). The chip→PCB axis mapping for
    # the mag (AK09916) is currently unverified — use this plot during the
    # orientation test (point each silkscreen mag-PCB axis to magnetic north
    # for ~10 s) to figure out which chip output channel responds.
    #
    # The |B| panel is the sanity check: total field magnitude should sit
    # roughly constant (~50 µT in mid-high latitudes; in Norway ~52 µT total,
    # ~14 µT horizontal + ~50 µT vertical due to ~74° inclination). A
    # wandering |B| means nearby ferrous metal is distorting the field.
    mag_x = np.asarray(data.get("imu_mag_x", []))
    mag_y = np.asarray(data.get("imu_mag_y", []))
    mag_z = np.asarray(data.get("imu_mag_z", []))
    has_mag = (
        mag_x.size > 0
        and not (np.all(mag_x == 0) and np.all(mag_y == 0) and np.all(mag_z == 0))
    )
    if has_mag:
        mag_total = np.sqrt(mag_x ** 2 + mag_y ** 2 + mag_z ** 2)
        fig_mag, axes_mag = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
        for k, (lab, c, arr) in enumerate(
            zip(labels, colors, (mag_x, mag_y, mag_z))
        ):
            axes_mag[k].plot(t, arr, color=c, linewidth=0.6)
            for gi in gap_idx:
                axes_mag[k].axvspan(t[gi], t[gi + 1], color="orange", alpha=0.25)
            axes_mag[k].axhline(0, color="k", linewidth=0.5, alpha=0.3)
            axes_mag[k].set_ylabel(f"Mag {lab} (µT)")
            axes_mag[k].grid(True, alpha=0.3)
        median_b = float(np.median(mag_total))
        axes_mag[3].plot(t, mag_total, color="tab:purple", linewidth=0.6)
        axes_mag[3].axhline(
            median_b, color="k", linewidth=0.5, linestyle="--", alpha=0.4,
            label=f"median |B| = {median_b:.1f} µT",
        )
        for gi in gap_idx:
            axes_mag[3].axvspan(t[gi], t[gi + 1], color="orange", alpha=0.25)
        axes_mag[3].set_ylabel("|B| (µT)")
        axes_mag[3].legend(loc="upper right", fontsize=8)
        axes_mag[3].grid(True, alpha=0.3)
        axes_mag[-1].set_xlabel("Time since first sample (s)")
        axes_mag[0].set_title(
            f"Raw magnetometer (chip frame) — {data_file.name}\n{gap_title}"
        )
        fig_mag.tight_layout()
        print(
            f"\nMag stats over recording:"
            f"\n  median |B|     = {median_b:5.1f} µT"
            f"\n  mx mean/range = {mag_x.mean():+6.1f} / [{mag_x.min():+6.1f}, {mag_x.max():+6.1f}] µT"
            f"\n  my mean/range = {mag_y.mean():+6.1f} / [{mag_y.min():+6.1f}, {mag_y.max():+6.1f}] µT"
            f"\n  mz mean/range = {mag_z.mean():+6.1f} / [{mag_z.min():+6.1f}, {mag_z.max():+6.1f}] µT"
        )
    else:
        print(
            "No magnetometer data in this recording "
            "(older firmware, or all-zero readings)."
        )

    plt.show()


if __name__ == "__main__":
    main()
