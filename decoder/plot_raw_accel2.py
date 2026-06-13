#!/usr/bin/env python3
"""Plot raw IMU signals (accel, gyro, magnetometer) from a .dat in 3 overlaid
panels — one panel per sensor, all three axes overlaid in each panel.

Same data and gap-marking as plot_raw_accel.py, just a more compact layout:
  Panel 1: accel X+Y+Z  (mg)
  Panel 2: gyro  X+Y+Z  (deg/s)
  Panel 3: mag   X+Y+Z  (uT)   + |B| total field

Usage:
    python plot_raw_accel2.py [path/to/file.dat]

If no path is given, picks the most recently modified DATA_BOOT_*.dat in
the script directory.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from decoder import decode_file, load_data_as_arrays


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
            print("Usage: python plot_raw_accel2.py [path/to/file.dat]")
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

    # If the load produced cleaned variants, overlay them on top of the raw
    # in a thicker line so the eye can compare raw vs cleaned signal.
    has_clean = "imu_acc_x_clean" in data

    # Build one figure with 3 panels stacked, x-axis shared
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    # Panel 1: accelerometer (mg) — all three axes overlaid
    for lab, c in zip(labels, colors):
        axes[0].plot(t, data[f"imu_acc_{lab.lower()}"], color=c, linewidth=0.6,
                     alpha=0.45 if has_clean else 1.0,
                     label=f"acc {lab}" + (" raw" if has_clean else ""))
    if has_clean:
        for lab, c in zip(labels, colors):
            axes[0].plot(t, data[f"imu_acc_{lab.lower()}_clean"], color=c, linewidth=0.7,
                         label=f"acc {lab} clean")
    for gi in gap_idx:
        axes[0].axvspan(t[gi], t[gi + 1], color="orange", alpha=0.25)
    axes[0].set_ylabel("Acc (mg)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8, ncol=3)
    axes[0].set_title(f"Raw IMU — {data_file.name}\n{gap_title}")

    # Panel 2: gyroscope. Decoder gives mdps; show in deg/s for readability.
    for lab, c in zip(labels, colors):
        gyr_dps = np.asarray(data[f"imu_gyr_{lab.lower()}"]) / 1000.0
        axes[1].plot(t, gyr_dps, color=c, linewidth=0.6,
                     alpha=0.45 if has_clean else 1.0,
                     label=f"gyr {lab}" + (" raw" if has_clean else ""))
    if has_clean:
        for lab, c in zip(labels, colors):
            gyr_dps_clean = np.asarray(data[f"imu_gyr_{lab.lower()}_clean"]) / 1000.0
            axes[1].plot(t, gyr_dps_clean, color=c, linewidth=0.7,
                         label=f"gyr {lab} clean")
    for gi in gap_idx:
        axes[1].axvspan(t[gi], t[gi + 1], color="orange", alpha=0.25)
    axes[1].axhline(0, color="k", linewidth=0.5, alpha=0.3)
    axes[1].set_ylabel("Gyr (deg/s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8, ncol=3)

    # Panel 3: magnetometer (raw chip-frame uT) + |B| total field. The chip→PCB
    # mag axis mapping is currently unverified, so values are shown in the
    # chip's native frame. |B| is the sanity check: total field magnitude
    # should sit roughly constant (~52 uT total in Norway with ~74° inclination).
    mag_x = np.asarray(data.get("imu_mag_x", []))
    mag_y = np.asarray(data.get("imu_mag_y", []))
    mag_z = np.asarray(data.get("imu_mag_z", []))
    has_mag = (
        mag_x.size > 0
        and not (np.all(mag_x == 0) and np.all(mag_y == 0) and np.all(mag_z == 0))
    )
    if has_mag:
        for lab, c, arr in zip(labels, colors, (mag_x, mag_y, mag_z)):
            axes[2].plot(t, arr, color=c, linewidth=0.6,
                         alpha=0.45 if has_clean else 1.0,
                         label=f"mag {lab}" + (" raw" if has_clean else ""))
        if has_clean:
            for lab, c in zip(labels, colors):
                axes[2].plot(t, data[f"imu_mag_{lab.lower()}_clean"], color=c, linewidth=0.7,
                             label=f"mag {lab} clean")
        mag_total = np.sqrt(mag_x ** 2 + mag_y ** 2 + mag_z ** 2)
        axes[2].plot(t, mag_total, color="tab:purple", linewidth=0.6, alpha=0.8, label="|B|")
        median_b = float(np.median(mag_total))
        axes[2].axhline(
            median_b, color="k", linewidth=0.5, linestyle="--", alpha=0.4,
            label=f"med |B|={median_b:.1f} uT",
        )
        for gi in gap_idx:
            axes[2].axvspan(t[gi], t[gi + 1], color="orange", alpha=0.25)
        axes[2].axhline(0, color="k", linewidth=0.5, alpha=0.3)
        axes[2].set_ylabel("Mag (uT)")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="upper right", fontsize=8, ncol=5)
        print(
            f"\nMag stats over recording:"
            f"\n  median |B|     = {median_b:5.1f} uT"
            f"\n  mx mean/range = {mag_x.mean():+6.1f} / [{mag_x.min():+6.1f}, {mag_x.max():+6.1f}] uT"
            f"\n  my mean/range = {mag_y.mean():+6.1f} / [{mag_y.min():+6.1f}, {mag_y.max():+6.1f}] uT"
            f"\n  mz mean/range = {mag_z.mean():+6.1f} / [{mag_z.min():+6.1f}, {mag_z.max():+6.1f}] uT"
        )
    else:
        axes[2].text(0.5, 0.5, "No magnetometer data in this recording",
                     ha="center", va="center", transform=axes[2].transAxes,
                     fontsize=11, color="gray")
        axes[2].set_ylabel("Mag (uT)")
        axes[2].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time since first sample (s)")
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
