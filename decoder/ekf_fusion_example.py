#!/usr/bin/env python3
"""End-to-end sensor fusion example.

Pipeline:
1. Decode a .dat recording → .npz
2. Load as numpy arrays
3. Run the 15-state ESKF (IMU + magnetometer + GNSS)
4. Plot fused position, velocity, attitude, and altitude with sigma bands

Usage:
    python ekf_fusion_example.py [path/to/file.dat]

Indoor / no-GPS recordings still run — the filter degrades gracefully
to IMU-only dead reckoning and position/velocity drift visibly. That's
expected; the script prints a warning when this happens.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from decoder import decode_file, load_data_as_arrays
from sensor_fusion import EKFNoiseParams, bandpass_result, run_fusion

# Frequency band of motion-of-interest. Tune to your application —
# 0.05-2.5 Hz is a sensible default for wave / human-scale motion.
BANDPASS_LOW_HZ = 0.05
BANDPASS_HIGH_HZ = 2.5


def find_default_data_file() -> Path | None:
    here = Path(__file__).parent
    candidates = sorted(here.glob("DATA_BOOT_*.dat"))
    return candidates[0] if candidates else None


def main():
    if len(sys.argv) > 1:
        data_file = Path(sys.argv[1])
    else:
        data_file = find_default_data_file()
        if data_file is None:
            print("No data file given and no DATA_BOOT_*.dat found.")
            print("Usage: python ekf_fusion_example.py [path/to/file.dat]")
            return
        print(f"Auto-selected: {data_file.name}")

    if not data_file.exists():
        print(f"Not found: {data_file}")
        return

    # 1) Decode + load
    result = decode_file(data_file, allow_no_pps=True)
    data = load_data_as_arrays(result["file"])

    # 2) Run EKF. Defaults are tuned for ICM-20948 + MAX-M10S; pass a custom
    # EKFNoiseParams() to tune (e.g. tighter GPS sigmas for RTK).
    # gps_horizontal_only=True drops GPS altitude / vel_down from the
    # measurement updates — recommended for indoor / weak-signal recordings
    # where GPS vertical is dominated by multipath. Set False outdoors with
    # clear sky view.
    fused = run_fusion(data, noise=EKFNoiseParams(), gps_horizontal_only=True)

    print("\n" + "=" * 60)
    print(f"Fusion summary")
    print("=" * 60)
    print(f"  Samples processed:  {len(fused.t)}")
    print(f"  GPS updates folded: {fused.n_gps_updates}")
    print(f"  Mag updates folded: {fused.n_mag_updates}")
    if fused.n_gps_updates == 0:
        print(f"  WARNING: No GPS updates — position is pure IMU dead-reckoning "
              f"and will drift heavily.")
    print(f"  Final accel bias: {fused.b_a[-1]} m/s²")
    print(f"  Final gyro bias:  {np.rad2deg(fused.b_g[-1])} deg/s")

    # 3) Plot
    t = fused.t

    # --- Position (NED) with sigma bands
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for k, label in enumerate(("North", "East", "Down")):
        axes[k].plot(t, fused.p_ned[:, k], "b-", linewidth=1, label=f"{label} (m)")
        axes[k].fill_between(
            t,
            fused.p_ned[:, k] - fused.sigma_p[:, k],
            fused.p_ned[:, k] + fused.sigma_p[:, k],
            color="b", alpha=0.15, label="±1σ",
        )
        axes[k].set_ylabel(f"{label} (m)")
        axes[k].grid(True, alpha=0.3)
        axes[k].legend(loc="upper right")
    axes[-1].set_xlabel("Time since start (s)")
    fig.suptitle(f"EKF position (NED) — {data_file.name}")
    fig.tight_layout()
    print("  ok  position NED plot")

    # --- Altitude (the answer to the original z-question)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, fused.altitude_m, "b-", linewidth=1, label="EKF altitude")
    if fused.n_gps_updates > 0 and "gnss_altitude_msl" in data and len(data["gnss_altitude_msl"]) > 0:
        # Overlay raw GPS altitude for visual comparison
        if "gnss_micros_unwrapped" in data and len(data["gnss_micros_unwrapped"]) > 0:
            imu_us = np.asarray(data["imu_micros_unwrapped"], dtype=np.float64)
            g_t = (data["gnss_micros_unwrapped"] - imu_us[0]) * 1e-6
            ax.plot(g_t, data["gnss_altitude_msl"], "g.", markersize=3,
                    alpha=0.5, label="raw GPS altitude")
    ax.set_xlabel("Time since start (s)")
    ax.set_ylabel("Altitude MSL (m)")
    ax.set_title(f"Fused vertical position — {data_file.name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    print("  ok  altitude plot")

    # --- Band-pass-filtered vertical displacement
    # Strip the slow drift / mean and the high-frequency spikes. What's left
    # is the wave-band / motion-of-interest vertical signal.
    try:
        fused_bp = bandpass_result(fused, BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(t, fused_bp.altitude_m, "b-", linewidth=1,
                label=f"Band-pass [{BANDPASS_LOW_HZ}, {BANDPASS_HIGH_HZ}] Hz")
        ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
        ax.set_xlabel("Time since start (s)")
        ax.set_ylabel("Vertical displacement (m)")
        ax.set_title(f"Band-pass-filtered vertical motion — {data_file.name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        print(f"  ok  band-pass altitude plot ([{BANDPASS_LOW_HZ}, {BANDPASS_HIGH_HZ}] Hz)")
    except Exception as e:
        print(f"  skip  band-pass plot: {e}")

    # --- Velocity
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for k, label in enumerate(("vN", "vE", "vD")):
        axes[k].plot(t, fused.v_ned[:, k], "b-", linewidth=1, label=f"EKF {label}")
        if fused.n_gps_updates > 0:
            imu_us = np.asarray(data["imu_micros_unwrapped"], dtype=np.float64)
            g_t = (data["gnss_micros_unwrapped"] - imu_us[0]) * 1e-6
            v_raw = {
                "vN": data["gnss_vel_north"] * 1e-3,
                "vE": data["gnss_vel_east"] * 1e-3,
                "vD": data["gnss_vel_down"] * 1e-3,
            }[label]
            axes[k].plot(g_t, v_raw, "g.", markersize=3, alpha=0.5, label="raw GPS")
        axes[k].set_ylabel(f"{label} (m/s)")
        axes[k].grid(True, alpha=0.3)
        axes[k].legend(loc="upper right")
    axes[-1].set_xlabel("Time since start (s)")
    fig.suptitle(f"EKF velocity (NED) — {data_file.name}")
    fig.tight_layout()
    print("  ok  velocity plot")

    # --- Attitude (Z-Y-X Euler: yaw, pitch, roll)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    labels = ("Yaw", "Pitch", "Roll")
    for k, label in enumerate(labels):
        axes[k].plot(t, np.rad2deg(fused.euler_zyx[:, k]), "b-", linewidth=1)
        axes[k].set_ylabel(f"{label} (deg)")
        axes[k].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time since start (s)")
    fig.suptitle(f"EKF attitude (Z-Y-X intrinsic Euler) — {data_file.name}")
    fig.tight_layout()
    print("  ok  attitude plot")

    # --- Bias trajectories — useful for sanity-checking convergence
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t, fused.b_a)
    axes[0].set_ylabel("Accel bias (m/s²)")
    axes[0].legend(["x", "y", "z"])
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, np.rad2deg(fused.b_g))
    axes[1].set_ylabel("Gyro bias (deg/s)")
    axes[1].set_xlabel("Time since start (s)")
    axes[1].legend(["x", "y", "z"])
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"EKF bias estimates — {data_file.name}")
    fig.tight_layout()
    print("  ok  bias plot")

    print("\nDisplaying plots. Close all windows to exit.")
    plt.show()
    print("Done.")


if __name__ == "__main__":
    main()
