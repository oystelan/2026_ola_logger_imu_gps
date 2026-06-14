#!/usr/bin/env python3
"""AHRS-based vertical-position estimation example.

Use this script for GPS-denied recordings (indoor basin, swimming pool,
shielded environments) where you care primarily about the oscillatory
vertical motion in a known frequency band.

Pipeline:
1. Decode the .dat file
2. Run Madgwick AHRS on accel + gyro to track attitude
3. Rotate body-frame accel into world frame, subtract gravity
4. Frequency-domain double-integrate the band-passed vertical accel
   → vertical displacement around zero

X/Y drift is irrelevant for this workflow — only vertical AC motion is
preserved. If you need horizontal position too, use ekf_fusion_example.py.

Usage:
    python ahrs_example.py [path/to/file.dat]
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from ahrs_vertical import (
    compute_vertical_motion,
    compute_vertical_motion_lowpass_gravity,
    compute_vertical_motion_savgol_detrend,
)
from plot_raw_accel import find_default_path, load_data_from_path, _clip_initial_seconds


# Frequency band of motion-of-interest. Tune to your basin / wave setup.
LOW_HZ = 0.1
HIGH_HZ = 5.

# Drop this many seconds from the start of the recording before running the
# AHRS. The DMP firmware emits warmup garbage for ~1-2 s after boot; feeding
# that into the attitude bootstrap / integration corrupts the whole result.
SKIP_START_S = 1.5

# Seconds at the start of the (post-skip) recording used to determine which
# body axis gravity is on, so we can rotate it onto +Z before the AHRS runs
# (avoids the Euler gimbal-lock singularity at pitch=±90°). Keep the device
# roughly still for this window.
GRAVITY_DETECT_S = 2.0

# Method for getting body→world attitude / gravity reference:
#   "madgwick" — classic AHRS quaternion with motion-gated accel correction.
#                Good when there are stationary moments; weak under sustained
#                rotation+translation (lever-arm centripetal biases accel
#                direction, Madgwick "corrects" toward the wrong direction).
#   "lowpass"  — complementary filter: gyro handles fast attitude tracking,
#                low-pass accel provides the slow gravity reference that kills
#                gyro drift. Above GRAVITY_CUTOFF_HZ, attitude follows gyro;
#                below it, attitude follows low-pass accel. Robust to combined
#                rotation+translation (wave buoys, basins). Recording length
#                should exceed ~3/GRAVITY_CUTOFF_HZ for clean edges.
#   "savgol"   — gyro AHRS with Savitzky-Golay detrend on the attitude. Assumes
#                the device's mean orientation is level; any slow trend in the
#                gyro-integrated Euler angles is treated as drift to subtract.
#                Effective high-pass at ~1/SAVGOL_ATTITUDE_WINDOW_S Hz on the
#                attitude. Robust to lever-arm centripetal (accel never used
#                for attitude estimation).
METHOD = "madgwick"
GRAVITY_CUTOFF_HZ = 0.05  # only used when METHOD == "lowpass"
SAVGOL_ATTITUDE_WINDOW_S = 10.0  # only used when METHOD == "savgol"
SAVGOL_ATTITUDE_POLYORDER = 3    # only used when METHOD == "savgol"

# Savitzky-Golay smoothing window for the raw-accel plot. Window length
# in seconds → samples is computed at plot time from the result's fs.
# 0.5 s + cubic polynomial is a sensible default for ~1 Hz wave motion;
# tighten the window to keep more high-frequency detail.
SAVGOL_WINDOW_S = 0.5
SAVGOL_POLYORDER = 3


def main():
    # Default to the BOOT_*/ folder with the highest boot number (same picker
    # as plot_raw_accel.py); an explicit path argument overrides it and may be
    # either a single .dat file or a BOOT_*/ folder.
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = find_default_path()
        if path is None:
            print("No data file or BOOT_*/ folder given and none found.")
            print("Usage: python ahrs_example.py [path/to/file.dat OR path/to/BOOT_NNNNNN]")
            return
        kind = "folder" if path.is_dir() else "file"
        print(f"Auto-selected newest {kind}: {path.name}")

    if not path.exists():
        print(f"Not found: {path}")
        return

    # 1) Decode + load. load_data_from_path handles both a single .dat and a
    #    whole BOOT_*/ folder (decoding every file and splicing them into one
    #    contiguous timeseries).
    data, source_files, _file_starts = load_data_from_path(path)
    data_name = path.name if path.is_dir() else source_files[0].name

    # 1a) Drop the DMP boot-warmup region so it doesn't corrupt the attitude
    #     bootstrap and the integration. _file_starts is recomputed but unused
    #     downstream here.
    data, _file_starts, n_dropped = _clip_initial_seconds(data, _file_starts, SKIP_START_S)
    if n_dropped > 0:
        print(f"Skipped first {n_dropped} IMU samples ({SKIP_START_S:.1f} s) of warmup.")

    # 1b) Gravity auto-align happens INSIDE the compute_* functions now
    #     (auto_align_gravity=True by default) — it detects the dominant
    #     gravity axis and rotates it onto body -Z so the AHRS works
    #     regardless of mounting orientation, avoiding the Euler gimbal-lock
    #     singularity. We just pass the detect window through.

    # 2) AHRS / gravity-tracking + frequency-domain double integration
    if METHOD == "lowpass":
        vmot = compute_vertical_motion_lowpass_gravity(
            data,
            low_hz=LOW_HZ,
            high_hz=HIGH_HZ,
            gravity_cutoff_hz=GRAVITY_CUTOFF_HZ,
            gravity_detect_seconds=GRAVITY_DETECT_S,
        )
    elif METHOD == "madgwick":
        vmot = compute_vertical_motion(
            data,
            low_hz=LOW_HZ,
            high_hz=HIGH_HZ,
            motion_gate_threshold=GRAVITY_CUTOFF_HZ,
            gravity_detect_seconds=GRAVITY_DETECT_S,
        )
    elif METHOD == "savgol":
        vmot = compute_vertical_motion_savgol_detrend(
            data,
            low_hz=LOW_HZ,
            high_hz=HIGH_HZ,
            gravity_detect_seconds=GRAVITY_DETECT_S,
            savgol_window_seconds=SAVGOL_ATTITUDE_WINDOW_S,
            savgol_polyorder=SAVGOL_ATTITUDE_POLYORDER,
        )
    else:
        raise ValueError(
            f"Unknown METHOD={METHOD!r}; expected 'lowpass', 'madgwick', or 'savgol'"
        )

    # 3) Summary
    print("\n" + "=" * 60)
    print("VERTICAL MOTION SUMMARY")
    print("=" * 60)
    print(f"  Resampled to:           {vmot.fs_hz:.1f} Hz")
    print(f"  Band of interest:       [{vmot.low_hz}, {vmot.high_hz}] Hz")
    print(f"  Duration:               {vmot.t[-1]:.1f} s")
    print(f"  RMS vertical accel:     {np.sqrt(np.mean(vmot.accel_z_up**2)):.3f} m/s^2")
    print(f"  RMS vertical velocity:  {np.sqrt(np.mean(vmot.velocity_z_up**2))*1000:.1f} mm/s")
    print(f"  Peak displacement:      +/- {np.max(np.abs(vmot.displacement_z_up))*1000:.1f} mm")
    print(f"  std of displacement:    {np.std(vmot.displacement_z_up)*1000:.1f} mm")
    print(f"  4*std (crude Hs):       {vmot.hs_band*1000:.1f} mm")

    # 4) Plots
    t = vmot.t

    # Plot 1: vertical acceleration. Three traces:
    #   - raw, mean-subtracted (background grey): residual after gravity
    #     removal, with any DC offset taken out so the curve sits around 0
    #   - Savitzky-Golay smoothed raw (orange): low-pass-ish view that keeps
    #     polynomial shape but suppresses high-frequency sensor noise
    #   - band-pass [LOW_HZ, HIGH_HZ] (blue): the signal that actually goes
    #     through to integration
    raw_demeaned = vmot.accel_z_up_raw - vmot.accel_z_up_raw.mean()
    # Savitzky-Golay window in samples (must be odd, and < signal length)
    sg_win = int(round(SAVGOL_WINDOW_S * vmot.fs_hz))
    if sg_win % 2 == 0:
        sg_win += 1
    sg_win = max(SAVGOL_POLYORDER + 2, min(sg_win, len(raw_demeaned) - 1))
    if sg_win % 2 == 0:
        sg_win -= 1
    raw_savgol = savgol_filter(raw_demeaned, sg_win, SAVGOL_POLYORDER)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, raw_demeaned, color="0.7", linewidth=0.5, alpha=0.7,
            label=f"raw - mean ({vmot.accel_z_up_raw.mean():+.3f} m/s² removed)")
    ax.plot(t, raw_savgol, color="tab:orange", linewidth=0.8,
            label=f"Savitzky-Golay (window={sg_win} samples ≈ {sg_win/vmot.fs_hz:.2f}s, order {SAVGOL_POLYORDER})")
    ax.plot(t, vmot.accel_z_up, "b-", linewidth=0.6,
            label=f"band-pass [{LOW_HZ}, {HIGH_HZ}] Hz")
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Vertical accel, +up (m/s^2)")
    ax.set_title("World-frame vertical acceleration")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    print("  ok  vertical accel plot (raw-demeaned + savgol + band-pass)")

    # Plot 2: vertical velocity
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, vmot.velocity_z_up * 1000, "b-", linewidth=0.6)
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Vertical velocity, +up (mm/s)")
    ax.set_title(f"Vertical velocity — {data_name}")
    ax.grid(True, alpha=0.3)
    print("  ok  vertical velocity plot")

    # Plot 3: vertical displacement — the main answer
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, vmot.displacement_z_up * 1000, "b-", linewidth=0.8)
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
    # Reference bands at ±1σ and ±4σ
    s = np.std(vmot.displacement_z_up) * 1000
    ax.axhline(+s, color="g", linewidth=0.5, linestyle="--", alpha=0.5, label="±1σ")
    ax.axhline(-s, color="g", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Vertical displacement, +up (mm)")
    ax.set_title(f"Vertical displacement (band-pass [{LOW_HZ}, {HIGH_HZ}] Hz) — {data_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    print("  ok  vertical displacement plot (main result)")

    # Plot 4: attitude diagnostics — roll, pitch (Euler) AND tilt magnitude.
    # The Euler roll/pitch traces are convenient but have a representation
    # singularity at pitch ≈ ±90°: at that point the roll axis aligns with
    # the yaw axis, so any noise snaps the decomposition between +180° and
    # -180° roll, even though the underlying quaternion is perfectly smooth.
    # The bottom panel shows tilt = angle(body-z, world-z), computed from the
    # quaternion as arccos(R[2,2]) = arccos(cos(pitch)·cos(roll)). It has no
    # singularity until the device is fully upside-down (180°), so a "spike"
    # that appears in roll but NOT in tilt is just an Euler artifact, not a
    # real physical event.
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(t, vmot.roll_deg, "b-", linewidth=0.5, label="Roll", alpha=0.7)
    axes[0].plot(t, vmot.pitch_deg, "r-", linewidth=0.5, label="Pitch", alpha=0.7)
    axes[0].set_ylabel("Roll / Pitch (deg)")
    axes[0].set_title("AHRS roll & pitch (Euler — flips by 180° near pitch=±90°)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    tilt_deg = np.rad2deg(np.arccos(
        np.cos(np.deg2rad(vmot.roll_deg)) * np.cos(np.deg2rad(vmot.pitch_deg))
    ))
    axes[1].plot(t, tilt_deg, "g-", linewidth=0.5)
    axes[1].axhline(0, color="k", linewidth=0.5, alpha=0.3)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Tilt from level (deg)")
    axes[1].set_title("Tilt magnitude = angle(body z, world z) — singularity-free")
    axes[1].grid(True, alpha=0.3)
    print("  ok  attitude diagnostics plot (roll/pitch + tilt)")

    # Plot 5: amplitude spectrum of vertical acceleration — verify band & peaks
    n = len(vmot.accel_z_up)
    Y = np.abs(np.fft.rfft(vmot.accel_z_up)) * 2 / n
    f = np.fft.rfftfreq(n, d=1 / vmot.fs_hz)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.loglog(f[1:], Y[1:], "b-", linewidth=0.6)
    ax.axvspan(LOW_HZ, HIGH_HZ, color="g", alpha=0.1, label=f"Band [{LOW_HZ}, {HIGH_HZ}] Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|A(f)| (m/s²)")
    ax.set_title("Vertical acceleration spectrum")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    print("  ok  spectrum plot")

    print("\nDisplaying plots. Close all windows to exit.")
    plt.show()
    print("Done.")


if __name__ == "__main__":
    main()
