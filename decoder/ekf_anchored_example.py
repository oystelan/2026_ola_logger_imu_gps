#!/usr/bin/env python3
"""EKF with a synthetic position anchor — for GNSS-denied recordings.

Use this when you've recorded indoors / in a basin / under thick cover and
have no usable GPS, but the device's MEAN position is approximately fixed
(handheld test, moored buoy, stationary indoor sensor). The "static anchor"
tells the EKF "you're approximately at the origin with this much uncertainty"
once per second — which makes accel bias observable and stops position from
running away by kilometres during 145-second IMU-only dead reckoning.

Pipeline:
1. Decode the .dat file
2. Run the 15-state ESKF with static-anchor pseudo-measurements at 1 Hz
3. Plot vertical accel / velocity / displacement and attitude
4. Show horizontal position drift (bounded by the anchor sigma)

If your recording HAS valid GPS fixes, prefer `ekf_fusion_example.py` — it
gives you genuinely-georeferenced output. The anchor is a substitute, not an
upgrade, over real GNSS.

Usage:
    python ekf_anchored_example.py [path/to/file.dat]
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from decoder import decode_file, load_data_as_arrays
from sensor_fusion import EKFNoiseParams, bandpass_result, run_fusion


# --- Band of interest. Same convention as ahrs_example.py.
LOW_HZ = 0.05
HIGH_HZ = 2.5

# --- Static-anchor settings.
#
# ANCHOR_SIGMA_HORIZ_M / _VERT_M:
#   "Uncertainty" of the synthetic anchor measurement. Loose = lets real
#   motion through but still bounds drift; tight = strong anchor but
#   suppresses motion you actually want to measure. Rules of thumb:
#     - hand-test / stationary indoor:   1 m  (motion ≤ ±10 cm)
#     - wave buoy in mild waves:         5 m horiz, 2 m vert
#     - wave buoy in big swells (>2 m):  10 m horiz, 5 m vert
#
# ANCHOR_CADENCE_HZ:
#   How often to inject the anchor. 1 Hz mirrors real-GPS rate and is the
#   safe default. Lower (0.1 Hz) gives more bias observability per inject
#   but more between-inject drift. Higher (10 Hz) fights motion harder.
ANCHOR_SIGMA_HORIZ_M = 0.1
ANCHOR_SIGMA_VERT_M = 0.1
ANCHOR_CADENCE_HZ = 10.0

# Magnetometer yaw updates. KEEP DISABLED by default for typical anchored
# recordings — empirically (BOOT_22 et al.) the mag yaw updates fight the
# gyro propagation when the initial attitude was set from a non-stationary
# window. The filter then dumps the disagreement into the gyro-bias state,
# which runs away to 100+ deg/s and the vertical position estimate goes
# with it. Set True only if the recording starts with a clean ≥5 s
# stationary period AND you need absolute yaw (rare for wave-buoy work,
# where vertical motion is what matters and yaw is unobservable from
# accel anyway).
APPLY_MAG_UPDATES = False


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
            print("Usage: python ekf_anchored_example.py [path/to/file.dat]")
            return
        print(f"Auto-selected: {data_file.name}")

    if not data_file.exists():
        print(f"Not found: {data_file}")
        return

    # 1) Decode + load
    result = decode_file(data_file, allow_no_pps=True)
    data = load_data_as_arrays(result["file"])

    # 2) Run EKF with static-anchor pseudo-measurements
    fused = run_fusion(
        data,
        noise=EKFNoiseParams(),
        gps_horizontal_only=True,  # ignored if no GPS, harmless if some
        apply_mag_updates=APPLY_MAG_UPDATES,
        static_anchor_sigma_horiz_m=ANCHOR_SIGMA_HORIZ_M,
        static_anchor_sigma_vert_m=ANCHOR_SIGMA_VERT_M,
        static_anchor_cadence_hz=ANCHOR_CADENCE_HZ,
    )
    # Band-pass post-process for displacement around zero
    fused_bp = bandpass_result(fused, low_hz=LOW_HZ, high_hz=HIGH_HZ)

    # 3) Summary
    t = fused.t
    z_up_raw = -fused.p_ned[:, 2]                # "+up" convention
    vz_up_raw = -fused.v_ned[:, 2]
    z_up_bp = -fused_bp.p_ned[:, 2]
    vz_up_bp = -fused_bp.v_ned[:, 2]

    print("\n" + "=" * 60)
    print("ANCHORED-EKF SUMMARY")
    print("=" * 60)
    print(f"  Duration:                  {t[-1]:.1f} s")
    print(f"  IMU samples processed:     {len(t)}")
    print(f"  GPS updates folded:        {fused.n_gps_updates}")
    print(f"  Mag updates folded:        {fused.n_mag_updates}")
    print(f"  Static anchors injected:   {fused.n_static_anchors} "
          f"(σ_horiz={ANCHOR_SIGMA_HORIZ_M:.2f} m, σ_vert={ANCHOR_SIGMA_VERT_M:.2f} m)")
    print(f"  Final accel bias:          {fused.b_a[-1].round(3)} m/s²")
    print(f"  Final gyro bias:           {np.rad2deg(fused.b_g[-1]).round(3)} deg/s")
    print()
    print(f"  Raw vertical position range: [{z_up_raw.min():+.2f}, {z_up_raw.max():+.2f}] m")
    print(f"  Bandpass vertical disp std:  {z_up_bp.std()*1000:.1f} mm "
          f"(peak ±{np.max(np.abs(z_up_bp))*1000:.1f} mm)")
    print(f"  Horizontal drift range:      N=[{fused.p_ned[:,0].min():+.2f}, "
          f"{fused.p_ned[:,0].max():+.2f}] m, "
          f"E=[{fused.p_ned[:,1].min():+.2f}, {fused.p_ned[:,1].max():+.2f}] m")

    # ===== Plots =====

    # Plot 1: Vertical position (raw EKF + band-passed)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, z_up_raw * 1000, color="0.7", linewidth=0.7,
            label=f"raw EKF (anchored ±{ANCHOR_SIGMA_VERT_M:.1f} m)")
    ax.plot(t, z_up_bp * 1000, "b-", linewidth=0.8,
            label=f"band-pass [{LOW_HZ}, {HIGH_HZ}] Hz")
    s = z_up_bp.std() * 1000
    ax.axhline(+s, color="g", linewidth=0.5, linestyle="--", alpha=0.5, label="±1σ")
    ax.axhline(-s, color="g", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Vertical displacement, +up (mm)")
    ax.set_title(f"Anchored-EKF vertical displacement — {data_file.name}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    print("  ok  vertical displacement plot")

    # Plot 2: Vertical velocity
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, vz_up_raw * 1000, color="0.7", linewidth=0.5,
            label="raw EKF velocity")
    ax.plot(t, vz_up_bp * 1000, "b-", linewidth=0.6,
            label=f"band-pass [{LOW_HZ}, {HIGH_HZ}] Hz")
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Vertical velocity, +up (mm/s)")
    ax.set_title("Anchored-EKF vertical velocity")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    print("  ok  vertical velocity plot")

    # Plot 3: Attitude — roll, pitch, and tilt magnitude
    eul = fused.euler_zyx                    # [yaw, pitch, roll]
    yaw_deg = np.rad2deg(eul[:, 0])
    pitch_deg = np.rad2deg(eul[:, 1])
    roll_deg = np.rad2deg(eul[:, 2])
    tilt_deg = np.rad2deg(np.arccos(
        np.clip(np.cos(eul[:, 1]) * np.cos(eul[:, 2]), -1, 1)
    ))
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(t, roll_deg, "b-", linewidth=0.5, label="Roll", alpha=0.7)
    axes[0].plot(t, pitch_deg, "r-", linewidth=0.5, label="Pitch", alpha=0.7)
    axes[0].set_ylabel("Roll / Pitch (deg)")
    axes[0].set_title("EKF attitude (Euler ZYX — flips by 180° near pitch=±90°)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, tilt_deg, "g-", linewidth=0.5)
    axes[1].axhline(0, color="k", linewidth=0.5, alpha=0.3)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Tilt from level (deg)")
    axes[1].set_title("Tilt magnitude = angle(body z, world z) — singularity-free")
    axes[1].grid(True, alpha=0.3)
    print("  ok  attitude plot (roll/pitch + tilt)")

    # Plot 4: Horizontal position drift — the anchor is what keeps this bounded
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t, fused.p_ned[:, 0], "b-", linewidth=0.6, label="North (m)")
    axes[0].fill_between(
        t, -ANCHOR_SIGMA_HORIZ_M, +ANCHOR_SIGMA_HORIZ_M,
        color="g", alpha=0.10, label=f"±anchor σ ({ANCHOR_SIGMA_HORIZ_M} m)",
    )
    axes[0].axhline(0, color="k", linewidth=0.5, alpha=0.3)
    axes[0].set_ylabel("North (m)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, fused.p_ned[:, 1], "r-", linewidth=0.6, label="East (m)")
    axes[1].fill_between(
        t, -ANCHOR_SIGMA_HORIZ_M, +ANCHOR_SIGMA_HORIZ_M,
        color="g", alpha=0.10,
    )
    axes[1].axhline(0, color="k", linewidth=0.5, alpha=0.3)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("East (m)")
    axes[1].grid(True, alpha=0.3)
    axes[0].set_title("Horizontal position (NED) — the anchor keeps this bounded")
    print("  ok  horizontal-position plot")

    # Plot 5: Bias estimates over time — should converge with the anchor on
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for k, label in enumerate(("x", "y", "z")):
        axes[0].plot(t, fused.b_a[:, k], linewidth=0.7, label=f"b_a {label}")
    axes[0].set_ylabel("Accel bias estimate (m/s²)")
    axes[0].set_title("EKF bias states — should converge when anchor is on")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    for k, label in enumerate(("x", "y", "z")):
        axes[1].plot(t, np.rad2deg(fused.b_g[:, k]), linewidth=0.7, label=f"b_g {label}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Gyro bias estimate (deg/s)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    print("  ok  bias-convergence plot")

    print("\nDisplaying plots. Close all windows to exit.")
    plt.show()
    print("Done.")


if __name__ == "__main__":
    main()
