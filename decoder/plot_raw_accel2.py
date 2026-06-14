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

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from decoder import decode_file, load_data_as_arrays


# Drop this many seconds of IMU data from the start of the recording before
# plotting. The DMP firmware emits warmup garbage for ~1-2 s after boot which
# blows out the autoscaled y-axis on every panel. Set to 0 to disable.
SKIP_START_S = 1.5

# True = plot the x-axis as wall-clock UTC (datetime ticks). False = plot it
# as "elapsed seconds since first sample" (numeric). UTC needs the recording
# to have either PPS-regressed imu_utc, GNSS posix fixes, or a parseable
# filename timestamp; falls back to elapsed seconds and prints a notice if
# none of those are available.
USE_UTC = True


def _filename_timestamp_to_posix(name: str) -> float | None:
    """Pull POSIX seconds from a DATA_BOOT_NNNNNN_TIME_YYYYMMDDTHHMMSS.dat
    filename. UTC-interpreted. Returns None if the pattern doesn't match.
    """
    m = re.search(r"_TIME_(\d{8})T(\d{6})", name)
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        return dt.replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def _build_time_axis(data: dict, source_filename: str | None) -> tuple[np.ndarray, str, bool]:
    """Return (t, x_label, used_utc). Three-tier UTC fallback (mirrors
    plot_raw_accel.py): PPS-regressed imu_utc -> linear fit on GNSS posix ->
    filename anchor. Falls back to elapsed seconds if none of those work or
    if USE_UTC is False.
    """
    imu_micros = np.asarray(data["imu_micros_unwrapped"], dtype=np.float64)
    if USE_UTC and imu_micros.size > 0:
        # Path 1: PPS-regressed imu_utc
        utc_raw = np.asarray(data.get("imu_utc", []), dtype=np.float64)
        if utc_raw.size and np.any(np.isfinite(utc_raw)):
            return (utc_raw * 1e9).astype("datetime64[ns]"), "UTC", True

        # Path 2: linear fit gnss_micros_unwrapped -> gnss_posix
        g_micros = np.asarray(data.get("gnss_micros_unwrapped", []), dtype=np.float64)
        g_posix = np.asarray(data.get("gnss_posix", []), dtype=np.float64)
        if g_micros.size >= 2 and g_posix.size == g_micros.size:
            valid = g_posix > 1e9   # plausible POSIX after year 2001
            if valid.sum() >= 2:
                slope, intercept = np.polyfit(g_micros[valid], g_posix[valid], 1)
                est = slope * imu_micros + intercept
                return (est * 1e9).astype("datetime64[ns]"), "UTC (from GNSS posix fit)", True

        # Path 3: filename timestamp + elapsed micros
        anchor = _filename_timestamp_to_posix(source_filename or "")
        if anchor is not None:
            elapsed = (imu_micros - imu_micros[0]) * 1e-6
            return (
                (np.int64(anchor * 1e9) + (elapsed * 1e9).astype(np.int64))
                    .astype("datetime64[ns]"),
                "UTC (from filename anchor)",
                True,
            )

        print("USE_UTC=True but no usable UTC source found; falling back to elapsed seconds.")

    t = (imu_micros - imu_micros[0]) * 1e-6
    return t, "Time since first sample (s)", False


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

    # Drop the DMP warmup region (see SKIP_START_S at the top of this file).
    if SKIP_START_S > 0:
        imu_us_full = np.asarray(data["imu_micros_unwrapped"], dtype=np.float64)
        elapsed_full = (imu_us_full - imu_us_full[0]) * 1e-6
        n_drop = int(np.searchsorted(elapsed_full, SKIP_START_S, side="left"))
        # Don't strip the entire recording — leave at least one sample.
        n_drop = max(0, min(n_drop, len(imu_us_full) - 1))
        if n_drop > 0:
            n_imu = len(imu_us_full)
            for k in list(data.keys()):
                v = data[k]
                if not k.startswith("imu_"):
                    continue
                if not isinstance(v, np.ndarray) or v.size != n_imu:
                    continue
                data[k] = v[n_drop:]
            print(f"Dropped first {n_drop} IMU samples ({SKIP_START_S:.1f} s) "
                  f"of warmup / startup transient.")

    # Build the x-axis (UTC datetimes if USE_UTC and a UTC source is
    # available, else elapsed seconds — see _build_time_axis).
    t, t_label, used_utc = _build_time_axis(data, source_filename=data_file.name)

    # Highlight any gaps > 2× the median sample interval. Whether t is
    # elapsed-seconds or datetime64, np.diff gives the right type.
    if used_utc:
        dts = np.diff(t).astype("timedelta64[ns]").astype(np.float64) * 1e-9
    else:
        dts = np.diff(t)
    dt_med = float(np.median(dts))
    gap_threshold = max(2 * dt_med, 0.020)  # 20 ms floor
    gap_idx = np.where(dts > gap_threshold)[0]

    duration = (t[-1] - t[0])
    duration_s = (duration.astype("timedelta64[s]").astype(np.float64)
                  if used_utc else float(duration))
    print(f"\nSamples:           {len(t)}")
    print(f"Duration:          {duration_s:.1f} s")
    print(f"Median dt:         {dt_med*1000:.3f} ms (~{1/dt_med:.1f} Hz)")
    print(f"Max dt:            {dts.max()*1000:.1f} ms")
    print(f"Gaps > {gap_threshold*1000:.0f} ms:    {len(gap_idx)}")
    if len(gap_idx) > 0:
        worst = t[gap_idx[np.argmax(dts[gap_idx])]]
        worst_str = str(worst) if used_utc else f"{float(worst):.1f} s"
        print(f"  Worst gap: {dts[gap_idx].max()*1000:.1f} ms at t={worst_str}")

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

    axes[-1].set_xlabel(t_label)
    if used_utc:
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=None))
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
