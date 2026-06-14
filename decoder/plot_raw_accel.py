#!/usr/bin/env python3
"""Plot raw IMU signals (accel, gyro, integrated-gyro, magnetometer) from a
single .dat file OR a whole BOOT_NNNNNN folder of .dat files.

Four diagnostic figures:
  1. Accelerometer — mg per axis. Gaps in the sample stream highlighted in
     orange. File boundaries (when multiple files are combined) shown as
     thin dashed vertical lines.
  2. Gyroscope — deg/s per axis.
  3. Integrated gyroscope — cumulative angle per axis. X/Y are high-passed at
     HIGHPASS_HZ to suppress slow gyro-bias drift; Z is left raw (no gravity
     reference to anchor it).
  4. Magnetometer — uT per axis plus |B| total. Useful for the chip->PCB
     orientation test: point each silkscreen mag axis at magnetic north for
     ~10 s and observe which output channel responds.

Usage:
    python plot_raw_accel.py [PATH] [--utc]

PATH may be:
  - a single .dat file (legacy single-file mode)
  - a directory containing DATA_BOOT_*.dat files (e.g. BOOT_000090/);
    all files in the directory are decoded and combined into one
    contiguous timeseries

If no PATH is given, the newest DATA_BOOT_*.dat in the script directory is
used (or the newest BOOT_*/ folder, whichever has been modified more
recently). Add --utc to put UTC time on the x-axis instead of "seconds
since first sample"; UTC requires that the firmware obtained a GNSS fix
during the recording, otherwise the option is silently ignored.
"""

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

from decoder import decode_file, load_data_as_arrays


# X/Y integrated-angle high-pass cutoff. Removes slow gyro-bias drift
# (and any slow thermal wander on top) while keeping real rotation motion.
HIGHPASS_HZ = 0.1

# Drop this many seconds of IMU data from the start of the recording before
# plotting. The DMP firmware emits warmup garbage for ~1-2 s after boot which
# blows out the autoscaled y-axis on every panel. Set to 0 to disable.
SKIP_START_S = 1.5

# True = plot the x-axis as wall-clock UTC (datetime ticks). False = plot it
# as "elapsed seconds since first sample" (numeric). UTC needs the recording
# to have either PPS-regressed imu_utc, GNSS posix fixes, or a parseable
# filename timestamp — see _build_time_axis below.
USE_UTC = True

# uint32 micros() period — used to splice files when MCU micros wraps
# between consecutive files in the same boot folder.
MICROS_WRAP = 1 << 32


def find_default_path() -> Path | None:
    """Pick the BOOT_*/ folder with the highest boot number, or the newest
    single DAT if no BOOT folder exists.

    Boot number is parsed from the folder name (`BOOT_NNNNNN`) — that's a
    strictly monotonic counter on the OLA firmware so the highest number
    always identifies the most recent recording, even if filesystem mtimes
    have been clobbered by copying / extracting / sync tools.
    """
    here = Path(__file__).parent
    folders = [p for p in here.iterdir() if p.is_dir() and p.name.startswith("BOOT_")]
    if folders:
        def boot_number(p: Path) -> int:
            try:
                return int(p.name.split("_", 1)[1])
            except (IndexError, ValueError):
                return -1
        return max(folders, key=boot_number)
    files = list(here.glob("DATA_BOOT_*.dat"))
    if files:
        return max(files, key=lambda p: p.stat().st_mtime)
    return None


def _decode_and_load(dat_file: Path) -> dict:
    """Decode one .dat file and return the cleaned data dict."""
    result = decode_file(dat_file, allow_no_pps=True)
    return load_data_as_arrays(result["file"])


def load_data_from_path(path: Path) -> tuple[dict, list[Path], np.ndarray]:
    """Load one .dat file or a whole BOOT_*/ folder of .dat files.

    Returns:
        (data, files, file_starts)
        - data: concatenated arrays. Same keys as load_data_as_arrays().
        - files: list of source .dat files, in chronological order.
        - file_starts: array of imu sample indices where each file begins
          (for drawing file-boundary markers). file_starts[0] is always 0.
    """
    if path.is_dir():
        files = sorted(path.glob("DATA_BOOT_*.dat"))
        if not files:
            raise FileNotFoundError(f"No DATA_BOOT_*.dat files in {path}")
    elif path.is_file():
        files = [path]
    else:
        raise FileNotFoundError(path)

    per_file = [_decode_and_load(f) for f in files]

    # Concatenate every numeric or boolean array across files; pass scalars
    # (header strings, sensitivities) straight through from the first file.
    combined: dict = {}
    keys = per_file[0].keys()
    for k in keys:
        v0 = per_file[0][k]
        if not isinstance(v0, np.ndarray):
            combined[k] = v0
            continue
        if v0.ndim == 0 or v0.size == 0 and len(per_file) == 1:
            combined[k] = v0
            continue
        try:
            combined[k] = np.concatenate([d[k] for d in per_file if k in d and isinstance(d[k], np.ndarray)])
        except ValueError:
            # Mismatched dtypes / shapes — keep the first file's version.
            combined[k] = v0

    # Splice the per-file imu_micros_unwrapped into a globally monotonic
    # array. Within a boot session MCU micros() is monotonic except for the
    # uint32 wrap every ~71 min; per-file unwrap restarts at 0-aligned
    # phase, so we add a cumulative offset whenever the next file's first
    # micros is < the previous file's last (i.e. it wrapped).
    file_starts = [0]
    if "imu_micros_unwrapped" in combined and len(per_file) > 1:
        spliced = np.asarray(per_file[0]["imu_micros_unwrapped"], dtype=np.float64).copy()
        offset = 0.0
        for d in per_file[1:]:
            arr = np.asarray(d["imu_micros_unwrapped"], dtype=np.float64).copy()
            if arr.size == 0:
                file_starts.append(len(spliced))
                continue
            # Add wrap offsets if the new file's first sample is behind the
            # last sample of the running array.
            while arr[0] + offset < spliced[-1]:
                offset += MICROS_WRAP
            file_starts.append(len(spliced))
            spliced = np.concatenate((spliced, arr + offset))
        combined["imu_micros_unwrapped"] = spliced

    # Stash first filename for the filename-anchor UTC fallback.
    combined["_first_filename"] = files[0].name

    return combined, files, np.asarray(file_starts, dtype=np.int64)


def _clip_initial_seconds(
    data: dict, file_starts: np.ndarray, skip_s: float,
) -> tuple[dict, np.ndarray, int]:
    """Drop the first `skip_s` seconds of IMU data so the DMP warmup garbage
    at the start of every recording doesn't push the autoscaled y-axis out.

    Slices every `imu_*` array in `data` consistently from the same start
    index. GNSS/PPS arrays and scalar header fields are passed through
    untouched (their timestamps are absolute, not file-relative). The
    file-boundary indices are shifted to track the new IMU base index;
    boundaries that fell inside the clipped region are clamped to 0.

    Returns: (new_data, new_file_starts, n_dropped).
    """
    if skip_s <= 0:
        return data, file_starts, 0
    imu_us = np.asarray(data["imu_micros_unwrapped"], dtype=np.float64)
    if imu_us.size == 0:
        return data, file_starts, 0
    elapsed_s = (imu_us - imu_us[0]) * 1e-6
    # First sample whose elapsed time exceeds the skip window.
    n_drop = int(np.searchsorted(elapsed_s, skip_s, side="left"))
    if n_drop <= 0:
        return data, file_starts, 0
    if n_drop >= len(imu_us):
        # Recording is shorter than the skip window — keep at least
        # one sample so downstream code doesn't trip on empty arrays.
        n_drop = max(0, len(imu_us) - 1)
        if n_drop == 0:
            return data, file_starts, 0

    n_imu = len(imu_us)
    new_data = dict(data)
    for k, v in data.items():
        if not k.startswith("imu_"):
            continue
        if not isinstance(v, np.ndarray) or v.size != n_imu:
            continue
        new_data[k] = v[n_drop:]
    new_file_starts = np.maximum(file_starts - n_drop, 0).astype(np.int64)
    return new_data, new_file_starts, n_drop


def _build_time_axis(data: dict, use_utc: bool) -> tuple[np.ndarray, str, bool]:
    """Return (t, x_label, used_utc).

    UTC source preference (when use_utc=True):
      1. imu_utc — PPS-regressed UTC, only present when the firmware caught
         PPS edges and the decoder built a PPS->UTC regression. Best.
      2. gnss_posix anchored to gnss_micros_unwrapped via linear fit — works
         whenever the recording has GNSS PVT fixes (typical case when a
         GNSS is connected but PPS wiring is absent). Decent.
      3. Filename TIME_YYYYMMDDTHHMMSS embedded as a constant offset over
         elapsed seconds. Coarse (only as accurate as the filename timestamp)
         but always available.

    If use_utc=False, returns plain elapsed seconds since the first IMU sample.
    """
    imu_micros = np.asarray(data["imu_micros_unwrapped"], dtype=np.float64)

    if use_utc:
        # Path 1: PPS-regressed imu_utc
        utc_raw = np.asarray(data.get("imu_utc", []), dtype=np.float64)
        if utc_raw.size and np.any(np.isfinite(utc_raw)):
            t = (utc_raw * 1e9).astype("datetime64[ns]")
            return t, "UTC", True

        # Path 2: linear fit from gnss_micros_unwrapped -> gnss_posix
        g_micros = np.asarray(data.get("gnss_micros_unwrapped", []), dtype=np.float64)
        g_posix = np.asarray(data.get("gnss_posix", []), dtype=np.float64)
        if g_micros.size >= 2 and g_posix.size == g_micros.size and np.all(g_posix > 0):
            # Skip the "drift marker" entries the firmware stamps with
            # posix=0 immediately after a GNSS-vs-RTC re-sync.
            valid = g_posix > 1e9  # any plausible POSIX seconds after year 2001
            if valid.sum() >= 2:
                slope, intercept = np.polyfit(g_micros[valid], g_posix[valid], 1)
                imu_utc_est = slope * imu_micros + intercept
                t = (imu_utc_est * 1e9).astype("datetime64[ns]")
                print(f"--utc: PPS-regressed imu_utc unavailable, falling back to "
                      f"GNSS posix fit ({valid.sum()} fixes, "
                      f"slope={slope*1e6:.6f} s/Mcounts)")
                return t, "UTC (from GNSS posix fit)", True

        # Path 3: parse filename timestamp as anchor (assumes file order
        # corresponds to chronological order; we anchor at micros[0]).
        anchor = _parse_filename_timestamp(data.get("_first_filename"))
        if anchor is not None:
            elapsed = (imu_micros - imu_micros[0]) * 1e-6  # seconds
            anchor_ns = np.int64(anchor * 1e9)
            t = anchor_ns + (elapsed * 1e9).astype(np.int64)
            t = t.astype("datetime64[ns]")
            print("--utc: GNSS posix unavailable, anchoring at filename "
                  "timestamp (sub-second accuracy not guaranteed).")
            return t, "UTC (from filename anchor)", True

        print("--utc requested but no usable UTC source found. Falling "
              "back to elapsed seconds.")

    t = (imu_micros - imu_micros[0]) * 1e-6
    return t, "Time since first sample (s)", False


def _parse_filename_timestamp(name: str | None) -> float | None:
    """Extract POSIX seconds from a DATA_BOOT_*_TIME_YYYYMMDDTHHMMSS.dat
    filename. Returns None if the pattern doesn't match. The timestamp is
    interpreted as UTC.
    """
    if not name:
        return None
    import re
    from datetime import datetime, timezone
    m = re.search(r"_TIME_(\d{8})T(\d{6})", name)
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        return dt.replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def _gap_threshold_idx(t: np.ndarray, used_utc: bool) -> tuple[float, np.ndarray]:
    """Return (gap_threshold_seconds, indices_with_gap)."""
    if used_utc:
        dt_sec = np.diff(t).astype("timedelta64[ns]").astype(np.float64) * 1e-9
    else:
        dt_sec = np.diff(t)
    dt_med = float(np.median(dt_sec))
    gap_threshold = max(2 * dt_med, 0.020)
    gap_idx = np.where(dt_sec > gap_threshold)[0]
    return gap_threshold, gap_idx, dt_med, dt_sec


def _format_time_axis(ax, used_utc: bool):
    """Apply UTC date formatting to the x-axis if we're using UTC."""
    if used_utc:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=None))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", type=Path, default=None,
                    help="Single .dat file OR a BOOT_*/ folder of .dat files")
    args = ap.parse_args()

    path = args.path
    if path is None:
        path = find_default_path()
        if path is None:
            print("No data file or BOOT_*/ folder given and none found.")
            print("Usage: python plot_raw_accel.py [path/to/file.dat OR path/to/BOOT_NNNNNN]")
            return
        kind = "folder" if path.is_dir() else "file"
        print(f"Auto-selected newest {kind}: {path.name}")

    if not path.exists():
        print(f"Not found: {path}")
        return

    
    # Decode + load (handles both single file and folder)
    data, source_files, file_starts = load_data_from_path(path)

    if len(data["imu_micros_unwrapped"]) == 0:
        print("No IMU samples in this recording.")
        return

    # Trim the DMP warmup region off the front of the IMU data (see
    # SKIP_START_S at the top of this file).
    data, file_starts, n_dropped = _clip_initial_seconds(data, file_starts, SKIP_START_S)
    if n_dropped > 0:
        print(f"Dropped first {n_dropped} IMU samples ({SKIP_START_S:.1f} s) "
              f"of warmup / startup transient.")

    title_name = path.name if path.is_dir() else source_files[0].name
    if len(source_files) > 1:
        title_name = f"{path.name} ({len(source_files)} files)"

    # Time axis
    t, t_label, used_utc = _build_time_axis(data, use_utc=USE_UTC)

    # Gap detection (works for both seconds and datetime64 x-arrays)
    gap_threshold, gap_idx, dt_med, dt_sec = _gap_threshold_idx(t, used_utc)

    # Stats
    duration = (t[-1] - t[0])
    if used_utc:
        duration_sec = duration.astype("timedelta64[s]").astype(np.float64)
    else:
        duration_sec = duration
    print(f"\nSource:            {title_name}")
    print(f"Samples:           {len(t)}")
    print(f"Duration:          {duration_sec:.1f} s")
    print(f"Median dt:         {dt_med*1000:.3f} ms (~{1/dt_med:.1f} Hz)")
    print(f"Max dt:            {dt_sec.max()*1000:.1f} ms")
    print(f"Gaps > {gap_threshold*1000:.0f} ms:    {len(gap_idx)}")
    if len(gap_idx) > 0:
        worst = gap_idx[np.argmax(dt_sec[gap_idx])]
        # Pretty-print the location of the worst gap
        if used_utc:
            print(f"  Worst gap: {dt_sec[gap_idx].max()*1000:.1f} ms at t={t[worst]}")
        else:
            print(f"  Worst gap: {dt_sec[gap_idx].max()*1000:.1f} ms at t={t[worst]:.1f} s")

    labels = ("X", "Y", "Z")
    colors = ("tab:blue", "tab:green", "tab:red")
    gap_title = (
        f"{len(t)} samples, {len(gap_idx)} gaps > {gap_threshold*1000:.0f} ms"
        + (f" (max {dt_sec.max()*1000:.0f} ms)" if len(gap_idx) else "")
    )

    def _shade_gaps(ax_):
        for gi in gap_idx:
            ax_.axvspan(t[gi], t[gi + 1], color="orange", alpha=0.25)

    def _mark_file_boundaries(ax_):
        # Vertical dashed line at each file boundary (skip the first, which is
        # the start of the recording itself).
        for fs in file_starts[1:]:
            ax_.axvline(t[fs], color="0.4", linestyle="--", linewidth=0.7, alpha=0.6)

    # Plot 1: accelerometer (mg)
    fig_acc, axes_acc = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for k, (lab, c) in enumerate(zip(labels, colors)):
        axes_acc[k].plot(t, data[f"imu_acc_{lab.lower()}"], color=c, linewidth=0.6)
        _shade_gaps(axes_acc[k])
        _mark_file_boundaries(axes_acc[k])
        axes_acc[k].set_ylabel(f"Acc {lab} (mg)")
        axes_acc[k].grid(True, alpha=0.3)
    _format_time_axis(axes_acc[-1], used_utc)
    axes_acc[-1].set_xlabel(t_label)
    axes_acc[0].set_title(f"Raw accelerometer — {title_name}\n{gap_title}")
    fig_acc.tight_layout()

    # Plot 2: gyroscope. Decoder gives mdps; show in deg/s for readability.
    fig_gyr, axes_gyr = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for k, (lab, c) in enumerate(zip(labels, colors)):
        gyr_dps = np.asarray(data[f"imu_gyr_{lab.lower()}"]) / 1000.0
        axes_gyr[k].plot(t, gyr_dps, color=c, linewidth=0.6)
        _shade_gaps(axes_gyr[k])
        _mark_file_boundaries(axes_gyr[k])
        axes_gyr[k].axhline(0, color="k", linewidth=0.5, alpha=0.3)
        axes_gyr[k].set_ylabel(f"Gyr {lab} (deg/s)")
        axes_gyr[k].grid(True, alpha=0.3)
    _format_time_axis(axes_gyr[-1], used_utc)
    axes_gyr[-1].set_xlabel(t_label)
    axes_gyr[0].set_title(f"Raw gyroscope — {title_name}\n{gap_title}")
    fig_gyr.tight_layout()

    # Plot 3: integrated gyroscope (per-axis cumulative angle). Trapezoidal
    # integration of raw deg/s gives degrees; with no bias correction this is
    # the unfiltered "where did the gyro think it pointed". Slow drift here
    # is gyro bias; sharp ramps are real rotations.
    fig_int, axes_int = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fs_int = 1.0 / max(dt_med, 1e-6)
    sos_hp = butter(4, HIGHPASS_HZ, btype="highpass", fs=fs_int, output="sos")
    for k, (lab, c) in enumerate(zip(labels, colors)):
        gyr_dps = np.asarray(data[f"imu_gyr_{lab.lower()}"]) / 1000.0
        angle = np.concatenate((
            [0.0],
            np.cumsum(0.5 * (gyr_dps[:-1] + gyr_dps[1:]) * dt_sec),
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
        _shade_gaps(axes_int[k])
        _mark_file_boundaries(axes_int[k])
        axes_int[k].axhline(0, color="k", linewidth=0.5, alpha=0.3)
        axes_int[k].set_ylabel(f"Integrated Gyr {lab} (deg)")
        axes_int[k].grid(True, alpha=0.3)
    _format_time_axis(axes_int[-1], used_utc)
    axes_int[-1].set_xlabel(t_label)
    axes_int[0].set_title(
        f"Integrated gyroscope (X/Y high-pass > {HIGHPASS_HZ} Hz) — {title_name}\n{gap_title}"
    )
    fig_int.tight_layout()

    # Plot 4: magnetometer (raw chip-frame uT).
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
            _shade_gaps(axes_mag[k])
            _mark_file_boundaries(axes_mag[k])
            axes_mag[k].axhline(0, color="k", linewidth=0.5, alpha=0.3)
            axes_mag[k].set_ylabel(f"Mag {lab} (uT)")
            axes_mag[k].grid(True, alpha=0.3)
        median_b = float(np.median(mag_total))
        axes_mag[3].plot(t, mag_total, color="tab:purple", linewidth=0.6)
        axes_mag[3].axhline(
            median_b, color="k", linewidth=0.5, linestyle="--", alpha=0.4,
            label=f"median |B| = {median_b:.1f} uT",
        )
        _shade_gaps(axes_mag[3])
        _mark_file_boundaries(axes_mag[3])
        axes_mag[3].set_ylabel("|B| (uT)")
        axes_mag[3].legend(loc="upper right", fontsize=8)
        axes_mag[3].grid(True, alpha=0.3)
        _format_time_axis(axes_mag[-1], used_utc)
        axes_mag[-1].set_xlabel(t_label)
        axes_mag[0].set_title(
            f"Raw magnetometer (chip frame) — {title_name}\n{gap_title}"
        )
        fig_mag.tight_layout()
        print(
            f"\nMag stats over recording:"
            f"\n  median |B|     = {median_b:5.1f} uT"
            f"\n  mx mean/range = {mag_x.mean():+6.1f} / [{mag_x.min():+6.1f}, {mag_x.max():+6.1f}] uT"
            f"\n  my mean/range = {mag_y.mean():+6.1f} / [{mag_y.min():+6.1f}, {mag_y.max():+6.1f}] uT"
            f"\n  mz mean/range = {mag_z.mean():+6.1f} / [{mag_z.min():+6.1f}, {mag_z.max():+6.1f}] uT"
        )
    else:
        print(
            "No magnetometer data in this recording "
            "(older firmware, or all-zero readings)."
        )

    plt.show()


if __name__ == "__main__":
    main()
