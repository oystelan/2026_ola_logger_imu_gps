#!/usr/bin/env python3
"""Compare vertical-acceleration estimates from two co-located wave buoys:
the OLA prototype (BOOT_000093 folder) and the SFY reference buoy (test.nc).

Pipeline:
  1. Load test.nc with xarray. Variable `w_z` is the SFY world-frame vertical
     acceleration in m/s², including gravity (its mean is ~9.56 m/s² here).
     Subtract its mean over the comparison window to get the linear vertical
     accel that's directly comparable to the OLA AHRS output.

  2. Decode all DATA_BOOT_*.dat files in BOOT_000093/ and combine them. Build
     an MCU-micros -> UTC mapping via linear regression on the GNSS PVT
     entries' posix timestamps (same trick the plot_raw_accel.py UTC fallback
     uses). This anchors every IMU sample to absolute UTC.

  3. Run compute_vertical_motion() (Madgwick + FFT band-pass integrator) on
     the full OLA timeseries — the AHRS needs the recording from sample 0
     for its stationary-init bootstrap, so we don't pre-slice. The output is
     world-frame vertical accel +up in m/s², gravity removed and band-passed
     in [LOW_HZ, HIGH_HZ].

  4. Slice both signals to the user-specified UTC window WINDOW_START..WINDOW_END
     and plot the two vertical accelerations on a shared time axis. Also
     report basic per-signal statistics in the window.
"""

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Allow imports from the parent decoder/ folder.
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from ahrs_vertical import (
    compute_vertical_motion,
    compute_vertical_motion_lowpass_gravity,
    compute_vertical_motion_savgol_detrend,
    compute_vertical_motion_fixed_attitude,
    compute_vertical_motion_mahony,
)
from decoder import decode_file, load_data_as_arrays


# ----------------------------- user knobs ------------------------------------

# Comparison window (UTC).
WINDOW_START = np.datetime64("2026-06-14T20:29:50")
WINDOW_END   = np.datetime64("2026-06-14T20:35:20")

OLA_FOLDER  = HERE / "BOOT_000008"
SFY_NETCDF  = HERE / "test5_fixed.nc"

# AHRS band-pass edges. Both buoys are looking at the same waves so we want
# the same passband for both.
LOW_HZ  = 0.05
HIGH_HZ = 5.

# Which OLA vertical-motion estimator to run:
#   "fixed"    — fixed bootstrap attitude, no per-sample tracking. Best for a
#                near-level float; preserves the full vertical amplitude
#                (the tracking methods attenuate it ~2x on this data).
#   "madgwick" — Madgwick AHRS (accel+gyro).
#   "lowpass"  — complementary filter.
#   "savgol"   — gyro AHRS + Savitzky-Golay attitude detrend.
#   "mahony"   — Mahony complementary filter w/ online gyro-bias (Ki) feedback,
#                accel correction never gated. Robust to sustained rotation —
#                the SFY-style continuous-correction approach.
METHOD = "mahony"

# Mahony gains (only used when METHOD == "mahony"). Higher kp = stronger pull
# toward gravity = better rotation rejection, but too high attenuates real
# heave (the filter explains heave accel away as tilt). kp~5 is the sweet
# spot found against the group-3 rotation recording. ki learns gyro bias.
MAHONY_KP = 8.0
MAHONY_KI = 1.25

# Save the time-corrected vertical-acceleration timeseries (the middle panel:
# OLA accel_z_up gravity-removed + SFY w_z-mean) to an .npz so a downstream
# script can integrate accel -> velocity -> displacement. Each buoy is saved on
# its OWN corrected-UTC grid (OLA: #4 timebase fix + OLA_TIME_SHIFT_S; SFY:
# rate/offset rebuild). Times are int64 ns since the Unix epoch (UTC).
SAVE_TIMESERIES = True

# Manual time shift applied to the OLA timestamps before plotting (seconds;
# positive = shift OLA later in time). Set to 0.0 to trust the absolute UTC
# of both recordings (the expectation now that both are GNSS-time-stamped).
# Use ENABLE_XCORR_DIAG below to PRINT the cross-correlation lag for sanity
# without applying it.
OLA_TIME_SHIFT_S = -1.1

# === SFY time-axis rebuild ===========================================
# We established that OLA is the GPS-true clock (its IMU micros are mapped
# per-sample through the GNSS->UTC regression, and its motion onsets land on
# the whole-minute marks the user started on). SFY, by contrast, drifts: its
# netCDF advertises estimated_frequency = 56.11 Hz but the *stored* time axis
# advances at ~55.84 Hz (1 ms-quantized 17/18 ms steps), a ~0.5% slow rate
# that accumulates to ~1-2 s over a few minutes. So instead of shifting OLA we
# REBUILD the SFY time axis: anchor it at the window's first stored sample and
# replay elapsed time scaled by SFY_RATE_SCALE, then add SFY_TIME_OFFSET_S to
# line up the absolute offset.
#   t_sfy_corr = t_sfy[0] + (t_sfy - t_sfy[0]) * SFY_RATE_SCALE + SFY_TIME_OFFSET_S
# SFY_RATE_SCALE < 1 compresses SFY's elapsed time onto its true rate (removes
# the drift slope); SFY_TIME_OFFSET_S is the constant offset (replaces what
# OLA_TIME_SHIFT_S used to absorb). Set SFY_RATE_SCALE=1 and SFY_TIME_OFFSET_S=0
# to fall back to the raw stored axis.
SFY_RATE_SCALE = 1.
#SFY_RATE_SCALE   = 55.54 / 56.11   # ~0.98984 — visually tuned to near-perfect:
                                   # residual drift -0.07%, crest scatter 65 ms
                                   # over the whole record. (Raw SFY drift vs OLA
                                   # was ~-1%, more than 55.84/56.11 predicted, so
                                   # the numerator was lowered 55.84 -> 55.54.)
#SFY_TIME_OFFSET_S = 6.420          # constant offset after rate correction (s)
SFY_TIME_OFFSET_S = 0.6

# When True, still compute and print the cross-correlation lag between the
# OLA raw acc_x and SFY w_z as a diagnostic — but it is NOT applied unless
# you copy the printed value into OLA_TIME_SHIFT_S yourself.
ENABLE_XCORR_DIAG = False

# Lead-in (seconds) prepended to the comparison window before running the
# AHRS on the OLA data. The AHRS is run on [WINDOW_START - PAD, WINDOW_END]
# (NOT the whole recording) so the attitude bootstrap and gyro-bias estimate
# come from data near the window — not from any violent device-handling
# transients elsewhere in the recording, which otherwise corrupt the
# Madgwick attitude and collapse the world-vertical output. The pad gives
# the filter a few seconds to settle before the window of interest; only the
# WINDOW_START..WINDOW_END portion is plotted/scored. Keep the device roughly
# still during the pad for the cleanest gravity/bias init.
WINDOW_PAD_S = 5.0


# ------------------------ OLA load + UTC mapping -----------------------------


def _load_ola_combined(folder: Path) -> dict:
    """Decode every DAT in the folder, concatenate, and return one dict."""
    files = sorted(folder.glob("DATA_BOOT_*.dat"))
    if not files:
        raise FileNotFoundError(f"No DATA_BOOT_*.dat under {folder}")
    per_file = []
    for f in files:
        result = decode_file(f, allow_no_pps=True)
        per_file.append(load_data_as_arrays(result["file"]))

    combined: dict = {}
    for k in per_file[0].keys():
        v0 = per_file[0][k]
        if not isinstance(v0, np.ndarray):
            combined[k] = v0
            continue
        try:
            combined[k] = np.concatenate([
                d[k] for d in per_file if isinstance(d[k], np.ndarray)
            ])
        except ValueError:
            combined[k] = v0
    return combined


def _cross_correlate_lag(
    ola_t_ns: np.ndarray, ola_signal: np.ndarray,
    sfy_t_ns: np.ndarray, sfy_signal: np.ndarray,
    max_lag_s: float = 15.0,
) -> float:
    """Find the lag (s) at which OLA most strongly matches SFY.

    Resamples both signals onto a common uniform grid at the SFY rate, then
    computes the full cross-correlation and reports the lag in seconds at
    which it peaks within the search range ±max_lag_s. Positive return value
    means the OLA signal is *later* than SFY (so subtract it from OLA times).
    """
    # Pick a common UTC reference (the earlier of the two signal starts) and
    # resample both onto a uniform grid at SFY's nominal rate.
    t0_ns = min(ola_t_ns[0], sfy_t_ns[0])
    ola_rel = (ola_t_ns - t0_ns).astype("timedelta64[ns]").astype(np.float64) * 1e-9
    sfy_rel = (sfy_t_ns - t0_ns).astype("timedelta64[ns]").astype(np.float64) * 1e-9
    fs = 52.0  # SFY nominal rate
    t_grid = np.arange(max(ola_rel[0], sfy_rel[0]),
                       min(ola_rel[-1], sfy_rel[-1]),
                       1.0 / fs)
    ola_r = np.interp(t_grid, ola_rel, ola_signal)
    sfy_r = np.interp(t_grid, sfy_rel, sfy_signal)
    # Zero-mean and unit-norm so the lag-zero cross-correlation magnitude is
    # bounded by 1 (i.e. a real Pearson correlation).
    ola_r = ola_r - ola_r.mean()
    sfy_r = sfy_r - sfy_r.mean()
    ola_r = ola_r / (np.linalg.norm(ola_r) + 1e-12)
    sfy_r = sfy_r / (np.linalg.norm(sfy_r) + 1e-12)
    xcorr = np.correlate(ola_r, sfy_r, mode="full")
    lags_samples = np.arange(-len(sfy_r) + 1, len(ola_r))
    # Restrict the search to ±max_lag_s
    keep = np.abs(lags_samples) <= int(round(max_lag_s * fs))
    best = lags_samples[keep][int(np.argmax(xcorr[keep]))]
    return float(best) / fs


def _build_utc_mapping(data: dict) -> tuple[float, float]:
    """Return (slope, intercept) so that utc_posix = slope*micros + intercept.

    Fits over GNSS PVT entries whose `gnss_posix` looks plausible. Skips
    posix=0 entries (firmware's drift-marker spike).
    """
    g_micros = np.asarray(data.get("gnss_micros_unwrapped", []), dtype=np.float64)
    g_posix  = np.asarray(data.get("gnss_posix", []), dtype=np.float64)
    if g_micros.size < 2 or g_micros.size != g_posix.size:
        raise RuntimeError("OLA recording lacks GNSS posix data for UTC mapping")
    valid = g_posix > 1e9
    if valid.sum() < 2:
        raise RuntimeError("No usable GNSS fixes in the OLA recording")
    slope, intercept = np.polyfit(g_micros[valid], g_posix[valid], 1)
    return float(slope), float(intercept)


# ------------------------------- main ----------------------------------------


def main() -> None:
    # === SFY ===
    print(f"Loading SFY  -> {SFY_NETCDF}")
    sfy = xr.open_dataset(SFY_NETCDF)
    print(f"  variables : {list(sfy.data_vars)}")
    print(f"  time range: {sfy.time.values[0]} .. {sfy.time.values[-1]}")
    print(f"  frequency : {sfy.attrs.get('frequency')} Hz "
          f"(estimated {sfy.attrs.get('estimated_frequency'):.2f})")

    sfy_win = sfy.sel(time=slice(WINDOW_START, WINDOW_END))
    sfy_t   = sfy_win.time.values
    sfy_wz  = sfy_win.w_z.values.astype(np.float64)

    # --- Rebuild SFY's (drifting) time axis onto its true rate (see config).
    if SFY_RATE_SCALE != 1.0 or SFY_TIME_OFFSET_S != 0.0:
        t0_ns = sfy_t[0]
        elapsed_s = (sfy_t - t0_ns).astype("timedelta64[ns]").astype(np.float64) * 1e-9
        corr_s = elapsed_s * SFY_RATE_SCALE + SFY_TIME_OFFSET_S
        sfy_t = t0_ns + (corr_s * 1e9).astype("timedelta64[ns]")
        print(f"  SFY time axis rebuilt: rate_scale={SFY_RATE_SCALE:.5f} "
              f"(stored ~55.84 Hz -> true ~56.11 Hz), offset={SFY_TIME_OFFSET_S:+.3f} s; "
              f"span {elapsed_s[-1]:.1f}s -> {corr_s[-1]:.1f}s "
              f"(drift removed = {elapsed_s[-1]*(1-SFY_RATE_SCALE):+.2f} s)")
    # Remove gravity: SFY w_z averages ~9.56 m/s² over the recording. The
    # user's comparison window is short enough that this is essentially
    # constant. Subtract the WINDOW mean so the result is the AC vertical
    # acceleration directly comparable to AHRS.
    sfy_wz_ac = sfy_wz - float(np.mean(sfy_wz))
    print(f"  SFY in window: {sfy_t.size} samples, w_z mean={np.mean(sfy_wz):.3f} m/s², "
          f"AC std={np.std(sfy_wz_ac):.3f} m/s²")

    # === OLA ===
    print(f"\nLoading OLA -> {OLA_FOLDER}")
    ola = _load_ola_combined(OLA_FOLDER)
    n_imu = len(ola["imu_micros_unwrapped"])
    print(f"  IMU samples: {n_imu}")

    # IMU-counter rate correction (for recordings made with the OLD firmware,
    # before the timestamp-PLL fix). The firmware stamped each IMU sample with
    # a counter advancing at the ASSUMED ODR (IMU_ODR_HZ=100), but the DMP
    # actually emits accel at 1125/(1+SMPLRT_DIV)=102.27 Hz. The counter's
    # elapsed time is therefore overstated by 102.27/100, which stretches the
    # OLA timeline ~2.27% when mapped to UTC and shows up as a drift vs SFY.
    # We undo it by scaling each sample's counter-elapsed-since-start back to
    # real elapsed before applying the GNSS→UTC fit:
    #     real_elapsed = counter_elapsed * (ASSUMED_ODR / TRUE_ODR)
    # Set to 1.0 for recordings made with the PLL-fixed firmware (it locks the
    # sample timeline to the real micros() clock at the source).
    # BOOT_000005 was recorded with the PLL-fixed firmware -> no correction.
    OLA_IMU_RATE_CORRECTION = 1.0

    imu_micros = np.asarray(ola["imu_micros_unwrapped"], dtype=np.float64)
    imu_micros_corr = imu_micros[0] + (imu_micros - imu_micros[0]) * OLA_IMU_RATE_CORRECTION
    slope, intercept = _build_utc_mapping(ola)
    imu_utc_sec = slope * imu_micros_corr + intercept
    imu_utc_ns  = (imu_utc_sec * 1e9).astype("datetime64[ns]")
    print(f"  IMU rate correction: x{OLA_IMU_RATE_CORRECTION:.5f} "
          f"(counter@100Hz -> true@{1125.0/11:.2f}Hz)")
    print(f"  IMU UTC range: {imu_utc_ns[0]} .. {imu_utc_ns[-1]}")

    # Slice the OLA data to [WINDOW_START - PAD, WINDOW_END] BEFORE the AHRS.
    # Running the AHRS over the whole recording lets device-handling transients
    # outside the window (large rotations, gravity flips) corrupt the gyro-bias
    # estimate and Madgwick attitude convergence, which collapses the in-window
    # world-vertical output. Restricting to the window + a short settling pad
    # gives a clean stationary-init near the data of interest. This mirrors how
    # ahrs_example.py works on a clean recording.
    pad_td = np.timedelta64(int(round(WINDOW_PAD_S * 1e9)), "ns")
    ahrs_lo = WINDOW_START - pad_td
    ahrs_sel = (imu_utc_ns >= ahrs_lo) & (imu_utc_ns <= WINDOW_END)
    n_imu_total = len(imu_utc_ns)
    ola_ahrs = dict(ola)
    for k, v in ola.items():
        if isinstance(v, np.ndarray) and v.size == n_imu_total:
            ola_ahrs[k] = v[ahrs_sel]
    ahrs_imu_utc_sec = imu_utc_sec[ahrs_sel]
    print(f"  AHRS input slice: {ahrs_sel.sum()} samples "
          f"({ahrs_imu_utc_sec[-1] - ahrs_imu_utc_sec[0]:.1f} s, "
          f"incl. {WINDOW_PAD_S:.0f}s lead-in)")

    # AHRS on the sliced data. Gravity axis is auto-detected and rotated onto
    # body -Z inside compute_vertical_motion (auto_align_gravity=True default),
    # so mounting orientation is handled transparently.
    print(f"\nRunning ahrs_vertical (METHOD={METHOD!r}, low_hz={LOW_HZ}, "
          f"high_hz={HIGH_HZ}) with built-in gravity auto-align...")
    _ahrs_fn = {
        "fixed":    compute_vertical_motion_fixed_attitude,
        "madgwick": compute_vertical_motion,
        "lowpass":  compute_vertical_motion_lowpass_gravity,
        "savgol":   compute_vertical_motion_savgol_detrend,
        "mahony":   compute_vertical_motion_mahony,
    }[METHOD]
    # Mahony (the SFY-style continuous-correction filter) needs its own gains.
    # Key lessons from group-3 diagnosis: do NOT pre-subtract a one-shot gyro
    # bias (the init window can catch motion / bias drifts over minutes) and do
    # NOT hard-gate the accel correction (that forces open-loop gyro exactly
    # during the rotation). Instead keep the accel correction always on with a
    # strong proportional gain and let the Ki integral learn the bias online.
    _method_kwargs = {
        "mahony": dict(kp=MAHONY_KP, ki=MAHONY_KI,
                       motion_gate_threshold=1e9, calibrate_gyro_bias=False),
    }.get(METHOD, {})
    res = _ahrs_fn(
        ola_ahrs, low_hz=LOW_HZ, high_hz=HIGH_HZ, **_method_kwargs,
    )
    # res.t is 0-anchored at the first sample of the SLICE, but it is in
    # MCU-CLOCK seconds (compute_vertical_* builds its time base from the raw
    # imu_micros elapsed). The Artemis crystal is ~+450 ppm fast, so we must
    # apply the SAME GNSS rate correction (slope) used for the per-sample UTC
    # mapping before re-anchoring — otherwise world_z runs ~0.15 s fast vs the
    # GNSS-true acc_x_raw across the window (a growing phase shift). slope is
    # UTC-seconds per MCU-microsecond, so slope/1e-6 converts MCU-elapsed-
    # seconds -> UTC-elapsed-seconds. (Assumes OLA_IMU_RATE_CORRECTION == 1.0.)
    ahrs_utc_sec = ahrs_imu_utc_sec[0] + res.t * (slope / 1e-6)
    ahrs_utc_ns  = (ahrs_utc_sec * 1e9).astype("datetime64[ns]")

    # Slice the AHRS output down to the actual comparison window (drop the pad).
    sel = (ahrs_utc_ns >= WINDOW_START) & (ahrs_utc_ns <= WINDOW_END)
    ola_t  = ahrs_utc_ns[sel]
    ola_az = res.accel_z_up[sel]
    ola_az_raw = res.accel_z_up_raw[sel]
    ola_roll = res.roll_deg[sel]
    ola_pitch = res.pitch_deg[sel]
    print(f"  OLA in window: {ola_t.size} samples, band-passed accel std={np.std(ola_az):.3f} m/s²")

    # Raw OLA acc_x — body-frame X axis. On this OLA hardware the silkscreen
    # PCB-X arrow points up (gravity on +X register reads ~+1g), so acc_x is
    # the dominant gravity-containing axis. Subtract the window mean to get
    # the AC component for direct apples-to-apples comparison with SFY's
    # (w_z - mean). Convert from mg to m/s² to match SFY units.
    mg_to_ms2 = 9.80665 / 1000.0
    sel_raw = (imu_utc_ns >= WINDOW_START) & (imu_utc_ns <= WINDOW_END)
    raw_acc_x_window = np.asarray(ola["imu_acc_x"])[sel_raw] * mg_to_ms2
    raw_acc_x_window_ac = raw_acc_x_window - float(np.mean(raw_acc_x_window))
    raw_acc_x_t = imu_utc_ns[sel_raw]
    print(f"  OLA raw acc_x in window: {raw_acc_x_t.size} samples, "
          f"mean={float(np.mean(raw_acc_x_window)):.3f} m/s² (~= gravity), "
          f"AC std={np.std(raw_acc_x_window_ac):.3f} m/s²")

    # === Time alignment ===
    # Both recordings are now GNSS-time-stamped, so by default we trust their
    # absolute UTC and apply no shift (OLA_TIME_SHIFT_S = 0). The
    # cross-correlation is still computed as a DIAGNOSTIC when ENABLE_XCORR_DIAG
    # is set, so you can see how well the absolute clocks actually agree — but
    # it is not applied automatically; if you want to apply it, copy the
    # printed lag into OLA_TIME_SHIFT_S.
    if ENABLE_XCORR_DIAG:
        lag_s = _cross_correlate_lag(
            ola_t_ns=raw_acc_x_t, ola_signal=raw_acc_x_window_ac,
            sfy_t_ns=sfy_t,       sfy_signal=sfy_wz_ac,
            max_lag_s=15.0,
        )
        print(f"  [diag] cross-correlation lag (OLA - SFY): {lag_s:+.3f} s "
              f"(NOT applied; OLA_TIME_SHIFT_S = {OLA_TIME_SHIFT_S:+.3f} s)")

    if OLA_TIME_SHIFT_S != 0.0:
        shift_td = np.timedelta64(int(round(OLA_TIME_SHIFT_S * 1e9)), "ns")
        ola_t       = ola_t       + shift_td
        raw_acc_x_t = raw_acc_x_t + shift_td

    # === Save the time-corrected vertical-accel timeseries (middle panel) ===
    # OLA accel_z_up (gravity removed, +up) and SFY (w_z - window mean), each on
    # its own corrected-UTC grid. For the downstream integration script. Times
    # are int64 nanoseconds since the Unix epoch (UTC); recover seconds with
    # (t_ns - t_ns[0]) / 1e9. accel_up is m/s², +up, gravity already removed.
    if SAVE_TIMESERIES:
        ola_t_ns = ola_t.astype("datetime64[ns]").astype(np.int64)
        sfy_t_ns = sfy_t.astype("datetime64[ns]").astype(np.int64)
        out_path = HERE / f"vertical_timeseries_{OLA_FOLDER.name}.npz"
        np.savez(
            out_path,
            ola_time_ns=ola_t_ns,
            ola_accel_up=ola_az_raw.astype(np.float64),          # gravity removed, NOT band-passed
            ola_accel_up_bandpassed=ola_az.astype(np.float64),   # AHRS band-passed [LOW_HZ, HIGH_HZ]
            sfy_time_ns=sfy_t_ns,
            sfy_accel_up=sfy_wz_ac.astype(np.float64),           # w_z - window mean
            meta=np.array(
                f"method={METHOD}; band=[{LOW_HZ},{HIGH_HZ}]Hz; "
                f"ola_time_shift_s={OLA_TIME_SHIFT_S}; sfy_rate_scale={SFY_RATE_SCALE}; "
                f"sfy_time_offset_s={SFY_TIME_OFFSET_S}; "
                f"window={WINDOW_START}..{WINDOW_END}"),
        )
        ola_hz = 1.0 / np.median(np.diff(ola_t_ns) / 1e9)
        sfy_hz = 1.0 / np.median(np.diff(sfy_t_ns) / 1e9)
        print(f"\nSaved vertical timeseries -> {out_path}")
        print(f"  OLA: {ola_t_ns.size} samples @ ~{ola_hz:.2f} Hz (accel_up + accel_up_bandpassed)")
        print(f"  SFY: {sfy_t_ns.size} samples @ ~{sfy_hz:.2f} Hz (accel_up)")

    # === Plot ===
    fig, axes = plt.subplots(3, 1, figsize=(13, 9.5), sharex=True)

    # Panel 1: band-passed AHRS vs SFY (best apples-to-apples)
    axes[0].plot(ola_t,  ola_az, color="tab:blue", lw=0.8, label="OLA — accel_z_up (band-passed)")
    axes[0].plot(sfy_t, sfy_wz_ac, color="tab:red",  lw=0.8, label="SFY — w_z - mean (AC)")
    axes[0].set_ylabel("Vertical accel (m/s²)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[0].set_title(
        f"Vertical acceleration — OLA {OLA_FOLDER.name} vs SFY {SFY_NETCDF.name}\n"
        f"window {WINDOW_START}..{WINDOW_END}   METHOD={METHOD}   "
        f"AHRS band [{LOW_HZ}, {HIGH_HZ}] Hz"
    )

    # Panel 2: AHRS raw (gravity-removed but not band-passed) vs SFY
    axes[1].plot(ola_t,  ola_az_raw, color="tab:blue", lw=0.8, label="OLA — accel_z_up (raw, gravity removed)")
    axes[1].plot(sfy_t, sfy_wz_ac,   color="tab:red",  lw=0.8, label="SFY — w_z - mean (AC)")
    axes[1].set_ylabel("Vertical accel (m/s²)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    # Panel 3: raw OLA body-frame acc_x (mean removed) vs SFY. Shows the
    # pre-AHRS body-frame signal — if the OLA is roughly level so PCB-X
    # already points up, this should look very similar to the AHRS output.
    # Useful to confirm the AHRS isn't introducing or removing energy.
    axes[2].plot(raw_acc_x_t, raw_acc_x_window_ac, color="tab:purple", lw=0.8,
                 label="OLA — raw acc_x - mean (body frame)")
    axes[2].plot(sfy_t, sfy_wz_ac, color="tab:red", lw=0.8, label="SFY — w_z - mean (AC)")
    axes[2].set_ylabel("Vertical accel (m/s²)")
    axes[2].set_xlabel("UTC")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=None))

    fig.tight_layout()

    # === Gyro plot (separate figure) ===
    # Body-frame gyro in the window, to inspect whether the device was rotating
    # during the comparison interval. If the body→NED transform looks wrong,
    # large gyro activity here is the usual culprit (the AHRS attitude is being
    # driven by gyro and any bias / fast rotation tilts the world frame). The
    # decoder stores gyro in mdps; show deg/s.
    sel_gyr = (imu_utc_ns >= WINDOW_START) & (imu_utc_ns <= WINDOW_END)
    gyr_t = imu_utc_ns[sel_gyr]
    gx = np.asarray(ola["imu_gyr_x"])[sel_gyr] / 1000.0
    gy = np.asarray(ola["imu_gyr_y"])[sel_gyr] / 1000.0
    gz = np.asarray(ola["imu_gyr_z"])[sel_gyr] / 1000.0
    fig_g, axg = plt.subplots(figsize=(13, 4.5))
    axg.plot(gyr_t, gx, color="tab:blue", lw=0.7, label="gyr X")
    axg.plot(gyr_t, gy, color="tab:green", lw=0.7, label="gyr Y")
    axg.plot(gyr_t, gz, color="tab:red", lw=0.7, label="gyr Z")
    axg.axhline(0, color="k", lw=0.5, alpha=0.3)
    axg.set_ylabel("Body-frame angular rate (deg/s)")
    axg.set_xlabel("UTC")
    axg.grid(True, alpha=0.3)
    axg.legend(loc="upper right")
    axg.set_title(
        f"OLA body-frame gyro in window — {OLA_FOLDER.name}\n"
        f"std (deg/s): X={gx.std():.1f} Y={gy.std():.1f} Z={gz.std():.1f}; "
        f"max|.|: X={np.abs(gx).max():.0f} Y={np.abs(gy).max():.0f} Z={np.abs(gz).max():.0f}"
    )
    axg.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=None))
    fig_g.tight_layout()
    print(f"\nWindow gyro (deg/s): "
          f"X std={gx.std():.2f} max|.|={np.abs(gx).max():.1f}; "
          f"Y std={gy.std():.2f} max|.|={np.abs(gy).max():.1f}; "
          f"Z std={gz.std():.2f} max|.|={np.abs(gz).max():.1f}")

    # === Roll/pitch plot (separate figure) ===
    # The AHRS-estimated attitude (Euler roll & pitch) over the window. This is
    # the body→NED transform's tilt estimate: if it swings wildly or drifts, the
    # world-vertical projection of accel will be wrong. Gravity was auto-aligned
    # onto -Z, so a level device sits near roll=pitch=0; the AC swing here is the
    # device's real tilt motion as tracked by the filter.
    fig_rp, axrp = plt.subplots(figsize=(13, 4.5))
    axrp.plot(ola_t, ola_roll, color="tab:blue", lw=0.8, label="roll")
    axrp.plot(ola_t, ola_pitch, color="tab:orange", lw=0.8, label="pitch")
    axrp.axhline(0, color="k", lw=0.5, alpha=0.3)
    axrp.set_ylabel("AHRS attitude (deg)")
    axrp.set_xlabel("UTC")
    axrp.grid(True, alpha=0.3)
    axrp.legend(loc="upper right")
    axrp.set_title(
        f"OLA AHRS roll/pitch in window — {OLA_FOLDER.name}\n"
        f"roll mean={np.mean(ola_roll):+.1f}° std={np.std(ola_roll):.1f}°; "
        f"pitch mean={np.mean(ola_pitch):+.1f}° std={np.std(ola_pitch):.1f}°"
    )
    axrp.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=None))
    fig_rp.tight_layout()
    print(f"Window attitude (deg): "
          f"roll mean={np.mean(ola_roll):+.1f} std={np.std(ola_roll):.1f}; "
          f"pitch mean={np.mean(ola_pitch):+.1f} std={np.std(ola_pitch):.1f}")

    # === Summary stats ===
    print("\nWindow stats (vertical accel in m/s²):")
    print(f"  SFY  (w_z - mean)  : std={np.std(sfy_wz_ac):.4f}  "
          f"range=[{sfy_wz_ac.min():+.3f}, {sfy_wz_ac.max():+.3f}]  "
          f"n={sfy_t.size} @ ~{sfy.attrs.get('frequency')} Hz")
    print(f"  OLA  (BP, AHRS)    : std={np.std(ola_az):.4f}  "
          f"range=[{ola_az.min():+.3f}, {ola_az.max():+.3f}]  "
          f"n={ola_t.size} @ ~{res.fs_hz:.1f} Hz")
    print(f"  OLA  (raw, AHRS)   : std={np.std(ola_az_raw):.4f}  "
          f"range=[{ola_az_raw.min():+.3f}, {ola_az_raw.max():+.3f}]")
    print(f"  OLA  (raw acc_x)   : std={np.std(raw_acc_x_window_ac):.4f}  "
          f"range=[{raw_acc_x_window_ac.min():+.3f}, {raw_acc_x_window_ac.max():+.3f}]  "
          f"n={raw_acc_x_t.size} @ 225 Hz")

    plt.show()


if __name__ == "__main__":
    main()
