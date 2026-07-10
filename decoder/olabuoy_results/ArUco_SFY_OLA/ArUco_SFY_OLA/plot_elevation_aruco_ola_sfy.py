#!/usr/bin/env python3
"""Overlay surface elevation from three sources for one wave-tank run:

  * ArUco  — optical ground truth, `CoG_Pos_Z_Rel` from the ArUco batch CSV
             (metres, ~30 Hz). Its timestamps are a *relative* camera clock
             (Date/Time start at 0 and are NOT real UTC), so we align it to the
             buoys by CROSS-CORRELATION against SFY.
  * OLA    — `ola_disp` from the OLA_SFY integrated-timeseries .npz (true UTC).
  * SFY    — `sfy_disp` from the same .npz (true UTC).

The ArUco time offset (and, if the camera Z axis points the other way, its sign)
is chosen to maximise correlation with the SFY displacement. Everything is then
mean-removed and plotted on a shared UTC axis.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import butter, filtfilt, correlate, welch

HERE = Path(__file__).parent

# ----------------------------- user knobs ------------------------------------
TP  = "Tp1"    # peak-period tag, e.g. Tp1 / Tp1p5 / Tp2
RUN = "run1"     # run tag

# ArUco vertical-elevation column. NOTE: the camera's UP axis is Y here, not Z —
# CoG_Pos_Z_Rel is depth (toward/away from the camera, a slow drift with no wave)
# and CoG_Pos_X_Rel is horizontal sway. CoG_Pos_Y_Rel carries the ~0.5 Hz heave
# that matches the buoys. ("_Rel" = relative to the reference marker, so camera
# motion is removed; "_Cam" is the raw camera frame.)
ARUCO_ELEV_COL = "CoG_Pos_Y_Rel"

ARUCO_DIR  = HERE / "ArUco"
OLASFY_DIR = HERE / "OLA_SFY"

XCORR_FS   = 30.0        # common resample rate for the alignment cross-correlation
BAND_HZ    = (0.1, 2.0)  # band-pass (Hz) used ONLY for the alignment xcorr
MAX_SHIFT_S = 180.0      # limit the ArUco<->SFY search to +/- this many seconds
# OLA in the .npz can carry a small residual time offset vs SFY (the OLA<->SFY
# processing offset). Cross-correlate OLA to SFY too and shift it into line so
# the 3-way overlay is meaningful. Bounded to +/- this (< one wave period, so we
# don't lock onto the wrong cycle).
ALIGN_OLA_TO_SFY = True
OLA_MAX_SHIFT_S  = 1.5

# --- Recompute OLA displacement OURSELVES from the raw .dat (4th trace) --------
# To test the OLA post-processing (gravity removal + integration) independently
# of the pre-baked `ola_disp` in the .npz. We decode the raw BOOT folder, run the
# ahrs_vertical AHRS (same pipeline as ahrs_example.py), and integrate — so you
# can sweep AHRS settings here and see the effect on the displacement.
OLA_RECOMPUTE    = True
OLA_RAW_BOOT     = HERE.parent.parent / "OLA_data" / "OLA_data" / "BOOT_000105"
OLA_DECODER_DIR  = Path(r"C:/projects/2026_ola_logger_imu_gps/decoder")
OLA_WINDOW_PAD_S = 30.0      # lead-in before the window for AHRS/integration settling
# AHRS method + band + per-method kwargs (compute_vertical_motion_* in ahrs_vertical):
#   "mahony"   -> kp, ki, motion_gate_threshold, accel_reject_band, calibrate_gyro_bias
#   "madgwick" -> beta, motion_gate_threshold
#   "lowpass"  -> gravity_cutoff_hz
#   "savgol"   -> savgol_window_seconds, savgol_polyorder
OLA_AHRS_METHOD  = "mahony"
# Band corners for the accel processing before integration. "adaptive" finds
# each corner from the cumulative accel spectrum (low: bottom-up 1%-energy
# point; high: top-down 0.3%) — see adaptive_corners in compare_accel_sfy_ola.
# This cuts the sub-wave tilt-leak (which 1/w^2 integration otherwise amplifies
# into a huge low-frequency displacement swing) without hand-tuning per Tp.
# Either can also be pinned to a fixed number (e.g. 0.2).
OLA_LOW_HZ       = "adaptive"
OLA_HIGH_HZ      = "adaptive"
# Knobs for the adaptive corners (passed to adaptive_corners; this script's
# values override the defaults in compare_accel_sfy_ola):
ADAPTIVE_ENERGY_FRAC_LOW  = 0.05        # bottom-up fraction for the low corner
ADAPTIVE_ENERGY_FRAC_HIGH = 0.003   # top-down fraction for the high corner —
                                    # smaller than the low one so the roll-off
                                    # tail / harmonics shoulder isn't shaved
ADAPTIVE_MIN_HZ       = 0.03   # low corner: never below this (DC/detrend region)
ADAPTIVE_MAX_PEAK_FRAC = 0.6   # low corner: never above this fraction of f_peak
ADAPTIVE_MIN_PEAK_MULT = 1.5   # high corner: never below this multiple of f_peak
# Band-pass / inter-stage HP filter order for the recompute. Zero-phase
# application SQUARES the magnitude response, so a low order droops well into
# the passband (3rd order: ~20% loss at 1.25x corner). Order 6 keeps the
# passband flat to within ~3% beyond 1.33x corner. Applied as SOS for
# numerical stability.
FILTER_ORDER = 6
OLA_AHRS_KWARGS  = dict(kp=8.0, ki=1.25, motion_gate_threshold=1e9,
                        calibrate_gyro_bias=False)
# The buoy `disp` is high-passed (integration removes drift); the ArUco optical
# position is NOT and carries a large slow drift (the buoy translating across
# the tank). High-pass ALL THREE at this corner so we compare wave-band surface
# elevation apples-to-apples. Set below the wave frequency (Tp2 = 0.5 Hz).
ELEV_HP_HZ = 0.10
PLOT_FS    = 30.0        # uniform rate the plotted/compared signals are put on

# --- SWL (waterline) lever-arm correction --------------------------------------
# The three sensors sit at different heights along the buoy axis. A sensor at
# distance d above the waterline point measures z_sensor = z_SWL + d*cos(tilt),
# so its heave is contaminated by a rectified -d*theta^2/2 signal at 2x the roll
# frequency (with tilt 20-40 deg and d=6 cm: 0.4-1.4 cm — significant vs ~3 cm
# waves). Correct every trace to the waterline: z_SWL = z_sensor - d*cos(tilt),
# using the OLA AHRS (accel-corrected gyro) attitude — the buoy is rigid, so one
# attitude serves all sensors. Offsets in metres, + = above SWL.
SWL_CORRECT = True
SWL_OFFSET_M = {
    "ola":   -0.06,   # OLA IMU ~6 cm below the waterline
    "sfy":    0.00,   # SFY ~at the waterline (reference)
    "aruco": +0.01,   # ArUco rotation centre ~1 cm above the waterline
}


def _bandpass(x, fs, lo, hi):
    b, a = butter(3, [lo / (fs / 2), hi / (fs / 2)], btype="bandpass")
    return filtfilt(b, a, x)


def _hp_resampled(t, z, fs, hp_hz):
    """Resample (t, z) to a uniform grid at fs, mean-remove, zero-phase high-pass.
    Returns (t_uniform, z_hp)."""
    tu = np.arange(t[0], t[-1], 1.0 / fs)
    zu = np.interp(tu, t, z - np.nanmean(z))
    b, a = butter(3, hp_hz / (fs / 2), btype="highpass")
    return tu, filtfilt(b, a, zu)


def _zc_extrema(x):
    """Per-zero-crossing extrema of a zero-mean signal.

    Splits `x` at its zero crossings; each excursion above zero contributes its
    maximum (crest), each excursion below zero its minimum (trough). Returns
    (maxima, minima) arrays — the standard zero-crossing wave analysis.
    """
    s = np.sign(x)
    s[s == 0] = 1
    segs = np.split(x, np.where(np.diff(s) != 0)[0] + 1)
    maxima, minima = [], []
    for seg in segs:
        if len(seg) < 2:
            continue
        if seg[0] > 0:
            maxima.append(seg.max())
        else:
            minima.append(seg.min())
    return np.array(maxima), np.array(minima)


def _qq(a, b, n=None):
    """Matched quantiles of two samples (ascending). Interpolates both to a
    common quantile grid so different sample counts can be QQ-plotted."""
    n = min(len(a), len(b)) if n is None else n
    probs = (np.arange(n) + 0.5) / n
    return np.quantile(a, probs), np.quantile(b, probs)


def _best_shift(ref_t, ref_z, sig_t, sig_z, fs, band, max_shift, try_flip):
    """Cross-correlate `sig` onto `ref` (both mean-removed, band-passed at `band`).
    Returns (shift_s, sign, score): add shift_s to sig_t (and multiply sig_z by
    sign) to best match ref. Search bounded to +/- max_shift seconds."""
    rg = np.arange(ref_t[0], ref_t[-1], 1.0 / fs)
    sg = np.arange(sig_t[0], sig_t[-1], 1.0 / fs)
    r_bp = _bandpass(np.interp(rg, ref_t, ref_z - np.nanmean(ref_z)), fs, *band)
    s_bp = _bandpass(np.interp(sg, sig_t, sig_z - np.nanmean(sig_z)), fs, *band)
    base = ref_t[0] - sig_t[0]           # grid-index 0 offset between the two axes
    best = None
    for sign in ((+1.0, -1.0) if try_flip else (+1.0,)):
        cc = correlate(r_bp, sign * s_bp, mode="full")
        lags = np.arange(-(len(s_bp) - 1), len(r_bp)) / fs + base
        ok = np.abs(lags) <= max_shift
        if not ok.any():
            continue
        k = np.argmax(cc[ok])
        score = cc[ok][k] / (np.linalg.norm(r_bp) * np.linalg.norm(s_bp))
        if best is None or score > best[2]:
            best = (lags[ok][k], sign, score)
    return best


def _import_ola_decoder():
    import sys
    if str(OLA_DECODER_DIR) not in sys.path:
        sys.path.insert(0, str(OLA_DECODER_DIR))
    try:
        from decoder import decode_file, load_data_as_arrays  # noqa
        import ahrs_vertical  # noqa
    except Exception as exc:
        raise SystemExit(f"Could not import OLA decoder from {OLA_DECODER_DIR}: "
                         f"{type(exc).__name__}: {exc}")
    try:
        from loguru import logger as _L
        _L.remove()
    except Exception:
        pass
    return decode_file, load_data_as_arrays, ahrs_vertical


def _dat_time(p: Path):
    import re
    m = re.search(r"TIME_(\d{8})T(\d{6})", p.name)
    if not m:
        return None
    dstr, tstr = m.groups()
    return np.datetime64(f"{dstr[:4]}-{dstr[4:6]}-{dstr[6:8]}T"
                         f"{tstr[:2]}:{tstr[2:4]}:{tstr[4:6]}")


def _select_dat_files(folder: Path, ws, we, pad_s):
    """.dat files whose time-range overlaps [ws - pad, we] (files are ~15 min)."""
    files = sorted((f for f in folder.glob("DATA_BOOT_*.dat") if _dat_time(f) is not None),
                   key=_dat_time)
    times = [_dat_time(f) for f in files]
    lo = np.datetime64(ws) - np.timedelta64(int(pad_s), "s")
    hi = np.datetime64(we)
    sel = []
    for i, (f, t) in enumerate(zip(files, times)):
        t_end = times[i + 1] if i + 1 < len(times) else t + np.timedelta64(1200, "s")
        if t < hi and t_end > lo:
            sel.append(f)
    return sel


def recompute_ola_disp(boot, ws, we, pad_s, method, low_hz, high_hz, kwargs):
    """Decode raw BOOT, run the AHRS, integrate -> (t_ns, disp, lo, hi) clipped
    to the window.

    New accel-processing procedure (mirrors compare_accel_sfy_ola.py): the AHRS
    world-vertical accel is band-passed with a TRUE zero-phase Butterworth whose
    corners are found adaptively from the cumulative accel spectrum (low:
    bottom-up 1%-energy point — cuts the tilt-leak floor that 1/w^2 integration
    would otherwise amplify; high: top-down 0.3%). The filtering and the double
    integration (with the same low corner as zero-phase high-pass between
    stages) all happen on the PADDED signal; the window clip comes last."""
    decode_file, load_data_as_arrays, ahrs = _import_ola_decoder()
    files = _select_dat_files(boot, ws, we, pad_s)
    if not files:
        raise SystemExit(f"No raw .dat overlapping {ws}..{we} in {boot}")
    print(f"  decoding {len(files)} raw file(s): {[f.name for f in files]}")
    parts = [load_data_as_arrays(decode_file(f, allow_no_pps=True)["file"]) for f in files]
    ola = {}
    for k in parts[0]:
        v0 = parts[0][k]
        if isinstance(v0, np.ndarray):
            try:
                ola[k] = np.concatenate([p[k] for p in parts if isinstance(p[k], np.ndarray)])
            except ValueError:
                ola[k] = v0
        else:
            ola[k] = v0

    gm = np.asarray(ola["gnss_micros_unwrapped"], float)
    gp = np.asarray(ola["gnss_posix"], float)
    ok = gp > 1e9
    slope, intercept = np.polyfit(gm[ok], gp[ok], 1)
    im = np.asarray(ola["imu_micros_unwrapped"], float)
    utc = slope * im + intercept
    A = np.datetime64(ws).astype("M8[ns]").astype("int64") / 1e9
    B = np.datetime64(we).astype("M8[ns]").astype("int64") / 1e9
    sel = (utc >= A - pad_s) & (utc <= B)
    n = len(im)
    w = dict(ola)
    for k, v in ola.items():
        if isinstance(v, np.ndarray) and v.size == n:
            w[k] = v[sel]

    fn = {"mahony":   ahrs.compute_vertical_motion_mahony,
          "madgwick": ahrs.compute_vertical_motion,
          "lowpass":  ahrs.compute_vertical_motion_lowpass_gravity,
          "savgol":   ahrs.compute_vertical_motion_savgol_detrend}[method]
    # Band edges only steer library outputs we don't use; pass nominals if adaptive.
    lib_low  = low_hz  if isinstance(low_hz,  (int, float)) else 0.05
    lib_high = high_hz if isinstance(high_hz, (int, float)) else 2.5
    vmot = fn(w, low_hz=lib_low, high_hz=lib_high, **kwargs)

    # --- new accel processing: adaptive-corner band-pass on the PADDED accel ---
    from compare_accel_sfy_ola import adaptive_corners
    if low_hz == "adaptive" or high_hz == "adaptive":
        ad_lo, ad_hi = adaptive_corners(
            vmot.accel_z_up_raw, vmot.fs_hz,
            frac_low=ADAPTIVE_ENERGY_FRAC_LOW,
            frac_high=ADAPTIVE_ENERGY_FRAC_HIGH,
            min_hz=ADAPTIVE_MIN_HZ,
            max_peak_frac=ADAPTIVE_MAX_PEAK_FRAC,
            min_peak_mult=ADAPTIVE_MIN_PEAK_MULT)
    lo = ad_lo if low_hz  == "adaptive" else float(low_hz)
    hi = ad_hi if high_hz == "adaptive" else float(high_hz)
    fs = vmot.fs_hz
    from scipy.signal import sosfiltfilt
    sos_bp = butter(FILTER_ORDER, [lo / (fs / 2), hi / (fs / 2)],
                    btype="bandpass", output="sos")
    acc_bp = sosfiltfilt(sos_bp, vmot.accel_z_up_raw)

    # --- double integration on the padded signal, HP(lo) between stages ---
    try:
        from scipy.integrate import cumulative_trapezoid as cumtrapz
    except ImportError:
        from scipy.integrate import cumtrapz
    sos_hp = butter(FILTER_ORDER, lo / (fs / 2), btype="highpass", output="sos")
    vel = cumtrapz(acc_bp, dx=1.0 / fs, initial=0.0)
    vel = sosfiltfilt(sos_hp, vel)
    disp = cumtrapz(vel, dx=1.0 / fs, initial=0.0)
    disp = sosfiltfilt(sos_hp, disp)

    utc0 = utc[sel][0]
    t_utc = utc0 + vmot.t * (slope / 1e-6)          # MCU-seconds -> UTC seconds
    m = (t_utc >= A) & (t_utc <= B)                 # clip LAST
    # cos(tilt) from the AHRS attitude — used for the SWL lever-arm correction
    # (one attitude serves all sensors on the rigid buoy).
    cos_tilt = (np.cos(np.deg2rad(vmot.roll_deg)) *
                np.cos(np.deg2rad(vmot.pitch_deg)))[m]
    return (t_utc[m] * 1e9).astype(np.int64), disp[m], lo, hi, cos_tilt


def _find_run(directory: Path, pattern: str, kind: str) -> Path:
    hits = sorted(directory.glob(pattern))
    if not hits:
        raise SystemExit(f"No {kind} match for {pattern!r} in {directory}")
    if len(hits) > 1:
        print(f"  ({kind}: {len(hits)} matches, using {hits[0].name})")
    return hits[0]


def main() -> None:
    # --- locate the two inputs for this Tp/run ---
    aruco_run = _find_run(ARUCO_DIR, f"JONSWAP_{TP}_{RUN}_*", "ArUco run")
    aruco_csv = aruco_run / "processed_buoy_data_batch.csv"
    npz_path  = _find_run(OLASFY_DIR, f"{TP}_{RUN}_*.npz", "OLA/SFY npz")
    print(f"ArUco : {aruco_csv.relative_to(HERE)}")
    print(f"OLA/SFY: {npz_path.relative_to(HERE)}")

    # --- ArUco: relative-time optical elevation ---
    df = pd.read_csv(aruco_csv)
    a_t = df["Time_Rel_Sec"].to_numpy(float)          # seconds from camera start
    a_z = df[ARUCO_ELEV_COL].to_numpy(float)          # metres (camera up-axis = Y)
    a_z = a_z - np.nanmean(a_z)

    # --- OLA + SFY: true-UTC displacement ---
    d = np.load(npz_path, allow_pickle=True)
    sfy_ns = d["sfy_utc_ns"].astype(np.int64)
    sfy_z  = d["sfy_disp"].astype(float)
    ola_ns = d["ola_utc_ns"].astype(np.int64)
    ola_z  = d["ola_disp"].astype(float)
    t0_ns  = sfy_ns[0]                                # anchor everything to SFY start
    sfy_t  = (sfy_ns - t0_ns) / 1e9                   # seconds from SFY start
    ola_t  = (ola_ns - t0_ns) / 1e9

    # --- align ArUco to SFY by cross-correlation (per-run: the camera clock
    #     restarts at 0 each recording, so its offset to buoy-UTC is arbitrary
    #     and must be found for every run). Also finds the camera Z-axis sign. ---
    fs = XCORR_FS
    shift_s, sign, score = _best_shift(
        sfy_t, sfy_z, a_t, a_z, fs, BAND_HZ, MAX_SHIFT_S, try_flip=True)
    print(f"ArUco alignment: shift = {shift_s:+.3f} s, sign = {sign:+.0f}, "
          f"corr = {score:+.3f}  (sign -1 means the camera axis is inverted)")
    a_z = sign * a_z
    a_t = a_t + shift_s

    # --- align OLA to SFY too (small residual OLA<->SFY processing offset) ---
    ola_shift = 0.0
    if ALIGN_OLA_TO_SFY:
        ola_shift, _, oscore = _best_shift(
            sfy_t, sfy_z, ola_t, ola_z, fs, BAND_HZ, OLA_MAX_SHIFT_S, try_flip=False)
        print(f"OLA alignment  : shift = {ola_shift:+.3f} s, corr = {oscore:+.3f} "
              f"(residual OLA<->SFY offset; +-{OLA_MAX_SHIFT_S}s search)")
        ola_t = ola_t + ola_shift

    # --- optionally recompute OLA displacement OURSELVES from the raw .dat ---
    r_tu = r_hp = None
    if OLA_RECOMPUTE:
        print(f"\nRecomputing OLA from raw ({OLA_RAW_BOOT.name}; AHRS={OLA_AHRS_METHOD} "
              f"[{OLA_LOW_HZ},{OLA_HIGH_HZ}]Hz {OLA_AHRS_KWARGS})")
        r_ns, r_z, r_lo, r_hi, r_cos_tilt = recompute_ola_disp(
            OLA_RAW_BOOT, str(d["window_start"]), str(d["window_end"]),
            OLA_WINDOW_PAD_S, OLA_AHRS_METHOD, OLA_LOW_HZ, OLA_HIGH_HZ, OLA_AHRS_KWARGS)
        r_t = (r_ns - t0_ns) / 1e9
        rshift, _, rscore = _best_shift(
            sfy_t, sfy_z, r_t, r_z, fs, BAND_HZ, OLA_MAX_SHIFT_S, try_flip=False)
        r_t = r_t + rshift
        print(f"OLA(recomputed) alignment: shift = {rshift:+.3f} s, corr = {rscore:+.3f}")

        # --- SWL lever-arm correction: refer every trace to the waterline ---
        # z_SWL = z_sensor - d*cos(tilt), with the AHRS attitude (on the
        # recompute's aligned time base) interpolated onto each trace's times.
        # Constant parts of d*cos(tilt) vanish in the later mean-removal/HP;
        # what matters is the rectified -d*theta^2/2 wobble at 2x roll freq.
        if SWL_CORRECT:
            ct = lambda t: np.interp(t, r_t, r_cos_tilt)
            corr_std = SWL_OFFSET_M["ola"] * (r_cos_tilt - r_cos_tilt.mean())
            print(f"SWL correction: OLA d={SWL_OFFSET_M['ola']*100:+.0f} cm "
                  f"(wobble std {corr_std.std()*1000:.1f} mm), "
                  f"ArUco d={SWL_OFFSET_M['aruco']*100:+.0f} cm, "
                  f"SFY d={SWL_OFFSET_M['sfy']*100:+.0f} cm")
            r_z   = r_z   - SWL_OFFSET_M["ola"]   * r_cos_tilt
            ola_z = ola_z - SWL_OFFSET_M["ola"]   * ct(ola_t)
            a_z   = a_z   - SWL_OFFSET_M["aruco"] * ct(a_t)
            sfy_z = sfy_z - SWL_OFFSET_M["sfy"]   * ct(sfy_t)

        r_tu, r_hp = _hp_resampled(r_t, r_z, PLOT_FS, ELEV_HP_HZ)

    # --- high-pass all three at the same corner (drift-comparable), on UTC ---
    a_tu, a_hp = _hp_resampled(a_t,   a_z,   PLOT_FS, ELEV_HP_HZ)
    o_tu, o_hp = _hp_resampled(ola_t, ola_z, PLOT_FS, ELEV_HP_HZ)
    s_tu, s_hp = _hp_resampled(sfy_t, sfy_z, PLOT_FS, ELEV_HP_HZ)
    to_utc = lambda tu: (t0_ns + (tu * 1e9).astype(np.int64)).astype("datetime64[ns]")

    # correlation of each buoy vs ArUco on the mutual overlap (wave band)
    def _corr(btu, bhp):
        g0 = max(a_tu[0], btu[0]); g1 = min(a_tu[-1], btu[-1])
        gg = np.arange(g0, g1, 1.0 / PLOT_FS)
        return np.corrcoef(np.interp(gg, a_tu, a_hp), np.interp(gg, btu, bhp))[0, 1]
    c_sfy, c_ola = _corr(s_tu, s_hp), _corr(o_tu, o_hp)
    print(f"  corr(ArUco, SFY) = {c_sfy:+.3f}   corr(ArUco, OLA-npz) = {c_ola:+.3f}"
          + (f"   corr(ArUco, OLA-recomp) = {_corr(r_tu, r_hp):+.3f}" if r_tu is not None else ""))
    print(f"  wave-band std (cm): ArUco={a_hp.std()*100:.1f}  "
          f"OLA-npz={o_hp.std()*100:.1f}  SFY={s_hp.std()*100:.1f}"
          + (f"  OLA-recomp={r_hp.std()*100:.1f}" if r_tu is not None else "")
          + f"   (HP {ELEV_HP_HZ} Hz)")

    # --- plot: timeseries (top) + wave spectrum (bottom) ---
    series = [(a_tu, a_hp, "k",        1.4, "-", f"ArUco {ARUCO_ELEV_COL} (optical)"),
              (o_tu, o_hp, "tab:blue", 0.9, "-", "OLA ola_disp (npz)"),
              (s_tu, s_hp, "tab:red",  0.9, "-", "SFY sfy_disp")]
    if r_tu is not None:
        series.append((r_tu, r_hp, "tab:green", 1.1, "--",
                       f"OLA recomputed ({OLA_AHRS_METHOD}, "
                       f"band [{r_lo:.3f},{r_hi:.2f}] Hz)"))

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 9))

    # top: elevation timeseries
    for tu, hp, c, lw, ls, lab in series:
        ax.plot(to_utc(tu), hp * 100, color=c, lw=lw, ls=ls, label=lab)
    ax.set_ylabel("surface elevation (cm)")
    ax.set_xlabel("UTC")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_title(f"Surface elevation — {TP} {RUN}   "
                 f"(ArUco shifted {shift_s:+.2f}s, sign {sign:+.0f}, HP {ELEV_HP_HZ} Hz; "
                 f"corr ArUco-SFY={c_sfy:.2f}, ArUco-OLA={c_ola:.2f})")

    # bottom: wave (elevation) spectrum via Welch — all on the PLOT_FS grid
    f_peak = None
    for tu, hp, c, lw, ls, lab in series:
        nps = int(min(2048, len(hp)))
        f, psd = welch(hp * 100, fs=PLOT_FS, nperseg=nps)   # cm -> PSD in cm^2/Hz
        ax2.plot(f, psd, color=c, lw=1.3, ls=ls, label=lab.split(" (")[0])
        if lab.startswith("SFY"):                            # peak from SFY (buoy ref)
            f_peak = f[1:][np.argmax(psd[1:])]
    ax2.set_xlim(0, 2.5)
    ax2.set_xlabel("frequency (Hz)")
    ax2.set_ylabel("elevation PSD (cm²/Hz)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")
    if f_peak:
        ax2.axvline(f_peak, color="0.5", ls=":", lw=1)
        ax2.set_title(f"Wave spectrum (Welch)   peak {f_peak:.3f} Hz "
                      f"(Tp≈{1/f_peak:.2f} s)")

    fig.tight_layout()
    out = HERE / f"elevation_{TP}_{RUN}.png"
    fig.savefig(out, dpi=90)
    print(f"\nSaved {out}")

    # --- QQ diagram: crest/trough populations, ArUco (x) vs OLA (y) ---
    # Zero-crossing analysis on the mutual overlap of the HP'd signals: each
    # up-excursion contributes its maximum (crest), each down-excursion its
    # |minimum| (trough). Sorted ascending and quantile-matched. Points on the
    # 1:1 line = identical wave-height distributions; a slope !=1 = amplitude
    # scale difference; curvature at the top = disagreement on extreme waves.
    ola_qq_t, ola_qq_z, ola_qq_lab = (
        (r_tu, r_hp, f"OLA recomputed ({OLA_AHRS_METHOD})") if r_tu is not None
        else (o_tu, o_hp, "OLA ola_disp (npz)"))
    g0 = max(a_tu[0], ola_qq_t[0]); g1 = min(a_tu[-1], ola_qq_t[-1])
    a_seg = a_hp[(a_tu >= g0) & (a_tu <= g1)]
    o_seg = ola_qq_z[(ola_qq_t >= g0) & (ola_qq_t <= g1)]
    a_max, a_min = _zc_extrema(a_seg)
    o_max, o_min = _zc_extrema(o_seg)
    print(f"QQ extrema counts: ArUco {len(a_max)}+/{len(a_min)}-  "
          f"OLA {len(o_max)}+/{len(o_min)}-")

    qx_max, qy_max = _qq(a_max, o_max)
    qx_min, qy_min = _qq(np.abs(a_min), np.abs(o_min))

    fig3, axq = plt.subplots(figsize=(7.5, 7.5))
    axq.scatter(qx_max * 100, qy_max * 100, s=14, color="tab:blue",
                label=f"crests / maxima (n={len(a_max)}/{len(o_max)})")
    axq.scatter(qx_min * 100, qy_min * 100, s=14, color="tab:orange",
                marker="s", label=f"|troughs| / |minima| (n={len(a_min)}/{len(o_min)})")
    lim = 1.05 * max(qx_max.max(), qy_max.max(),
                     qx_min.max(), qy_min.max()) * 100
    axq.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6, label="1:1")
    axq.set_xlim(0, lim); axq.set_ylim(0, lim)
    axq.set_aspect("equal")
    axq.set_xlabel("ArUco elevation extrema (cm)")
    axq.set_ylabel(f"{ola_qq_lab} extrema (cm)")
    axq.grid(True, alpha=0.3)
    axq.legend(loc="upper left")
    axq.set_title(f"QQ: zero-crossing crest/trough distributions — {TP} {RUN}\n"
                  f"ArUco vs {ola_qq_lab}")
    fig3.tight_layout()
    out3 = HERE / f"elevation_qq_{TP}_{RUN}.png"
    fig3.savefig(out3, dpi=90)
    print(f"Saved {out3}")

    plt.show()


if __name__ == "__main__":
    main()
