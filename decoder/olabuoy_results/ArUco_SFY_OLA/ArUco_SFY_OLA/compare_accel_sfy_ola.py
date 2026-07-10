#!/usr/bin/env python3
"""Compare vertical ACCELERATION: SFY's onboard-fused w_z vs our own AHRS
processing of the raw OLA data. No ArUco, no integration — this isolates the
AHRS + gravity-removal step from all the displacement/filter choices.

  * SFY — `sfy_accel` (w_z - window mean) from the OLA_SFY .npz, on its own UTC.
  * OLA — recomputed by us from the raw BOOT .dat: decode -> GNSS->UTC mapping
    (incl. the MCU-clock #4 timebase fix) -> compute_vertical_motion_<METHOD>
    from ahrs_vertical -> accel_z_up_raw (world-vertical, gravity removed,
    NOT band-passed).

Both are then aligned (small OLA<->SFY offset found by cross-correlation) and
overlaid: full-window timeseries, a zoom panel, and the acceleration spectrum.
If the AHRS is healthy these should align crest-for-crest.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import correlate, welch

HERE = Path(__file__).parent

# ----------------------------- user knobs ------------------------------------
TP  = "Tp1p5"    # peak-period tag, e.g. Tp1 / Tp1p5 / Tp2
RUN = "run3"     # run tag

OLASFY_DIR = HERE / "OLA_SFY"

# Raw OLA data + decoder (all runs came from one continuous BOOT recording).
OLA_RAW_BOOT    = HERE.parent.parent / "OLA_data" / "OLA_data" / "BOOT_000105"
OLA_DECODER_DIR = Path(r"C:/projects/2026_ola_logger_imu_gps/decoder")
OLA_WINDOW_PAD_S = 30.0     # AHRS settling lead-in before the window

# AHRS settings to test (compute_vertical_motion_* in ahrs_vertical):
#   "mahony"   -> kp, ki, motion_gate_threshold, accel_reject_band, calibrate_gyro_bias
#   "madgwick" -> beta, motion_gate_threshold
#   "lowpass"  -> gravity_cutoff_hz
#   "savgol"   -> savgol_window_seconds, savgol_polyorder
OLA_AHRS_METHOD = "mahony"
OLA_AHRS_KWARGS = dict(kp=8.0, ki=1.25, motion_gate_threshold=1e9,
                       calibrate_gyro_bias=False)
# AHRS band edges. These define the band-passed output `accel_z_up`, which is
# the trace to compare against SFY: SFY's w_z is likewise band-limited by its
# onboard FIR, so band-passed-vs-band-passed is the fair comparison. The raw
# (unfiltered) world-z is plotted alongside as reference.
#
# OLA_LOW_HZ can be a number (fixed corner) or "adaptive": the corner is then
# set where the cumulative accel spectrum (integrated from f=0 upward) reaches
# ADAPTIVE_ENERGY_FRAC of the total area. Because the accel spectrum is
# strongly wave-dominated, the sub-wave leak floor contributes only a tiny
# fraction of the total energy, so this point lands just below the wave band —
# for any sea state / Tp — cutting the tilt-leak without hand-tuning. Safety
# rails: clamped to [ADAPTIVE_MIN_HZ, ADAPTIVE_MAX_PEAK_FRAC * f_peak] so it
# can never eat into the wave band.
# OLA_HIGH_HZ works the same way with "adaptive": the cumulative spectrum is
# integrated from the TOP (Nyquist) downward and the corner set where it passes
# ADAPTIVE_ENERGY_FRAC of the total — i.e. the upper band edge above which only
# ~1% of the energy (sensor noise) lives. Clamped to >= ADAPTIVE_MIN_PEAK_MULT
# x f_peak so it can never cut into the wave band.
OLA_LOW_HZ  = "adaptive"
OLA_HIGH_HZ = "adaptive"
ADAPTIVE_ENERGY_FRAC_LOW  = 0.05    # bottom-up fraction for the low corner
ADAPTIVE_ENERGY_FRAC_HIGH = 0.003   # top-down fraction for the high corner —
                                    # smaller than the low one so the roll-off
                                    # tail / harmonics shoulder isn't shaved
ADAPTIVE_MIN_HZ       = 0.03   # low corner: never below this (DC/detrend region)
ADAPTIVE_MAX_PEAK_FRAC = 0.6   # low corner: never above this fraction of f_peak
ADAPTIVE_MIN_PEAK_MULT = 1.5   # high corner: never below this multiple of f_peak

# Band-pass filter order. Applied zero-phase (forward+backward), which SQUARES
# the magnitude response — a 3rd-order Butterworth then still attenuates ~20%
# at 1.25x the corner, i.e. well into the passband. Order 6 confines the droop
# to a narrow region near the corner (<3% by 1.33x). Implemented as second-order
# sections (SOS): high-order (b,a) coefficients are numerically unstable at low
# normalized corners.
FILTER_ORDER = 6

# NOTE: no comparison filtering is applied — both signals are already processed
# (SFY: onboard fusion + FIR; OLA: our AHRS pipeline). They are only mean-removed
# and resampled to a common grid for the xcorr/plot, then plotted AS-IS.
PLOT_FS     = 30.0          # common uniform grid for compare/spectrum
OLA_MAX_SHIFT_S = 1.5       # bound for the OLA<->SFY xcorr alignment
ZOOM_S = 30.0               # length of the zoom panel (starts mid-window)


def adaptive_corners(x, fs, frac_low=None, frac_high=None,
                     min_hz=None, max_peak_frac=None, min_peak_mult=None):
    """Adaptive band edges from the cumulative accel spectrum.

    Low corner: integrate the Welch PSD from f=0 upward; the corner is where the
    cumulative energy first exceeds `frac_low` of the total (clamped to
    [min_hz, max_peak_frac * f_peak]).
    High corner: integrate from the top (Nyquist) downward; the corner is where
    the top-down cumulative energy exceeds `frac_high` (clamped to
    >= min_peak_mult * f_peak). Returns (low_hz, high_hz).

    All knobs default to this module's ADAPTIVE_* constants; callers (e.g.
    plot_elevation_aruco_ola_sfy.py) may override any of them per-call.
    """
    frac_low  = ADAPTIVE_ENERGY_FRAC_LOW  if frac_low  is None else frac_low
    frac_high = ADAPTIVE_ENERGY_FRAC_HIGH if frac_high is None else frac_high
    min_hz = ADAPTIVE_MIN_HZ if min_hz is None else min_hz
    max_peak_frac = ADAPTIVE_MAX_PEAK_FRAC if max_peak_frac is None else max_peak_frac
    min_peak_mult = ADAPTIVE_MIN_PEAK_MULT if min_peak_mult is None else min_peak_mult

    f, psd = welch(x - x.mean(), fs=fs, nperseg=int(min(4096, len(x))))
    cum = np.cumsum(psd)
    cum /= cum[-1]
    f_peak = f[1:][int(np.argmax(psd[1:]))]

    f_lo = f[int(np.searchsorted(cum, frac_low))]
    lo = float(np.clip(f_lo, min_hz, max_peak_frac * f_peak))

    f_hi = f[int(np.searchsorted(cum, 1.0 - frac_high))]  # top-down frac point
    hi = float(np.clip(f_hi, min_peak_mult * f_peak, 0.45 * fs))

    print(f"  adaptive corners: {frac_low*100:.1f}%-up = {f_lo:.3f} Hz, "
          f"{frac_high*100:.1f}%-down = {f_hi:.3f} Hz, peak = {f_peak:.3f} Hz "
          f"-> band = [{lo:.3f}, {hi:.3f}] Hz")
    return lo, hi


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


def recompute_ola_accel(boot, ws, we, pad_s):
    """Decode raw BOOT + run the AHRS -> (t_ns, accel_raw, accel_bp) clipped to
    the window. accel_raw = world-vertical accel, gravity removed, unfiltered;
    accel_bp = the same after the AHRS band-pass [OLA_LOW_HZ, OLA_HIGH_HZ]."""
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
          "savgol":   ahrs.compute_vertical_motion_savgol_detrend}[OLA_AHRS_METHOD]
    # The band edges only steer the library's internal outputs we don't use;
    # pass nominal values when a corner is adaptive.
    lib_low  = OLA_LOW_HZ  if isinstance(OLA_LOW_HZ,  (int, float)) else 0.05
    lib_high = OLA_HIGH_HZ if isinstance(OLA_HIGH_HZ, (int, float)) else 2.5
    vmot = fn(w, low_hz=lib_low, high_hz=lib_high, **OLA_AHRS_KWARGS)

    # NOTE: the library's vmot.accel_z_up is NOT band-passed at low_hz — it is
    # only polynomial-detrended + low-passed at high_hz (low_hz is applied only
    # inside the displacement integrator). For an SFY-comparable accel we apply
    # a true zero-phase band-pass ourselves, on the PADDED signal (edge
    # transients land in the pad), and clip to the window last. Corners are
    # either fixed or found adaptively from the cumulative accel spectrum.
    if OLA_LOW_HZ == "adaptive" or OLA_HIGH_HZ == "adaptive":
        ad_lo, ad_hi = adaptive_corners(vmot.accel_z_up_raw, vmot.fs_hz)
    low_hz  = ad_lo if OLA_LOW_HZ  == "adaptive" else float(OLA_LOW_HZ)
    high_hz = ad_hi if OLA_HIGH_HZ == "adaptive" else float(OLA_HIGH_HZ)
    from scipy.signal import butter, sosfiltfilt
    sos = butter(FILTER_ORDER, [low_hz / (vmot.fs_hz / 2), high_hz / (vmot.fs_hz / 2)],
                 btype="bandpass", output="sos")
    accel_bp = sosfiltfilt(sos, vmot.accel_z_up_raw)

    utc0 = utc[sel][0]
    t_utc = utc0 + vmot.t * (slope / 1e-6)     # MCU-clock seconds -> UTC seconds (#4 fix)
    m = (t_utc >= A) & (t_utc <= B)
    return ((t_utc[m] * 1e9).astype(np.int64),
            vmot.accel_z_up_raw[m], accel_bp[m], low_hz, high_hz,
            vmot.roll_deg[m], vmot.pitch_deg[m])


def main() -> None:
    # --- SFY accel from the npz ---
    hits = sorted(OLASFY_DIR.glob(f"{TP}_{RUN}_*.npz"))
    if not hits:
        raise SystemExit(f"No npz for {TP}_{RUN} in {OLASFY_DIR}")
    npz = hits[0]
    print(f"OLA/SFY npz: {npz.name}")
    d = np.load(npz, allow_pickle=True)
    sfy_ns = d["sfy_utc_ns"].astype(np.int64)
    sfy_a  = d["sfy_accel"].astype(float)      # w_z - window mean (gravity removed)
    ws, we = str(d["window_start"]), str(d["window_end"])
    print(f"window: {ws} .. {we}")

    # --- OLA accel recomputed from raw ---
    print(f"\nRecomputing OLA accel from raw ({OLA_RAW_BOOT.name}; "
          f"AHRS={OLA_AHRS_METHOD} {OLA_AHRS_KWARGS})")
    (ola_ns, ola_raw, ola_flt, low_hz_used, high_hz_used,
     ola_roll, ola_pitch) = recompute_ola_accel(
        OLA_RAW_BOOT, ws, we, OLA_WINDOW_PAD_S)

    # --- common grid; NO further filtering (signals are already processed) ---
    # The comparison trace is the AHRS band-passed accel_z_up (SFY's w_z is
    # likewise band-limited onboard); the raw world-z rides along as reference.
    t0_ns = sfy_ns[0]
    st = (sfy_ns - t0_ns) / 1e9
    ot = (ola_ns - t0_ns) / 1e9
    fs = PLOT_FS
    sg = np.arange(st[0], st[-1], 1 / fs)
    og = np.arange(ot[0], ot[-1], 1 / fs)
    s_bp  = np.interp(sg, st, sfy_a - sfy_a.mean())
    o_bp  = np.interp(og, ot, ola_flt - ola_flt.mean())     # band-passed (compare this)
    o_raw = np.interp(og, ot, ola_raw - ola_raw.mean())     # unfiltered (reference)

    # --- align OLA to SFY (small residual offset) ---
    cc = correlate(s_bp, o_bp, mode="full")
    lags = np.arange(-(len(o_bp) - 1), len(s_bp)) / fs + (sg[0] - og[0])
    okm = np.abs(lags) <= OLA_MAX_SHIFT_S
    k = int(np.argmax(cc[okm]))
    shift = lags[okm][k]
    og = og + shift

    # overlap correlation after alignment
    g0, g1 = max(sg[0], og[0]), min(sg[-1], og[-1])
    gg = np.arange(g0, g1, 1 / fs)
    sv = np.interp(gg, sg, s_bp)
    ov = np.interp(gg, og, o_bp)
    corr = np.corrcoef(sv, ov)[0, 1]
    print(f"\nOLA alignment: shift = {shift:+.3f} s   "
          f"corr(SFY, OLA-bandpassed) = {corr:+.3f}")
    print(f"accel std (m/s2): SFY={s_bp.std():.3f}  OLA-bp={o_bp.std():.3f} "
          f"(ratio {o_bp.std()/s_bp.std():.3f})   OLA-raw={o_raw.std():.3f}")

    to_utc = lambda tu: (t0_ns + (tu * 1e9).astype(np.int64)).astype("datetime64[ns]")

    # --- plot: full timeseries, zoom, spectrum, trim angle ---
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(14, 14))

    ax0.plot(to_utc(og), o_raw, color="0.75", lw=0.6,
             label="OLA world-z raw (unfiltered, reference)")
    ax0.plot(to_utc(sg), s_bp, color="tab:red",  lw=0.8, label="SFY w_z (onboard fusion)")
    adaptive_any = "adaptive" in (OLA_LOW_HZ, OLA_HIGH_HZ)
    ax0.plot(to_utc(og), o_bp, color="tab:blue", lw=0.8, alpha=0.85,
             label=f"OLA AHRS band-passed [{low_hz_used:.3f},{high_hz_used:.2f}] Hz "
                   f"({OLA_AHRS_METHOD}"
                   + (", adaptive corners)" if adaptive_any else ")"))
    ax0.set_ylabel("vert accel (m/s²)")
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax0.grid(True, alpha=0.3); ax0.legend(loc="upper right")
    ax0.set_title(f"Vertical acceleration — {TP} {RUN}   "
                  f"(OLA shift {shift:+.2f}s; corr SFY vs OLA-bp = {corr:.3f})")

    # zoom: ZOOM_S seconds starting mid-window
    zc = (g0 + g1) / 2
    zm = (gg >= zc) & (gg <= zc + ZOOM_S)
    ax1.plot(to_utc(gg[zm]), sv[zm], color="tab:red",  lw=1.2, label="SFY")
    ax1.plot(to_utc(gg[zm]), ov[zm], color="tab:blue", lw=1.2, alpha=0.85, label="OLA")
    ax1.set_ylabel("vert accel (m/s²)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax1.grid(True, alpha=0.3); ax1.legend(loc="upper right")
    ax1.set_title(f"Zoom ({ZOOM_S:.0f} s, mid-window) — crest-level alignment check")

    # spectrum: SFY vs OLA band-passed, with OLA raw as reference
    for sig, c, lw_, lab in ((o_raw, "0.75",     1.0, "OLA raw (unfiltered)"),
                             (s_bp,  "tab:red",  1.3, "SFY"),
                             (o_bp,  "tab:blue", 1.3, "OLA band-passed")):
        f, psd = welch(sig, fs=fs, nperseg=int(min(2048, len(sig))))
        ax2.plot(f, psd, color=c, lw=lw_, label=lab)
    ax2.axvline(low_hz_used, color="tab:green", ls="--", lw=1.5,
                label=f"low cutoff = {low_hz_used:.3f} Hz"
                      + (" (adaptive)" if OLA_LOW_HZ == "adaptive" else ""))
    ax2.axvline(high_hz_used, color="tab:olive", ls="--", lw=1.5,
                label=f"high cutoff = {high_hz_used:.2f} Hz"
                      + (" (adaptive)" if OLA_HIGH_HZ == "adaptive" else ""))
    ax2.set_xlim(0, 3.0)
    ax2.set_xlabel("frequency (Hz)")
    ax2.set_ylabel("accel PSD ((m/s²)²/Hz)")
    ax2.grid(True, alpha=0.3); ax2.legend(loc="upper right")
    ax2.set_title("Acceleration spectrum (Welch)")

    # trim angle (OLA AHRS attitude; the SFY npz carries no attitude data).
    # Roll/pitch are Euler angles; tilt = angle(body-z, world-z) =
    # arccos(cos(roll)·cos(pitch)) is the singularity-free total trim.
    t_att = to_utc(ot + shift)
    tilt = np.rad2deg(np.arccos(np.clip(
        np.cos(np.deg2rad(ola_roll)) * np.cos(np.deg2rad(ola_pitch)), -1, 1)))
    ax3.plot(t_att, ola_roll,  color="tab:blue",   lw=0.6, alpha=0.8, label="roll")
    ax3.plot(t_att, ola_pitch, color="tab:orange", lw=0.6, alpha=0.8, label="pitch")
    ax3.plot(t_att, tilt,      color="tab:green",  lw=0.9,
             label="tilt from level (singularity-free)")
    ax3.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax3.set_ylabel("trim angle (deg)")
    ax3.set_xlabel("UTC")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax3.grid(True, alpha=0.3); ax3.legend(loc="upper right")
    ax3.set_title(f"OLA trim angle (AHRS attitude) — roll std={ola_roll.std():.1f}°, "
                  f"pitch std={ola_pitch.std():.1f}°, tilt mean={tilt.mean():.1f}°")

    fig.tight_layout()
    out = HERE / f"accel_sfy_vs_ola_{TP}_{RUN}.png"
    fig.savefig(out, dpi=90)
    print(f"\nSaved {out}")
    plt.show()


if __name__ == "__main__":
    main()
