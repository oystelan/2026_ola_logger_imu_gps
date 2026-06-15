#!/usr/bin/env python3
"""Integrate the saved vertical-acceleration timeseries to velocity and
displacement, for OLA and SFY, and overlay them.

Input: vertical_timeseries_<BOOT>.npz written by compare_ola_vs_sfy.py
(SAVE_TIMESERIES=True). It holds each buoy's gravity-removed vertical accel
(+up, m/s²) on its own time-corrected UTC grid (int64 ns since epoch).

Double-integrating acceleration amplifies any low-frequency residual by 1/ω², so
a raw cumulative integral drifts away. Each integration stage therefore needs a
drift-removal step. Two approaches are compared:

  A_butterworth_hp — high-pass (zero-phase Butterworth) the velocity and the
      displacement. Clean in the wave band, but a Butterworth's impulse response
      RINGS at its cutoff, so any transient/spike in the record turns into a
      large sinusoidal bump at ~HP_HZ that can dominate the result.

  B_savgol_detrend — estimate the slow drift with a long, low-order Savitzky-
      Golay smooth and SUBTRACT it (at the velocity and displacement stages).
      The smooth barely follows a localized transient, so the transient survives
      the subtraction roughly intact — no ringing bump — while the slow drift is
      still removed. SAVGOL_WIN_S sets the window (~1/win is the effective
      corner); a low SAVGOL_ORDER makes it a "hard" smooth.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import butter, filtfilt, detrend, savgol_filter
try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:                       # older scipy
    from scipy.integrate import cumtrapz as _ct
    cumtrapz = lambda y, dx, initial=0: _ct(y, dx=dx, initial=initial)

HERE = Path(__file__).parent

# ----------------------------- user knobs ------------------------------------
NPZ = HERE / "vertical_timeseries_BOOT_000008.npz"
HP_ORDER = 4                      # Butterworth order (zero-phase via filtfilt)
OLA_ACCEL_KEY = "ola_accel_up"    # or "ola_accel_up_bandpassed"

# Savitzky-Golay drift-removal parameters (method B). A long, low-order savgol
# smooth estimates the slow drift; subtracting it high-passes the signal WITHOUT
# the ringing a Butterworth high-pass produces (a Butterworth's impulse response
# rings at its cutoff, so any transient/spike turns into a big sinusoidal bump at
# ~HP_HZ). The savgol smooth of a localized transient barely moves, so the
# transient survives the subtraction roughly intact.
SAVGOL_WIN_S  = 8.0              # smoothing window (s); ~1/win sets the effective corner
SAVGOL_ORDER  = 2                 # polynomial order of the smooth (low = "hard" smoothing)

# Each integration stage can independently DETREND (remove linear trend),
# HIGH-PASS (Butterworth, zero-phase), and/or SAVGOL-DETREND (subtract a savgol
# smooth). Pass the relevant *_hp_hz / *_savgol_s; leave None to skip.
#
#   A_butterworth_hp — detrend the ACCEL, then high-pass velocity (Butterworth).
#                      Clean in the wave band but rings on transients.
#   B_savgol_detrend — detrend the ACCEL, then at the velocity and displacement
#                      stages subtract a savgol smooth (drift estimate) instead
#                      of high-passing. Transient-friendly.
METHODS = {
    "A_butterworth_hp": dict(
        accel_detrend=True,  accel_hp_hz=None,
        vel_detrend=False,   vel_hp_hz=0.05,
        disp_detrend=False,  disp_hp_hz=0.2,
    ),
    "B_savgol_detrend": dict(
        accel_detrend=True,                         # 1) basic accel detrend (as A)
        vel_savgol_s=SAVGOL_WIN_S,                  # 2) subtract savgol drift from velocity
        disp_savgol_s=SAVGOL_WIN_S,                 # 3) subtract savgol drift from displacement
    ),
}


def _clean(x, fs, do_detrend=False, hp_hz=None, savgol_s=None,
           savgol_order=SAVGOL_ORDER, hp_order=HP_ORDER):
    """Clean a signal: optional linear-detrend, then savgol-drift subtraction,
    then zero-phase Butterworth high-pass (any subset, in that order)."""
    if do_detrend:
        x = detrend(x, type="linear")
    if savgol_s is not None:
        win = int(round(savgol_s * fs))
        win += 1 - (win % 2)                       # force odd
        win = min(win, len(x) - (1 - len(x) % 2))  # keep < len, odd
        if win > savgol_order:
            x = x - savgol_filter(x, win, savgol_order)   # subtract the drift estimate
    if hp_hz is not None:
        b, a = butter(hp_order, hp_hz / (fs / 2.0), btype="highpass")
        x = filtfilt(b, a, x)
    return x


def integrate_to_displacement(
    t_ns, accel,
    accel_detrend=True, accel_hp_hz=None, accel_savgol_s=None,
    vel_detrend=False,  vel_hp_hz=None,   vel_savgol_s=None,
    disp_detrend=False, disp_hp_hz=None,  disp_savgol_s=None,
    savgol_order=SAVGOL_ORDER, hp_order=HP_ORDER,
):
    """accel (m/s², +up) on an int64-ns grid -> (tu, accel_raw, accel, vel, disp, fs).

    Resamples to a uniform grid at the median rate (so the zero-phase filters are
    valid), then integrates twice. Each stage (accel, velocity, displacement) is
    independently detrended, savgol-drift-subtracted, and/or high-passed.
    """
    t = (t_ns - t_ns[0]) / 1e9                 # seconds from start
    fs = 1.0 / np.median(np.diff(t))
    tu = np.arange(0.0, t[-1], 1.0 / fs)       # uniform grid
    au_raw = np.interp(tu, t, accel)           # resampled, BEFORE any cleaning
    dt = 1.0 / fs

    au = _clean(au_raw, fs, accel_detrend, accel_hp_hz, accel_savgol_s, savgol_order, hp_order)
    vel = cumtrapz(au, dx=dt, initial=0.0)
    vel = _clean(vel, fs, vel_detrend, vel_hp_hz, vel_savgol_s, savgol_order, hp_order)
    disp = cumtrapz(vel, dx=dt, initial=0.0)
    disp = _clean(disp, fs, disp_detrend, disp_hp_hz, disp_savgol_s, savgol_order, hp_order)
    return tu, au_raw, au, vel, disp, fs


def main():
    d = np.load(NPZ, allow_pickle=True)
    print(f"Loaded {NPZ.name}")
    if "meta" in d:
        print(f"  meta: {d['meta']}")

    buoys = [("OLA", "ola_time_ns", OLA_ACCEL_KEY),
             ("SFY", "sfy_time_ns", "sfy_accel_up")]

    # out[method][buoy] -> dict(utc, accel, vel, disp, fs)
    out = {m: {} for m in METHODS}
    for mname, cfg in METHODS.items():
        print(f"\nMethod {mname}: {cfg}")
        for bname, tkey, akey in buoys:
            t_ns = d[tkey].astype(np.int64)
            accel = d[akey].astype(np.float64)
            tu, au_raw, au, vel, disp, fs = integrate_to_displacement(t_ns, accel, **cfg)
            utc = (t_ns[0] + (tu * 1e9).astype(np.int64)).astype("datetime64[ns]")
            out[mname][bname] = dict(utc=utc, accel_raw=au_raw, accel=au,
                                     vel=vel, disp=disp, fs=fs)
            print(f"  {bname}: vel std={vel.std()*100:5.1f} cm/s  "
                  f"disp std={disp.std()*100:5.1f} cm  "
                  f"(Hs~4*std={4*disp.std()*100:5.1f} cm)")

    # cross-method displacement difference per buoy (how much the choice matters)
    print("\nDisplacement: method A vs B (on the common overlap):")
    ma, mb = list(METHODS.keys())
    for bname, *_ in buoys:
        da, db = out[ma][bname], out[mb][bname]
        n = min(len(da["disp"]), len(db["disp"]))
        rms = np.sqrt(np.mean((da["disp"][:n] - db["disp"][:n]) ** 2)) * 100
        print(f"  {bname}: RMS(A-B) = {rms:.2f} cm  "
              f"(A std={da['disp'].std()*100:.1f}, B std={db['disp'].std()*100:.1f} cm)")

    # ---- plot: 3 rows (accel, velocity, displacement) x 2 cols (OLA, SFY) ----
    fig, ax = plt.subplots(3, 2, figsize=(15, 11), sharex=True)
    mcol = {ma: "tab:green", mb: "tab:orange"}
    for j, (bname, *_ ) in enumerate(buoys):
        # Top row: raw (pre-cleaning) accel once, plus each method's cleaned accel.
        raw = out[ma][bname]
        ax[0, j].plot(raw["utc"], raw["accel_raw"], color="0.6", lw=0.6,
                      label="raw (resampled)")
        for mname in METHODS:
            o = out[mname][bname]
            ax[0, j].plot(o["utc"], o["accel"], mcol[mname], lw=0.7, label=mname)
            ax[1, j].plot(o["utc"], o["vel"] * 100, mcol[mname], lw=0.7, label=mname)
            ax[2, j].plot(o["utc"], o["disp"] * 100, mcol[mname], lw=0.8, label=mname)
        ax[0, j].set_title(f"{bname}")
        ax[2, j].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax[2, j].set_xlabel("UTC")

    # Shared y-limits per ROW so OLA (left) and SFY (right) are on the same
    # scale for easy comparison. accel: scale to the raw signal; velocity /
    # displacement: scale to the first (well-behaved) method — a no-HP pipeline
    # can diverge by 1/w^2 and would otherwise dominate. Methods that exceed the
    # panel are annotated per-buoy as off-scale.
    acc_lim = 1.1 * max(np.max(np.abs(out[ma][bn]["accel_raw"]))
                        for bn, *_ in buoys)
    for j in range(2):
        ax[0, j].set_ylim(-acc_lim, acc_lim)
    for row, key in ((1, "vel"), (2, "disp")):
        lim = 1.3 * max(np.max(np.abs(out[ma][bn][key] * 100)) for bn, *_ in buoys)
        for j, (bname, *_ ) in enumerate(buoys):
            ax[row, j].set_ylim(-lim, lim)
            for k, mname in enumerate(METHODS):
                peak = np.max(np.abs(out[mname][bname][key] * 100))
                if peak > lim:
                    ax[row, j].text(
                        0.02, 0.9 - 0.1 * k,
                        f"{mname} off-scale (±{peak:.0f} cm)",
                        transform=ax[row, j].transAxes, fontsize=7, color=mcol[mname])
    ax[0, 0].set_ylabel("accel +up (m/s²)")
    ax[1, 0].set_ylabel("velocity (cm/s)")
    ax[2, 0].set_ylabel("displacement (cm)")
    for a_ in ax.ravel():
        a_.grid(True, alpha=0.3); a_.legend(loc="upper right", fontsize=8)
    fig.suptitle("Integration drift-removal comparison: "
                 "A (Butterworth high-pass) vs B (savgol drift subtraction)")
    fig.tight_layout()
    out_png = HERE / "integrated_displacement.png"
    fig.savefig(out_png, dpi=90)
    print(f"\nSaved {out_png}")

    # ---- second figure: columns = method (A | B), each panel overlays OLA+SFY ----
    # This view answers "do the two buoys agree?" within each pipeline.
    fig2, ax2 = plt.subplots(3, 2, figsize=(15, 11), sharex=True)
    bcol = {"OLA": "tab:blue", "SFY": "tab:red"}
    methods = list(METHODS)
    rows = [("accel", "accel +up (m/s²)", 1.0),
            ("vel",   "velocity (cm/s)",  100.0),
            ("disp",  "displacement (cm)", 100.0)]
    for jc, mname in enumerate(methods):
        for ri, (key, _ylab, scale) in enumerate(rows):
            for bname, *_ in buoys:
                o = out[mname][bname]
                ax2[ri, jc].plot(o["utc"], o[key] * scale, bcol[bname],
                                 lw=0.7 if ri < 2 else 0.8, label=bname)
            ax2[ri, jc].grid(True, alpha=0.3)
            ax2[ri, jc].legend(loc="upper right", fontsize=8)
        ax2[0, jc].set_title(mname)
        ax2[2, jc].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax2[2, jc].set_xlabel("UTC")
    # Shared y-limits per ROW across the two method columns (scaled to the
    # well-behaved method A so a diverging pipeline doesn't dominate).
    for ri, (key, ylab, scale) in enumerate(rows):
        ax2[ri, 0].set_ylabel(ylab)
        if key == "accel":
            lim = 1.1 * max(np.max(np.abs(out[ma][bn]["accel_raw"])) for bn, *_ in buoys)
        else:
            lim = 1.3 * max(np.max(np.abs(out[ma][bn][key] * scale)) for bn, *_ in buoys)
        for jc, mname in enumerate(methods):
            ax2[ri, jc].set_ylim(-lim, lim)
            for k, (bname, *_ ) in enumerate(buoys):
                peak = np.max(np.abs(out[mname][bname][key] * scale))
                if peak > lim:
                    ax2[ri, jc].text(
                        0.02, 0.9 - 0.1 * k,
                        f"{bname} off-scale (±{peak:.0f})",
                        transform=ax2[ri, jc].transAxes, fontsize=7, color=bcol[bname])
    fig2.suptitle("OLA vs SFY per pipeline:  A (Butterworth high-pass)  |  B (savgol detrend)")
    fig2.tight_layout()
    out_png2 = HERE / "integrated_displacement_by_method.png"
    fig2.savefig(out_png2, dpi=90)
    print(f"Saved {out_png2}")

    plt.show()


if __name__ == "__main__":
    main()
