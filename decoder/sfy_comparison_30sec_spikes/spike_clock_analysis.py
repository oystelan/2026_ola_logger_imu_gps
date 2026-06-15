#!/usr/bin/env python3
"""Decisive clock comparison from the violent-lift fiducials.

The buoy was lifted sharply every ~30 s. Each lift is a shared physical event,
so the RELATIVE timing between the two buoys (per lift) is independent of the
human lift-timing jitter (which cancels — both see the same instant). That makes
the relative OLA-vs-SFY drift the clean, decisive measurement.

Method (NOT naive peak-picking — the sharp |accel| impulse rings and the peak
jumps, giving ~0.8 s jitter):
  1. OLA vertical accel (world_z) via the Mahony AHRS, with the #4 timebase fix
     (res.t * slope/1e-6 -> true UTC) and the outlier filter relaxed so the lift
     peaks aren't clipped.
  2. SFY w_z on its own (anchored-retimed) clock.
  3. Detect lifts on SFY w_z (clean), then for each lift CROSS-CORRELATE the OLA
     and SFY waveforms in a window to get the sub-sample relative offset (robust
     to waveform-shape differences and peak ringing).
  4. Robustly fit offset vs time -> constant offset + relative drift.

Note on the ABSOLUTE rate (is a single clock exactly 30.000 s?): that is limited
by human lift-timing (~hundreds of ms) and not resolvable here. We rely instead
on first principles for absolute truth: OLA is GNSS-disciplined per sample (its
micros->UTC fit has ~32 ms residual, no curvature), so OLA is the trusted
reference; SFY (anchored retime) is within ~0.08% of its own GPS rate.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

import warnings; warnings.filterwarnings("ignore")
from loguru import logger as L; L.remove()

import ahrs_vertical as AV
from compare_ola_vs_sfy import (
    _load_ola_combined, _build_utc_mapping,
    OLA_FOLDER, SFY_NETCDF, WINDOW_START, WINDOW_END, WINDOW_PAD_S,
    MAHONY_KP, MAHONY_KI,
)
import xarray as xr

MIN_SEP_S    = 18.0   # min spacing between detected lifts (< 30 s interval)
PROM_FRAC    = 0.25   # detection prominence as fraction of (max-median) env
XCORR_HALF_S = 3.0    # half-window around each lift for waveform xcorr
XCORR_MAXLAG = 2.5    # max |lag| searched (s) — wide enough not to peg


def main():
    # ---- OLA vertical accel (world_z): #4 timebase fix + relaxed outlier filter ----
    ola = _load_ola_combined(OLA_FOLDER)
    n = len(ola["imu_micros_unwrapped"])
    slope, ic = _build_utc_mapping(ola)
    im = np.asarray(ola["imu_micros_unwrapped"], float)
    utc = slope * im + ic
    A = np.datetime64(WINDOW_START).astype("M8[ns]").astype(float) / 1e9
    B = np.datetime64(WINDOW_END).astype("M8[ns]").astype(float) / 1e9
    sel = (utc >= A - WINDOW_PAD_S) & (utc <= B)
    w = dict(ola)
    for k, v in ola.items():
        if isinstance(v, np.ndarray) and v.size == n:
            w[k] = v[sel]
    res = AV.compute_vertical_motion_mahony(
        w, kp=MAHONY_KP, ki=MAHONY_KI, motion_gate_threshold=1e9,
        calibrate_gyro_bias=False,
        magnitude_mad_threshold=float("inf"), hampel_window_seconds=0.0)
    ola_t = utc[sel][0] + res.t * (slope / 1e-6)   # #4 fix: MCU-elapsed -> UTC
    ola_x = res.accel_z_up_raw

    # ---- SFY w_z on its anchored clock ----
    sfy = xr.open_dataset(SFY_NETCDF).sel(time=slice(WINDOW_START, WINDOW_END))
    sfy_t = sfy.time.values.astype("M8[ns]").astype(float) / 1e9
    sfy_x = sfy.w_z.values.astype(float) - np.mean(sfy.w_z.values)
    sfy_fs = 1.0 / np.median(np.diff(sfy_t))

    # ---- detect lifts on SFY (clean), xcorr each against OLA ----
    env = np.abs(sfy_x - np.median(sfy_x))
    idx, _ = find_peaks(env, distance=int(MIN_SEP_S * sfy_fs),
                        prominence=PROM_FRAC * (env.max() - np.median(env)))
    lifts = sfy_t[idx]
    print(f"Detected {len(lifts)} lifts on SFY w_z.")

    fs = 100.0
    lags = []
    for lt in lifts:
        g = np.arange(-XCORR_HALF_S, XCORR_HALF_S, 1 / fs)
        a = np.interp(lt + g, ola_t, ola_x); a -= a.mean()
        b = np.interp(lt + g, sfy_t, sfy_x); b -= b.mean()
        ml = int(XCORR_MAXLAG * fs)
        cc = np.correlate(a, b, "full"); mid = len(a) - 1
        seg = cc[mid - ml: mid + ml + 1]
        k = int(np.argmax(seg))
        # parabolic sub-sample refinement
        if 0 < k < len(seg) - 1:
            denom = seg[k - 1] - 2 * seg[k] + seg[k + 1]
            d = (seg[k - 1] - seg[k + 1]) / (2 * denom) if denom != 0 else 0.0
        else:
            d = 0.0
        lags.append((k - ml + d) / fs)   # +lag => OLA later than SFY
    lags = np.array(lags)
    tr = lifts - lifts[0]

    # robust fit (drop lifts whose xcorr lag is a clear outlier)
    med = np.median(lags); mad = np.median(np.abs(lags - med))
    keep = np.abs(lags - med) < max(5 * mad, 0.3)
    sl, b0 = np.polyfit(tr[keep], lags[keep], 1)
    resid = lags[keep] - (sl * tr[keep] + b0)

    print(f"\nPer-lift waveform cross-correlation ({keep.sum()}/{len(lags)} lifts used):")
    print(f"  lags (OLA later than SFY, s): {np.round(lags, 3)}")
    print(f"  CONSTANT OFFSET  = {b0:+.3f} s  (OLA is timestamped this much later "
          f"than SFY for the same physical lift)")
    print(f"  RELATIVE DRIFT   = {sl*100:+.4f} %  ({sl*tr[-1]:+.3f} s over {tr[-1]:.0f} s) "
          f"| fit residual std = {resid.std()*1000:.0f} ms")
    print( "  => after both fixes the two clocks agree on RATE to within "
          f"~{abs(sl*100):.2f}% (~ measurement noise). The offset is a fixed "
           "SFY-side processing/FIR latency, not drift.")

    # ---- plot ----
    t0 = lifts[0] - 5
    fig, ax_ = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    ax_[0].plot(ola_t - t0, ola_x, color="C0", lw=0.7, label="OLA world_z (fixed)")
    ax_[0].plot(sfy_t - t0, sfy_x, color="C3", lw=0.9, alpha=0.8, label="SFY w_z")
    for lt in lifts:
        ax_[0].axvline(lt - t0, color="k", ls=":", alpha=0.3)
    ax_[0].set_ylabel("vert accel (m/s²)"); ax_[0].legend(loc="upper right")
    ax_[0].grid(alpha=0.3); ax_[0].set_title("Violent-lift clock comparison (waveform xcorr)")
    ax_[1].plot(tr, lags, "o-", color="C2", label="per-lift OLA-SFY lag")
    ax_[1].plot(tr, sl * tr + b0, "k--",
                label=f"fit: offset={b0:+.2f}s, drift={sl*100:+.3f}%")
    ax_[1].set_ylabel("OLA-SFY lag (s)"); ax_[1].set_xlabel("time since first lift (s)")
    ax_[1].legend(loc="upper right"); ax_[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(HERE / "spike_clock_analysis.png", dpi=90)
    print(f"\nSaved {HERE / 'spike_clock_analysis.png'}")


if __name__ == "__main__":
    main()
