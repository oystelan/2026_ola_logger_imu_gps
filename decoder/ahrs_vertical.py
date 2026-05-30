"""AHRS-based vertical-position estimator.

Designed for indoor / basin / GPS-denied recordings where only the *oscillatory*
component of vertical motion matters. Pipeline:

1. Madgwick AHRS (accel + gyro, optional mag) → body→world quaternion
2. Rotate measured body-frame acceleration into world frame
3. Subtract gravity → world-frame linear acceleration
4. Take the z component (with sign flipped so +z is up)
5. Resample onto a uniform time grid
6. Frequency-domain double integration with a band-pass filter:
   multiply spectrum by -1/ω², zero out bins outside [low, high] Hz
   → inverse FFT → vertical displacement

The band-pass step is what makes this drift-free. DC and sub-low-Hz content
are explicitly zeroed in the spectrum, so the integrator can never accumulate
slow biases into runaway position. This is the standard wave-buoy technique
(see Rabault et al. 2022; sfy-processing/signal.py).

Tradeoff: you lose the absolute / mean vertical position. Output is
displacement *around zero*, valid only in the chosen frequency band.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger
from scipy.signal import butter, savgol_filter, sosfiltfilt
from scipy.signal.windows import tukey

from decoder import detect_outliers_stdcheck
from scipy.spatial.transform import Rotation


G_STD = 9.80665  # m/s², standard gravity


# ------------------------------ Madgwick AHRS ------------------------------


@dataclass
class MadgwickState:
    """Internal Madgwick state — quaternion q_body→world."""

    # (w, x, y, z) form. World frame is NED-like: z points DOWN, so at rest the
    # accelerometer reads (0, 0, -g) in body frame when body z = world z (level).
    q: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))

    def as_scipy(self) -> Rotation:
        """Return scipy Rotation. Note scipy uses (x, y, z, w) order."""
        w, x, y, z = self.q
        return Rotation.from_quat([x, y, z, w])


def madgwick_step_imu(
    state: MadgwickState,
    accel_body: np.ndarray,
    gyro_body: np.ndarray,
    dt: float,
    beta: float = 0.1,
    motion_gate_threshold: float = 1.5,
) -> bool:
    """Single Madgwick step using accel + gyro (no magnetometer).

    Updates `state` in place. World frame: z-down (NED-like). Gravity
    reference is (0, 0, 1) — the unit vector accelerometer at rest is
    expected to OPPOSE (sensor reads case-pushing-up, hence -g in body z).

    `beta` controls how much the accelerometer correction pulls on the
    pure-gyro propagation. Larger β → more accel weight (faster bias
    correction, more wobble from accel noise). 0.05–0.1 is typical for
    consumer MEMS. 0.033 (Madgwick's original) for slow / clean data.

    `motion_gate_threshold` (m/s²): if |accel| differs from the standard
    gravity magnitude (9.81 m/s²) by more than this, the accel-based
    attitude correction is SKIPPED for this step and only the gyro is
    used to propagate. This is the standard fix for Madgwick's gravity-
    vs-linear-acceleration confusion: when the device is being moved (so
    accel = gravity + linear), trusting accel = gravity puts a fake
    correction into the attitude. Default 1.5 m/s² (≈ 0.15 g) is a
    moderate gate. Lower = trust gyro more (slower bias correction);
    higher = trust accel more (more attitude noise during motion).

    Returns True if the accel correction was applied this step, False if
    it was gated (so the caller can count gated samples).
    """
    q0, q1, q2, q3 = state.q

    # Normalise accel — direction only is meaningful for gravity reference
    a_norm = np.linalg.norm(accel_body)
    # Two-condition gate:
    #   1. Near-zero accel (free-fall) — direction undefined
    #   2. |accel| far from 1g — device under non-gravity acceleration,
    #      accel is not pointing at "gravity", trusting it would create a
    #      false attitude correction. This is the *motion gate*.
    if a_norm < 1e-6:
        ax = ay = az = 0.0
        use_accel = False
    elif abs(a_norm - 9.80665) > motion_gate_threshold:
        ax = ay = az = 0.0
        use_accel = False
    else:
        ax, ay, az = accel_body / a_norm
        use_accel = True

    gx, gy, gz = gyro_body  # rad/s

    # Gyro-driven quaternion derivative (q̇_gyro = 0.5 q ⊗ ω)
    qDot = 0.5 * np.array([
        -q1 * gx - q2 * gy - q3 * gz,
        +q0 * gx + q2 * gz - q3 * gy,
        +q0 * gy - q1 * gz + q3 * gx,
        +q0 * gz + q1 * gy - q2 * gx,
    ])

    if use_accel:
        # Objective function f: misalignment of body-frame "down" (R^T·[0,0,1])
        # with the negated accel direction. Gradient descent step (Madgwick 2010).
        # World gravity direction expressed in body frame: R(q)^T · [0,0,1] =
        #   [2(q1*q3 - q0*q2), 2(q0*q1 + q2*q3), 1 - 2(q1² + q2²)]
        # We compare this to (-accel_normalised) since accel opposes gravity at rest.
        f1 = 2 * (q1 * q3 - q0 * q2) - (-ax)
        f2 = 2 * (q0 * q1 + q2 * q3) - (-ay)
        f3 = 1 - 2 * (q1 ** 2 + q2 ** 2) - (-az)

        # Jacobian
        s0 = -2 * q2 * f1 + 2 * q1 * f2
        s1 = 2 * q3 * f1 + 2 * q0 * f2 - 4 * q1 * f3
        s2 = -2 * q0 * f1 + 2 * q3 * f2 - 4 * q2 * f3
        s3 = 2 * q1 * f1 + 2 * q2 * f2

        s = np.array([s0, s1, s2, s3])
        s_norm = np.linalg.norm(s)
        if s_norm > 1e-12:
            s /= s_norm
            qDot -= beta * s

    # Integrate and renormalise
    q_new = state.q + qDot * dt
    q_new /= np.linalg.norm(q_new)
    state.q = q_new
    return use_accel


def initial_attitude_from_accel(accel_mean_body: np.ndarray) -> np.ndarray:
    """Roll/pitch from gravity direction; yaw arbitrary (set to 0).

    Returns quaternion (w, x, y, z), world frame z-down. Yaw is unobservable
    without a magnetometer, but irrelevant for vertical motion.
    """
    g_body = -accel_mean_body / np.linalg.norm(accel_mean_body)  # gravity dir in body
    # Tilt: rotate body-frame +z to align with g_body
    # roll about x, pitch about y (Z-Y-X intrinsic Euler with yaw=0)
    roll = np.arctan2(g_body[1], g_body[2])
    pitch = np.arctan2(-g_body[0], np.sqrt(g_body[1] ** 2 + g_body[2] ** 2))
    rot = Rotation.from_euler("ZYX", [0.0, pitch, roll])  # yaw=0
    x, y, z, w = rot.as_quat()
    return np.array([w, x, y, z])


# ------------------------ frequency-domain integration ----------------------


def _build_soft_bandpass_mask(
    freqs: np.ndarray, low_hz: float, high_hz: float, softness: float,
) -> np.ndarray:
    """Frequency-domain gain in [0, 1] with cosine-tapered cutoffs.

    `softness` is the FRACTIONAL half-width of each transition band:
      - low edge: ramps 0 → 1 over [low_hz*(1-s), low_hz*(1+s)]
      - high edge: ramps 1 → 0 over [high_hz*(1-s), high_hz*(1+s)]
    softness=0 reduces to the hard binary cutoff; softness=0.5 gives a
    half-octave transition on each side.
    """
    if softness <= 0:
        return ((freqs >= low_hz) & (freqs <= high_hz)).astype(float)

    low_a = low_hz * (1.0 - softness)
    low_b = low_hz * (1.0 + softness)
    high_a = high_hz * (1.0 - softness)
    high_b = high_hz * (1.0 + softness)
    # Guard against accidental overlap if softness is set absurdly wide
    if low_b > high_a:
        low_b = high_a = 0.5 * (low_b + high_a)

    gain = np.zeros_like(freqs, dtype=float)
    # Low-edge ramp: 0 → 1 via raised cosine
    in_low_ramp = (freqs >= low_a) & (freqs < low_b)
    if low_b > low_a:
        gain[in_low_ramp] = 0.5 * (
            1.0 - np.cos(np.pi * (freqs[in_low_ramp] - low_a) / (low_b - low_a))
        )
    # Flat passband
    gain[(freqs >= low_b) & (freqs < high_a)] = 1.0
    # High-edge ramp: 1 → 0
    in_high_ramp = (freqs >= high_a) & (freqs < high_b)
    if high_b > high_a:
        gain[in_high_ramp] = 0.5 * (
            1.0 + np.cos(np.pi * (freqs[in_high_ramp] - high_a) / (high_b - high_a))
        )
    return gain


def fft_double_integrate_bandpass(
    accel: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    taper_fraction: float = 0.1,
    cutoff_softness: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Double-integrate `accel` (m/s²) to displacement (m), band-pass-only.

    Implementation: take FFT, multiply by -1/ω² for double integration,
    apply a soft (cosine-tapered) band-pass mask in the frequency domain,
    inverse FFT. The mask kills DC and low-frequency drift — without it
    the -1/ω² operator blows up at ω→0.

    `taper_fraction` (default 0.1 = 10% of length tapered each side via a
    Tukey window) suppresses time-domain edge discontinuity leakage. Set
    to 0.0 to disable.

    `cutoff_softness` (default 0.3) controls how sharp the spectral cutoffs
    are. A HARD cutoff (softness=0) corresponds to a binary mask, which is
    a frequency-domain step — and by duality that's a long-duration sinc
    ring in time, manifesting as a spurious oscillation at the cutoff
    frequency. A SOFT cutoff replaces the step with a raised-cosine ramp,
    dramatically reducing the time-domain ring. softness=0.3 means each
    cutoff ramps over ±30% of its centre frequency (so low_hz=0.05 ramps
    from 0.035 to 0.065 Hz). The price is a slightly attenuated response
    near the cutoff edges — content right at low_hz is at half gain.

    Returns (velocity, displacement) — single and double integrated.
    """
    n = len(accel)
    if taper_fraction > 0:
        win = tukey(n, alpha=taper_fraction * 2)
        signal = accel * win
    else:
        signal = accel

    Y = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    omega = 2 * np.pi * freqs

    soft_mask = _build_soft_bandpass_mask(freqs, low_hz, high_hz, cutoff_softness)
    int1 = np.zeros_like(Y, dtype=complex)
    int2 = np.zeros_like(Y, dtype=complex)
    safe = omega > 0
    int1[safe] = soft_mask[safe] * (1.0 / (1j * omega[safe]))
    int2[safe] = soft_mask[safe] * (-1.0 / (omega[safe] ** 2))

    V = Y * int1
    D = Y * int2
    velocity = np.fft.irfft(V, n=n)
    displacement = np.fft.irfft(D, n=n)
    return velocity, displacement


def fft_double_integrate(
    accel: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Double-integrate `accel` (m/s²) to displacement (m) via FFT — raw.

    No band-pass mask, no Tukey taper. Just multiply the spectrum by 1/(jω) for
    velocity and -1/ω² for displacement; the DC bin is zeroed (the only way to
    keep 1/ω² well-defined at ω=0). Every other frequency passes through unchanged.

    Use this when upstream processing has already removed the slow content (e.g.
    polynomial detrend, savgol attitude detrend) and you don't want the spectral
    mask to chop more out. For raw / un-detrended signals on short recordings,
    `fft_double_integrate_bandpass` is usually safer — pure 1/ω² hugely amplifies
    any residual low-frequency content.

    Returns (velocity, displacement) — single and double integrated.
    """
    n = len(accel)
    Y = np.fft.rfft(accel)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    omega = 2 * np.pi * freqs

    int1 = np.zeros_like(Y, dtype=complex)
    int2 = np.zeros_like(Y, dtype=complex)
    safe = omega > 0
    int1[safe] = 1.0 / (1j * omega[safe])
    int2[safe] = -1.0 / (omega[safe] ** 2)

    velocity = np.fft.irfft(Y * int1, n=n)
    displacement = np.fft.irfft(Y * int2, n=n)
    return velocity, displacement


def bandpass_zero_phase(
    x: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 4,
) -> np.ndarray:
    """Forward-backward Butterworth band-pass (no phase shift)."""
    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def lowpass_zero_phase(
    x: np.ndarray, fs: float, high_hz: float, order: int = 4,
) -> np.ndarray:
    """Forward-backward Butterworth low-pass (no phase shift)."""
    sos = butter(order, high_hz, btype="lowpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def estimate_time_varying_gyro_bias(
    t: np.ndarray,
    accel: np.ndarray,
    gyro: np.ndarray,
    gravity_tolerance_mps2: float = 0.5,
    min_stationary_seconds: float = 0.5,
    rolling_window_seconds: float = 0.5,
    accel_mag_std_max: float = 0.4,
    gyro_std_max_dps: float = 1.5,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Estimate time-varying gyro bias using accel-and-gyro trust points.

    Algorithm:
      1. For each sample, look at a centred window of length
         `rolling_window_seconds`. The sample is a *trust point* when, over
         that window, ALL of the following hold:
           (a) mean |accel| is within `gravity_tolerance_mps2` of 9.81 m/s²
               (the device is NOT undergoing sustained linear acceleration)
           (b) std of |accel| is below `accel_mag_std_max` (no jerks/jolts)
           (c) max std of any gyro component is below `gyro_std_max_dps`
               (the device is NOT rotating)
         Crucially, (c) is what distinguishes "true stationarity" from
         "instant where |accel| ≈ g during a rotation" — without it, a
         sample mid-rotation where the centrifugal/translational accels
         momentarily cancel out gets falsely flagged as stationary.
      2. Group consecutive trust points into "stationary runs". Runs shorter
         than `min_stationary_seconds` are discarded (too noisy to average).
      3. For each surviving run, compute mean gyro = bias estimate, anchored
         at the midpoint time of the run.
      4. Returns (anchor_times, anchor_biases) — these can be linearly
         interpolated by `apply_time_varying_gyro_bias()` to give a smooth
         time-varying bias estimate across the whole recording.

    This is the offline *smoother* complement of Madgwick's online filter:
    instead of correcting attitude continuously with a possibly-unreliable
    accel signal, it explicitly solves for the gyro bias using only samples
    where BOTH sensors are known to be reliable (truly stationary).

    Returns (None, None) if no truly-stationary runs of sufficient length
    exist.
    """
    n = len(t)
    if n < 3:
        return None, None
    fs = 1.0 / max(np.median(np.diff(t)), 1e-6)
    win_n = max(3, int(round(rolling_window_seconds * fs)))
    if win_n >= n:
        return None, None

    accel_mag = np.linalg.norm(accel, axis=1)
    # Rolling mean & std via cumulative-sum convolution. mode='same' so the
    # result aligns sample-to-sample; edges use partial windows which is
    # fine for our purposes (they get reflected slightly).
    kernel = np.ones(win_n) / win_n
    am_mean = np.convolve(accel_mag, kernel, mode="same")
    am_sq_mean = np.convolve(accel_mag ** 2, kernel, mode="same")
    am_std = np.sqrt(np.maximum(am_sq_mean - am_mean ** 2, 0))

    gyr_dps = np.rad2deg(gyro)
    g_max_std = np.zeros(n)
    for axis in range(3):
        g_axis = gyr_dps[:, axis]
        g_mean = np.convolve(g_axis, kernel, mode="same")
        g_sq_mean = np.convolve(g_axis ** 2, kernel, mode="same")
        g_std_axis = np.sqrt(np.maximum(g_sq_mean - g_mean ** 2, 0))
        g_max_std = np.maximum(g_max_std, g_std_axis)

    trusted = (
        (np.abs(am_mean - 9.80665) < gravity_tolerance_mps2)
        & (am_std < accel_mag_std_max)
        & (g_max_std < gyro_std_max_dps)
    )

    anchors = []  # list of (mid_time, mean_gyro_for_run)
    in_run = False
    run_start = 0
    for i in range(n):
        if trusted[i] and not in_run:
            in_run = True
            run_start = i
        elif (not trusted[i]) and in_run:
            in_run = False
            run_end = i
            if t[run_end - 1] - t[run_start] >= min_stationary_seconds:
                mid_t = 0.5 * (t[run_start] + t[run_end - 1])
                mean_g = gyro[run_start:run_end].mean(axis=0)
                anchors.append((mid_t, mean_g))
    if in_run:
        run_end = n
        if t[run_end - 1] - t[run_start] >= min_stationary_seconds:
            mid_t = 0.5 * (t[run_start] + t[run_end - 1])
            mean_g = gyro[run_start:run_end].mean(axis=0)
            anchors.append((mid_t, mean_g))

    if not anchors:
        return None, None
    times = np.array([a[0] for a in anchors])
    biases = np.array([a[1] for a in anchors])
    return times, biases


def apply_time_varying_gyro_bias(
    t: np.ndarray,
    gyro: np.ndarray,
    anchor_times: np.ndarray,
    anchor_biases: np.ndarray,
) -> np.ndarray:
    """Subtract a linearly-interpolated gyro bias from the signal.

    Returns a new array; does not modify input. For samples outside the
    anchor span (start before first anchor or end after last anchor),
    np.interp clamps to the endpoint values (constant extrapolation), which
    is the safest behaviour: we trust the nearest anchor as the best bias
    estimate available.
    """
    if anchor_times is None or len(anchor_times) == 0:
        return gyro
    corrected = gyro.copy()
    for axis in range(3):
        b_axis = np.interp(t, anchor_times, anchor_biases[:, axis])
        corrected[:, axis] = gyro[:, axis] - b_axis
    return corrected


def polynomial_detrend(x: np.ndarray, order: int) -> np.ndarray:
    """Subtract a least-squares polynomial fit of given order.

    Robust alternative to a low-frequency high-pass filter for short
    recordings: a Butterworth high-pass at e.g. 0.05 Hz has a settling
    transient of ~3/f_c ≈ 60 s, which means recordings shorter than that
    are dominated by edge-ringing. A polynomial subtraction has no
    transient — it just removes whatever smooth trend best fits the data.
    Order 1 = remove DC + linear drift; order 2-3 = also remove slow
    parabolic / cubic curvature (e.g. from gyro bias warming up).
    """
    n = len(x)
    if order < 0 or n <= order + 1:
        return x - x.mean()
    t = np.linspace(-1.0, 1.0, n)  # normalised x-axis for conditioning
    coeffs = np.polyfit(t, x, order)
    trend = np.polyval(coeffs, t)
    return x - trend


# ------------------------- IMU outlier hold helper --------------------------


def _filter_imu_outliers(
    t_raw: np.ndarray,
    acc: np.ndarray,
    gyr: np.ndarray,
    data: dict,
    use_outlier_flags: bool,
    magnitude_mad_threshold: float,
    hampel_window_seconds: float,
    hampel_k_mad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Three-stage hold-last-good outlier filter for IMU samples.

    Shared by `compute_vertical_motion` (Madgwick AHRS path) and
    `compute_vertical_motion_lowpass_gravity` (no-AHRS wave-buoy path).
    Returns cleaned (acc, gyr); inputs are not mutated.

    Stage 1 — per-axis decoder outlier flags (if `use_outlier_flags`).
    Stage 2 — global MAD on |accel| catches end-of-stream / impact spikes.
    Stage 3 — local Hampel on |accel| catches single-sample anomalies inside
              bursts of real motion where the global MAD is too wide.
    """
    n_raw = len(t_raw)

    if use_outlier_flags:
        outlier_mask = np.zeros(n_raw, dtype=bool)
        for k in (
            "imu_acc_x_outlier", "imu_acc_y_outlier", "imu_acc_z_outlier",
            "imu_gyr_x_outlier", "imu_gyr_y_outlier", "imu_gyr_z_outlier",
        ):
            if k in data and len(data[k]) == n_raw:
                outlier_mask |= np.asarray(data[k], dtype=bool)
        acc_c = acc.copy()
        gyr_c = gyr.copy()
        last_a, last_g = acc[0].copy(), gyr[0].copy()
        n_held = 0
        for i in range(n_raw):
            if outlier_mask[i]:
                acc_c[i] = last_a
                gyr_c[i] = last_g
                n_held += 1
            else:
                last_a, last_g = acc[i], gyr[i]
        acc, gyr = acc_c, gyr_c
        if n_held:
            logger.info(f"Outlier-flag hold: replaced {n_held} samples")

    if np.isfinite(magnitude_mad_threshold):
        mag_acc = np.linalg.norm(acc, axis=1)
        median_mag = float(np.median(mag_acc))
        mad = float(np.median(np.abs(mag_acc - median_mag)))
        if mad > 0:
            limit = magnitude_mad_threshold * mad
            mask = np.abs(mag_acc - median_mag) > limit
            if mask.any():
                acc_c = acc.copy()
                gyr_c = gyr.copy()
                last_a, last_g = acc[0].copy(), gyr[0].copy()
                n_mag_held = 0
                for i in range(n_raw):
                    if mask[i]:
                        acc_c[i] = last_a
                        gyr_c[i] = last_g
                        n_mag_held += 1
                    else:
                        last_a, last_g = acc[i], gyr[i]
                acc, gyr = acc_c, gyr_c
                worst_idx = int(np.argmax(np.abs(mag_acc - median_mag)))
                logger.info(
                    f"Magnitude outlier filter: held {n_mag_held} samples "
                    f"(threshold {magnitude_mad_threshold}×MAD = "
                    f"{limit*1000:.0f} mm/s²; median |a| = "
                    f"{median_mag:.2f} m/s², worst sample {worst_idx} "
                    f"@ t={t_raw[worst_idx]:.1f}s with |a|={mag_acc[worst_idx]:.2f} m/s²)"
                )

    if hampel_window_seconds > 0 and np.isfinite(hampel_k_mad):
        dt_med_local = float(np.median(np.diff(t_raw)))
        n_neighbors = int(round(hampel_window_seconds / max(dt_med_local, 1e-6)))
        if n_neighbors >= 4 and n_neighbors < n_raw // 2:
            mag_acc = np.linalg.norm(acc, axis=1)
            outlier_indices = detect_outliers_stdcheck(
                mag_acc, n_neighbors=n_neighbors, n_sigma=hampel_k_mad
            )
            if len(outlier_indices) > 0:
                spike_set = set(int(i) for i in outlier_indices)
                acc_c = acc.copy()
                gyr_c = gyr.copy()
                last_a, last_g = acc[0].copy(), gyr[0].copy()
                n_local_held = 0
                for i in range(n_raw):
                    if i in spike_set:
                        acc_c[i] = last_a
                        gyr_c[i] = last_g
                        n_local_held += 1
                    else:
                        last_a, last_g = acc[i], gyr[i]
                acc, gyr = acc_c, gyr_c
                worst_i = int(np.argmax(np.abs(mag_acc - np.median(mag_acc))))
                logger.info(
                    f"Local spike filter (detect_outliers_stdcheck): held "
                    f"{n_local_held} samples (window={hampel_window_seconds:.2f}s"
                    f"={n_neighbors} neighbours, threshold={hampel_k_mad}σ); "
                    f"worst |a|={mag_acc[worst_i]:.2f} m/s² @ t={t_raw[worst_i]:.1f}s"
                )

    return acc, gyr


# ---------------------------- high-level driver -----------------------------


@dataclass
class VerticalAHRSResult:
    """Output of compute_vertical_motion."""

    t: np.ndarray              # (N,) uniform-grid time, s, starting at 0
    accel_z_up: np.ndarray     # (N,) world-frame vertical accel, +up, m/s² (band-passed)
    accel_z_up_raw: np.ndarray # (N,) same but BEFORE band-pass (gravity removed, unfiltered)
    velocity_z_up: np.ndarray  # (N,) world-frame vertical velocity, +up, m/s
    displacement_z_up: np.ndarray  # (N,) vertical displacement, +up, m
    roll_deg: np.ndarray       # (N,) body roll, deg
    pitch_deg: np.ndarray      # (N,) body pitch, deg
    fs_hz: float               # resampling frequency used
    low_hz: float              # band-pass low edge
    high_hz: float             # band-pass high edge

    @property
    def hs_band(self) -> float:
        """Crude significant wave-height-like statistic: 4× std of displacement.

        Useful for at-a-glance scale check; full spectral Hm0 needs a Welch
        estimate (out of scope here).
        """
        return 4.0 * float(np.std(self.displacement_z_up))


def compute_vertical_motion(
    data: dict,
    low_hz: float = 0.05,
    high_hz: float = 2.5,
    beta: float = 0.3,
    stationary_init_seconds: float = 2.0,
    resample_hz: float | None = None,
    use_outlier_flags: bool = True,
    calibrate_gyro_bias: bool = True,
    magnitude_mad_threshold: float = 8.0,
    hampel_window_seconds: float = 0.25,
    hampel_k_mad: float = 5.0,
    motion_gate_threshold: float = 0.5,
    use_trust_points_bias: bool = False,
    trust_points_gravity_tolerance: float = 0.5,
    trust_points_min_seconds: float = 0.5,
    detrend_polynomial_order: int = 3,
    fft_taper_fraction: float = 0.1,
    cutoff_softness: float = 0.3,
) -> VerticalAHRSResult:
    """Estimate band-pass vertical motion from a `load_data_as_arrays` dict.

    Required arrays: imu_micros_unwrapped, imu_acc_x/y/z (mg), imu_gyr_x/y/z (mdps).
    Magnetometer is optional and is NOT used — yaw is unobservable here and
    irrelevant for vertical position.

    Args:
        data: dict from `load_data_as_arrays`.
        low_hz, high_hz: band-pass edges. The output preserves only motion
            in this band — slower oscillations and DC are zeroed. NOTE:
            FFT double integration multiplies by 1/ω², so any residual
            accel near `low_hz` is *amplified*. Setting `low_hz` too low
            (e.g. 0.01 Hz) makes the result very sensitive to AHRS attitude
            drift. 0.05–0.1 Hz is a safer default for short recordings.
        beta: Madgwick gain. Larger β = faster accel-driven attitude
            correction = less low-frequency leakage from gyro bias; but
            also more high-frequency wobble in attitude. 0.3 is a good
            compromise for MEMS IMUs with `calibrate_gyro_bias=True`;
            0.1 is more conservative if you also want to filter accel noise.
        stationary_init_seconds: window at the start used (a) to seed
            initial attitude from gravity, and (b) when `calibrate_gyro_bias
            =True`, to measure and remove the gyro bias. **Keep the device
            still during this window** for best results.
        resample_hz: target uniform sampling rate for FFT. Default = the
            median IMU rate inferred from the data. Set lower (e.g. 50 Hz)
            for very long recordings if memory is an issue.
        use_outlier_flags: when True, IMU samples flagged by the decoder
            are replaced with the last good reading before AHRS processing.
        calibrate_gyro_bias: when True, the mean gyro reading over
            `stationary_init_seconds` is treated as the bias and subtracted
            from all gyro samples. This is the single biggest improvement
            over plain Madgwick for short recordings — without it, gyro
            bias slowly tilts the world Z axis, leaking gravity into the
            vertical channel as a slow sinusoid that the 1/ω² integrator
            then amplifies dramatically.

        magnitude_mad_threshold: threshold (in MAD units) for the robust
            accel-magnitude outlier filter. Samples whose |accel| deviates
            from the global median by more than this × MAD are replaced
            with the last good reading. Default 8.0 is conservative — it
            catches end-of-stream garbage spikes (where |a| jumps to 5g+
            for a few samples) without rejecting real motion (typically
            within ±2-3 g of the 1g resting magnitude). Set to math.inf to
            disable.

        hampel_window_seconds: window length (seconds) for the LOCAL
            neighbour-based spike detector (using decoder's
            detect_outliers_stdcheck). Catches spikes that the global MAD
            step above misses when there's a burst of real motion (which
            widens the global MAD). Default 0.25 s. Set to 0 to disable.

        hampel_k_mad: sigma threshold for the local spike detector. A
            sample is flagged when |sample - local mean| > k × local std.
            Default 5.0 is moderately aggressive; lower (3-4) is more
            aggressive but risks rejecting real fast motion edges; higher
            (8-10) is conservative.

        use_trust_points_bias: when True (default), runs the time-varying
            gyro-bias smoother BEFORE the Madgwick loop. It finds every
            "trust point" (sample where |accel| ≈ 1g), groups them into
            stationary runs, computes mean gyro = bias estimate for each
            run, and linearly interpolates between runs to give a smooth
            time-varying bias estimate across the whole recording. This
            corrects for slow thermal drift that a one-shot start-of-record
            bias cal can't catch — and unlike Madgwick's online correction,
            it cannot be fooled by linear acceleration because it only ever
            looks at samples where |accel| is consistent with pure gravity.
            Set to False to disable and use only the fixed init-window bias.

        trust_points_gravity_tolerance: how close |accel| must be to 9.81 m/s²
            for a sample to be considered a "trust point" (in m/s²). Default
            0.5 (≈ 0.05g). Tighten (0.2) for stricter gating, loosen (1.0)
            for more anchors at the cost of letting some motion samples in.

        trust_points_min_seconds: stationary runs shorter than this are
            discarded — too short to average down gyro noise meaningfully.
            Default 0.5 s. At 225 Hz that's 113 samples; gyro noise → bias
            noise of ~σ/√113 ≈ σ/10.6.

        motion_gate_threshold: gate (m/s²) for Madgwick accel correction.
            Madgwick assumes accel = gravity, which breaks during linear
            acceleration and creates spurious attitude spikes (visible as
            roll/pitch jumps that have NO corresponding gyro signal).
            When |accel| deviates from 9.81 m/s² by more than this gate,
            the accel-based correction is skipped for that sample and only
            gyro propagates. Default 1.5 m/s² (≈ 0.15 g) is moderate. Set
            very large (e.g. 100) to disable the gate and trust accel
            always; set 0 to disable accel correction entirely.

        detrend_polynomial_order: order of the least-squares polynomial fit
            subtracted from the vertical acceleration before integration.
            Replaces the low-frequency edge of the previous band-pass: a
            Butterworth high-pass at `low_hz` has settling transient ≈ 3/
            low_hz, so for short recordings (< 1 minute) the entire signal
            is inside the transient region and you see filter ringing as a
            spurious slow oscillation. Polynomial detrend has no transient
            — order 3 (default) removes DC + linear + parabolic + cubic
            trends, which covers slow gyro-bias drift and AHRS-convergence
            artifacts. Set to 0 to remove only DC, or to a negative value
            to disable entirely.

        fft_taper_fraction: fraction of the signal length tapered at each
            end with a Tukey (cosine) window before the FFT integration.
            Suppresses spectral leakage from the implicit edge step the FFT
            sees as periodic-extension discontinuity — even after polynomial
            detrending a tiny start-vs-end mismatch rings at the low-cutoff
            frequency. Default 0.1 = 10% taper on each side. The integrated
            output near the tapered edges (~10% at each end) is not
            quantitatively meaningful; the interior 80% is clean. Set to
            0.0 to disable.

        cutoff_softness: how gentle the FFT band-pass cutoffs are. A hard
            cutoff (softness=0) is a step in frequency space and by Fourier
            duality is a long sinc ring in the time domain — visible as a
            spurious oscillation at the cutoff frequency. softness=0.3
            (default) replaces each cutoff with a raised-cosine ramp
            spanning ±30% of the centre frequency: low_hz=0.05 ramps from
            0.035 to 0.065 Hz, high_hz=2.5 ramps from 1.75 to 3.25 Hz.
            Larger values = smoother frequency response but more in-band
            attenuation near the edges. Set to 0.0 for the old hard cutoff.
    """
    # --- Pull raw IMU arrays
    imu_us = np.asarray(data["imu_micros_unwrapped"], dtype=np.float64)
    n_raw = len(imu_us)
    if n_raw < 100:
        raise ValueError(f"Not enough IMU samples ({n_raw})")
    t_raw = (imu_us - imu_us[0]) * 1e-6  # seconds from start

    mg_to_ms2 = G_STD / 1000.0
    mdps_to_radps = (np.pi / 180.0) / 1000.0
    acc = np.column_stack([
        data["imu_acc_x"] * mg_to_ms2,
        data["imu_acc_y"] * mg_to_ms2,
        data["imu_acc_z"] * mg_to_ms2,
    ])
    gyr = np.column_stack([
        data["imu_gyr_x"] * mdps_to_radps,
        data["imu_gyr_y"] * mdps_to_radps,
        data["imu_gyr_z"] * mdps_to_radps,
    ])

    # --- Outlier filtering (3-stage hold-last-good)
    acc, gyr = _filter_imu_outliers(
        t_raw, acc, gyr, data,
        use_outlier_flags=use_outlier_flags,
        magnitude_mad_threshold=magnitude_mad_threshold,
        hampel_window_seconds=hampel_window_seconds,
        hampel_k_mad=hampel_k_mad,
    )

    # --- Initial attitude + gyro bias from first N seconds of stationary data
    dt_med = float(np.median(np.diff(t_raw)))
    n_init = max(1, min(n_raw, int(stationary_init_seconds / max(dt_med, 1e-6))))
    a_mean = acc[:n_init].mean(axis=0)
    state = MadgwickState(q=initial_attitude_from_accel(a_mean))
    logger.info(
        f"Initial attitude from first {n_init} samples (~{n_init * dt_med:.1f}s): "
        f"a_mean={a_mean.round(3)}, q0={state.q.round(3)}"
    )
    if calibrate_gyro_bias:
        gyro_bias = gyr[:n_init].mean(axis=0)
        gyr = gyr - gyro_bias
        logger.info(
            f"Estimated gyro bias from stationary init: "
            f"{np.rad2deg(gyro_bias).round(3)} deg/s — subtracted from all samples"
        )

    # --- Time-varying gyro bias smoother using accel trust points.
    # The fixed-bias step above uses only the first `stationary_init_seconds`
    # of data. This step looks for ALL "trust points" in the recording (any
    # sample where |accel| ≈ g, meaning no linear acceleration), groups them
    # into stationary runs, computes mean gyro = bias estimate for each run,
    # and linearly interpolates the bias between runs. The result is a
    # SMOOTHED bias correction that follows thermal / time drift across the
    # whole recording, not just a one-shot start-of-recording estimate.
    # This is the offline "smoother" complement to Madgwick's online filter.
    n_trust_anchors = 0
    if use_trust_points_bias:
        anchor_t, anchor_b = estimate_time_varying_gyro_bias(
            t_raw, acc, gyr,
            gravity_tolerance_mps2=trust_points_gravity_tolerance,
            min_stationary_seconds=trust_points_min_seconds,
        )
        if anchor_t is not None and len(anchor_t) > 0:
            n_trust_anchors = len(anchor_t)
            gyr = apply_time_varying_gyro_bias(t_raw, gyr, anchor_t, anchor_b)
            logger.info(
                f"Trust-points bias smoother: {n_trust_anchors} stationary "
                f"anchors over {t_raw[-1]:.1f}s (tolerance ±"
                f"{trust_points_gravity_tolerance} m/s² from 1g, min run "
                f"{trust_points_min_seconds}s); anchor biases (deg/s): "
                f"{np.rad2deg(anchor_b).round(3).tolist()}"
            )
        else:
            logger.info(
                f"Trust-points bias smoother: no stationary runs ≥"
                f"{trust_points_min_seconds}s found (recording may be all "
                f"motion); falling back to single fixed bias"
            )

    # --- Run Madgwick at native IMU timestamps
    quat_log = np.zeros((n_raw, 4))
    quat_log[0] = state.q
    n_motion_gated = 0
    for i in range(1, n_raw):
        dt = t_raw[i] - t_raw[i - 1]
        if dt <= 0 or dt > 1.0:
            # Bad timestamp / gigantic gap — skip step, keep last attitude
            quat_log[i] = state.q
            continue
        # Substep large gaps so quaternion integration stays well-conditioned
        if dt > 0.05:
            n_sub = int(np.ceil(dt / 0.05))
            sub_dt = dt / n_sub
            for _ in range(n_sub):
                used = madgwick_step_imu(
                    state, acc[i], gyr[i], sub_dt,
                    beta=beta, motion_gate_threshold=motion_gate_threshold,
                )
        else:
            used = madgwick_step_imu(
                state, acc[i], gyr[i], dt,
                beta=beta, motion_gate_threshold=motion_gate_threshold,
            )
        if not used:
            n_motion_gated += 1
        quat_log[i] = state.q

    # --- Rotate acc body→world, subtract gravity, take z-up
    # scipy expects (x, y, z, w) order
    rots = Rotation.from_quat(np.column_stack([
        quat_log[:, 1], quat_log[:, 2], quat_log[:, 3], quat_log[:, 0],
    ]))
    acc_world = rots.apply(acc)  # (N, 3), world frame z-down (specific force)
    # Inertial accel = specific force + gravity. In NED with z-down, gravity
    # vector is +z (9.81 pointing down). At rest, body reads (0,0,-9.81),
    # world reads the same after R≈I; adding +g gives (0,0,0) — no motion. ✓
    # (NOT subtract — that would give 2g and double-count the gravity term.)
    acc_world_linear = acc_world + np.array([0.0, 0.0, G_STD])
    # Flip sign so "up" is positive (intuitive for users)
    acc_z_up_raw = -acc_world_linear[:, 2]

    # Diagnostic angles
    eul = rots.as_euler("ZYX")  # yaw, pitch, roll
    roll_deg = np.rad2deg(eul[:, 2])
    pitch_deg = np.rad2deg(eul[:, 1])

    # --- Resample onto uniform grid for FFT
    if resample_hz is None:
        fs = 1.0 / dt_med
    else:
        fs = float(resample_hz)
    n_uniform = int(np.floor((t_raw[-1] - t_raw[0]) * fs)) + 1
    t_uniform = t_raw[0] + np.arange(n_uniform) / fs
    acc_z_up = np.interp(t_uniform, t_raw, acc_z_up_raw)
    roll_uniform = np.interp(t_uniform, t_raw, roll_deg)
    pitch_uniform = np.interp(t_uniform, t_raw, pitch_deg)

    # --- Detrend (slow drift) + low-pass (high-frequency noise)
    # Replaces what used to be a Butterworth band-pass on the low side. The
    # high-pass component of a band-pass at f_c has a settling transient of
    # ~3/f_c seconds, which dominates short recordings; a least-squares
    # polynomial subtraction has no transient and removes the same kind of
    # slow drift cleanly (DC offset, linear, parabolic, cubic).
    if detrend_polynomial_order >= 0:
        acc_z_detrended = polynomial_detrend(acc_z_up, detrend_polynomial_order)
    else:
        acc_z_detrended = acc_z_up
    # Low-pass for plotting / inspection. The FFT integrator below applies
    # its own spectral band-pass for the actual integration step.
    acc_z_bp = lowpass_zero_phase(acc_z_detrended, fs, high_hz, order=4)
    # Frequency-domain double integration. The integrator still uses a
    # spectral band-pass internally — zeroing the DC + sub-low_hz bins is
    # necessary to keep 1/ω² well-defined; this is *not* the same as a
    # time-domain Butterworth and doesn't have the same edge transient.
    # Tukey-taper the input to suppress edge-step spectral leakage at the
    # low cutoff (see fft_taper_fraction docstring).
    vel_z, disp_z = fft_double_integrate_bandpass(
        acc_z_detrended, fs, low_hz, high_hz,
        taper_fraction=fft_taper_fraction,
        cutoff_softness=cutoff_softness,
    )

    if n_motion_gated:
        logger.info(
            f"Motion-gating: skipped accel-correction on {n_motion_gated} samples "
            f"({100*n_motion_gated/max(n_raw,1):.1f}% — |accel| was >"
            f" {motion_gate_threshold} m/s² off from 1g)"
        )
    logger.success(
        f"Vertical AHRS complete: {n_raw} IMU samples → {n_uniform} uniform samples "
        f"at {fs:.1f} Hz; band [{low_hz}, {high_hz}] Hz; "
        f"σ_disp = {np.std(disp_z) * 1000:.1f} mm"
    )

    return VerticalAHRSResult(
        t=t_uniform - t_uniform[0],
        accel_z_up=acc_z_bp,
        accel_z_up_raw=acc_z_up,
        velocity_z_up=vel_z,
        displacement_z_up=disp_z,
        roll_deg=roll_uniform,
        pitch_deg=pitch_uniform,
        fs_hz=fs,
        low_hz=low_hz,
        high_hz=high_hz,
    )


def compute_vertical_motion_lowpass_gravity(
    data: dict,
    low_hz: float = 0.05,
    high_hz: float = 2.5,
    gravity_cutoff_hz: float = 0.02,
    butter_order: int = 4,
    stationary_init_seconds: float = 2.0,
    calibrate_gyro_bias: bool = True,
    resample_hz: float | None = None,
    use_outlier_flags: bool = True,
    magnitude_mad_threshold: float = 8.0,
    hampel_window_seconds: float = 0.25,
    hampel_k_mad: float = 5.0,
    detrend_polynomial_order: int = 3,
    fft_taper_fraction: float = 0.1,
    cutoff_softness: float = 0.3,
) -> VerticalAHRSResult:
    """Complementary-filter AHRS: gyro tracks fast attitude, low-pass accel anchors slow gravity.

    Architecture
    ------------
    1. Gyro integration → q_gyro(t). Tracks rotation cleanly (no accel
       contamination), but accumulates slow drift from residual bias.
    2. Low-pass body-frame accel at `gravity_cutoff_hz` → slow-gravity-reference
       roll/pitch from the filtered gravity direction. Yaw is unobservable from
       accel; left to the gyro.
    3. Complementary filter on roll & pitch:
           roll(t)  = roll_gyro(t)  − low_pass(roll_gyro  − roll_accel_lp)
           pitch(t) = pitch_gyro(t) − low_pass(pitch_gyro − pitch_accel_lp)
       Equivalently: high_pass(gyro) + low_pass(accel). Above the crossover, gyro
       dominates (fast rotations tracked cleanly). Below, accel dominates (slow
       drift gets killed by the gravity anchor).
    4. Build corrected attitude (yaw = gyro), rotate body→world accel, subtract
       gravity, polynomial-detrend, FFT-bandpass-double-integrate to displacement.

    Why this works for combined rotation + translation (wave buoys, basin tests)
    --------------------------------------------------------------------------
    - Fast rotation tracking is the gyro's job — it doesn't care about lever-arm
      centripetal or linear accel, so no spurious "correction" gets injected
      during rocking-while-translating motion (which is exactly what kills Madgwick).
    - Slow gravity reference comes from low-passed accel. Even with constant
      linear-accel contamination, averaging over many oscillations recovers the
      true slow gravity direction (because linear accel is zero-mean over wave
      cycles for a freely-floating buoy).
    - No motion gate. No beta. No tuning fights between gyro and accel.

    Trade-offs
    ----------
      + Robust under simultaneous rotation and translation.
      + Gyro drift bounded by the low-pass crossover scale (~1/gravity_cutoff_hz).
      - Euler-angle complementary filter has a singularity at pitch ≈ ±90°. For
        wave buoys this is irrelevant; for aggressive hand-flip tests it breaks.
        Use `compute_vertical_motion` (Madgwick) for those.
      - The accel side assumes the device's TIME-AVERAGED orientation matches true
        gravity (free-floating buoy: yes; held upside-down indefinitely: no).
      - Recordings shorter than ~3/gravity_cutoff_hz have edge artifacts.

    Args:
        data: dict from `load_data_as_arrays`.
        low_hz, high_hz: band-pass edges for displacement integration.
        gravity_cutoff_hz: complementary-filter crossover (Hz). Below this freq,
            attitude follows the low-pass accel; above, attitude follows the gyro.
            Default 0.02 Hz for a 0.05 Hz wave band (2.5× separation). Tighten to
            0.01 Hz for narrower separation; raise to 0.04 Hz for faster drift
            correction at the cost of some wave-band leakage into the attitude.
        butter_order: order of the gravity-tracking Butterworth.
        stationary_init_seconds: window at the start used for initial attitude
            and gyro-bias subtraction. Keep the device still during this window.
        calibrate_gyro_bias: subtract mean gyro reading over the init window
            from all gyro samples. Default True.
        ... (other args mirror `compute_vertical_motion`) ...
    """
    imu_us = np.asarray(data["imu_micros_unwrapped"], dtype=np.float64)
    n_raw = len(imu_us)
    if n_raw < 100:
        raise ValueError(f"Not enough IMU samples ({n_raw})")
    t_raw = (imu_us - imu_us[0]) * 1e-6

    mg_to_ms2 = G_STD / 1000.0
    mdps_to_radps = (np.pi / 180.0) / 1000.0
    acc = np.column_stack([
        data["imu_acc_x"] * mg_to_ms2,
        data["imu_acc_y"] * mg_to_ms2,
        data["imu_acc_z"] * mg_to_ms2,
    ])
    gyr = np.column_stack([
        data["imu_gyr_x"] * mdps_to_radps,
        data["imu_gyr_y"] * mdps_to_radps,
        data["imu_gyr_z"] * mdps_to_radps,
    ])

    acc, gyr = _filter_imu_outliers(
        t_raw, acc, gyr, data,
        use_outlier_flags=use_outlier_flags,
        magnitude_mad_threshold=magnitude_mad_threshold,
        hampel_window_seconds=hampel_window_seconds,
        hampel_k_mad=hampel_k_mad,
    )

    # --- Initial attitude + gyro bias from first N seconds of (assumed) stationary data
    dt_med = float(np.median(np.diff(t_raw)))
    n_init = max(1, min(n_raw, int(stationary_init_seconds / max(dt_med, 1e-6))))
    a_mean = acc[:n_init].mean(axis=0)
    q_init = initial_attitude_from_accel(a_mean)
    logger.info(
        f"Initial attitude from first {n_init} samples (~{n_init * dt_med:.1f}s): "
        f"a_mean={a_mean.round(3)}, q0={q_init.round(3)}"
    )
    if calibrate_gyro_bias:
        gyro_bias = gyr[:n_init].mean(axis=0)
        gyr = gyr - gyro_bias
        logger.info(
            f"Estimated gyro bias from stationary init: "
            f"{np.rad2deg(gyro_bias).round(3)} deg/s — subtracted from all samples"
        )

    # --- Resample onto uniform grid (sosfiltfilt + simple gyro integrator need it)
    fs = float(resample_hz) if resample_hz is not None else 1.0 / dt_med
    n_uniform = int(np.floor((t_raw[-1] - t_raw[0]) * fs)) + 1
    t_uniform = t_raw[0] + np.arange(n_uniform) / fs
    acc_uniform = np.column_stack([
        np.interp(t_uniform, t_raw, acc[:, 0]),
        np.interp(t_uniform, t_raw, acc[:, 1]),
        np.interp(t_uniform, t_raw, acc[:, 2]),
    ])
    gyr_uniform = np.column_stack([
        np.interp(t_uniform, t_raw, gyr[:, 0]),
        np.interp(t_uniform, t_raw, gyr[:, 1]),
        np.interp(t_uniform, t_raw, gyr[:, 2]),
    ])

    # --- Step 1: pure gyro attitude integration on the uniform grid
    # q̇ = 0.5 · q ⊗ (0, ω_body). Forward Euler + renormalise — fine at our dt.
    quat_gyro = np.zeros((n_uniform, 4))  # (w, x, y, z)
    quat_gyro[0] = q_init
    q = q_init.copy()
    dt_uniform = 1.0 / fs
    for i in range(1, n_uniform):
        wx, wy, wz = gyr_uniform[i]
        q0, q1, q2, q3 = q
        qdot = 0.5 * np.array([
            -q1 * wx - q2 * wy - q3 * wz,
            +q0 * wx + q2 * wz - q3 * wy,
            +q0 * wy - q1 * wz + q3 * wx,
            +q0 * wz + q1 * wy - q2 * wx,
        ])
        q = q + qdot * dt_uniform
        q /= np.linalg.norm(q)
        quat_gyro[i] = q

    # --- Step 2: gyro-only Euler angles (scipy wants (x, y, z, w))
    rots_gyro = Rotation.from_quat(quat_gyro[:, [1, 2, 3, 0]])
    eul_gyro = rots_gyro.as_euler("ZYX")  # [yaw, pitch, roll]
    yaw_gyro = eul_gyro[:, 0]
    pitch_gyro = eul_gyro[:, 1]
    roll_gyro = eul_gyro[:, 2]

    # --- Step 3: slow-gravity reference from low-passed accel
    # Specific force convention: at rest, accel = -gravity. So the slow part of
    # accel is dominated by -gravity (linear accel averages out in the wave band).
    sos_lp = butter(butter_order, gravity_cutoff_hz, btype="lowpass",
                    fs=fs, output="sos")
    f_lp = np.column_stack([
        sosfiltfilt(sos_lp, acc_uniform[:, 0]),
        sosfiltfilt(sos_lp, acc_uniform[:, 1]),
        sosfiltfilt(sos_lp, acc_uniform[:, 2]),
    ])
    g_body_lp = -f_lp
    g_norm = np.maximum(np.linalg.norm(g_body_lp, axis=1), 1e-9)
    g_unit_lp = g_body_lp / g_norm[:, None]
    roll_acc_lp = np.arctan2(g_unit_lp[:, 1], g_unit_lp[:, 2])
    pitch_acc_lp = np.arctan2(
        -g_unit_lp[:, 0],
        np.sqrt(g_unit_lp[:, 1] ** 2 + g_unit_lp[:, 2] ** 2),
    )

    # --- Step 4: complementary filter on roll and pitch.
    # The slow-drift estimate is low_pass(gyro − accel_lp). Subtract from gyro to
    # get the corrected attitude. Equivalent to high-pass(gyro) + low-pass(accel),
    # but expressed this way it makes the "we use accel to correct gyro drift"
    # logic visible.
    drift_roll = sosfiltfilt(sos_lp, roll_gyro - roll_acc_lp)
    drift_pitch = sosfiltfilt(sos_lp, pitch_gyro - pitch_acc_lp)
    roll_corrected = roll_gyro - drift_roll
    pitch_corrected = pitch_gyro - drift_pitch

    # --- Step 5: build corrected attitude quaternion (yaw stays as gyro)
    rot_corrected = Rotation.from_euler(
        "ZYX",
        np.column_stack([yaw_gyro, pitch_corrected, roll_corrected]),
    )

    # --- Step 6: rotate body accel → world, subtract gravity, take z-up
    acc_world = rot_corrected.apply(acc_uniform)
    acc_world_linear = acc_world + np.array([0.0, 0.0, G_STD])
    acc_z_up = -acc_world_linear[:, 2]

    # --- Step 7: detrend + low-pass for plotting + FFT bandpass-integrate
    if detrend_polynomial_order >= 0:
        acc_z_detrended = polynomial_detrend(acc_z_up, detrend_polynomial_order)
    else:
        acc_z_detrended = acc_z_up
    acc_z_bp = lowpass_zero_phase(acc_z_detrended, fs, high_hz, order=4)
    vel_z, disp_z = fft_double_integrate_bandpass(
        acc_z_detrended, fs, low_hz, high_hz,
        taper_fraction=fft_taper_fraction,
        cutoff_softness=cutoff_softness,
    )

    # --- Sanity check: warn if the recording is short relative to the crossover period
    rec_len = t_uniform[-1] - t_uniform[0]
    if rec_len < 3.0 / gravity_cutoff_hz:
        logger.warning(
            f"Recording length {rec_len:.1f}s is short relative to gravity "
            f"low-pass period (1/{gravity_cutoff_hz} = {1/gravity_cutoff_hz:.0f}s). "
            f"Complementary filter may be biased toward the gyro-only estimate "
            f"at the edges; consider raising gravity_cutoff_hz for short recordings."
        )

    logger.success(
        f"Complementary-filter vertical complete: {n_raw} IMU → {n_uniform} "
        f"uniform @ {fs:.1f} Hz; crossover {gravity_cutoff_hz} Hz; "
        f"band [{low_hz}, {high_hz}] Hz; σ_disp = {np.std(disp_z) * 1000:.1f} mm"
    )

    return VerticalAHRSResult(
        t=t_uniform - t_uniform[0],
        accel_z_up=acc_z_bp,
        accel_z_up_raw=acc_z_up,
        velocity_z_up=vel_z,
        displacement_z_up=disp_z,
        roll_deg=np.rad2deg(roll_corrected),
        pitch_deg=np.rad2deg(pitch_corrected),
        fs_hz=fs,
        low_hz=low_hz,
        high_hz=high_hz,
    )


def compute_vertical_motion_savgol_detrend(
    data: dict,
    low_hz: float = 0.05,
    high_hz: float = 2.5,
    savgol_window_seconds: float = 10.0,
    savgol_polyorder: int = 3,
    stationary_init_seconds: float = 2.0,
    calibrate_gyro_bias: bool = True,
    resample_hz: float | None = None,
    use_outlier_flags: bool = True,
    magnitude_mad_threshold: float = 8.0,
    hampel_window_seconds: float = 0.25,
    hampel_k_mad: float = 5.0,
    detrend_polynomial_order: int = 3,
    fft_taper_fraction: float = 0.1,
    cutoff_softness: float = 0.3,
) -> VerticalAHRSResult:
    """Gyro AHRS with Savitzky-Golay detrending on attitude.

    Assumes the device's TIME-AVERAGED orientation is level (gravity along
    world Z). All slow trends in the gyro-integrated Euler angles are treated
    as drift to be subtracted, leaving only the fast wave-band attitude motion.

    Method:
      1. Integrate gyro → q_gyro(t). Drift-prone but clean over short windows.
      2. Extract Euler (yaw, pitch, roll) from q_gyro.
      3. Savitzky-Golay filter on each angle (long window, low polynomial order)
         extracts the slow trend.
      4. Subtract trend: clean = gyro - savgol(gyro). Effective zero-phase
         high-pass at roughly 1/window Hz.
      5. Re-build quaternion from (yaw_clean, pitch_clean, roll_clean).
      6. Apply attitude to body accel → world → subtract gravity → integrate.

    Comparison to other methods:
      - vs Madgwick: no accel-based attitude correction; no motion gate. Gyro
        alone handles fast tracking; the detrend kills slow drift.
      - vs complementary filter (lowpass_gravity): does NOT use accel as the
        slow gravity reference at all. Robust to accel contamination from
        lever-arm centripetal — but unlike the complementary filter, cannot
        recover the true slow attitude; it forces the mean attitude to zero.

    Trade-offs:
      + Simpler — one tuning knob (window size).
      + Robust to lever-arm centripetal (accel direction never used for attitude).
      + Eliminates the systematic gravity-leakage problem: any mean attitude
        error from imperfect bias subtraction is detrended away.
      - Assumes device's mean attitude is level. Wrong for a tilted-buoy
        deployment or a hand-held device held at a fixed tilt.
      - Euler representation has the usual pitch ≈ ±90° singularity.
      - Savgol edge regions (first and last `window/2`) have larger error.

    Args:
        data: dict from `load_data_as_arrays`.
        low_hz, high_hz: band-pass edges for the displacement integrator.
        savgol_window_seconds: length of the Savitzky-Golay window that
            extracts the slow attitude trend. Effective high-pass cutoff is
            ~1/window. Default 10 s (cutoff ~0.1 Hz) — catches the 0.05–0.1 Hz
            leakage band that ate the previous methods. Widen to 20–30 s for
            true wave buoys where slower real attitude changes matter; tighten
            to 5 s to remove more aggressive drifts.
        savgol_polyorder: polynomial order of the Savitzky-Golay filter.
            Default 3 (cubic). Order 2 (parabolic) is more aggressive on slow
            content; order 4–5 preserves more curvature.
        ... (other args mirror `compute_vertical_motion_lowpass_gravity`) ...
    """
    imu_us = np.asarray(data["imu_micros_unwrapped"], dtype=np.float64)
    n_raw = len(imu_us)
    if n_raw < 100:
        raise ValueError(f"Not enough IMU samples ({n_raw})")
    t_raw = (imu_us - imu_us[0]) * 1e-6

    mg_to_ms2 = G_STD / 1000.0
    mdps_to_radps = (np.pi / 180.0) / 1000.0
    acc = np.column_stack([
        data["imu_acc_x"] * mg_to_ms2,
        data["imu_acc_y"] * mg_to_ms2,
        data["imu_acc_z"] * mg_to_ms2,
    ])
    gyr = np.column_stack([
        data["imu_gyr_x"] * mdps_to_radps,
        data["imu_gyr_y"] * mdps_to_radps,
        data["imu_gyr_z"] * mdps_to_radps,
    ])

    acc, gyr = _filter_imu_outliers(
        t_raw, acc, gyr, data,
        use_outlier_flags=use_outlier_flags,
        magnitude_mad_threshold=magnitude_mad_threshold,
        hampel_window_seconds=hampel_window_seconds,
        hampel_k_mad=hampel_k_mad,
    )

    # --- Init: attitude + gyro bias from first N seconds
    dt_med = float(np.median(np.diff(t_raw)))
    n_init = max(1, min(n_raw, int(stationary_init_seconds / max(dt_med, 1e-6))))
    a_mean = acc[:n_init].mean(axis=0)
    q_init = initial_attitude_from_accel(a_mean)
    logger.info(
        f"Initial attitude from first {n_init} samples (~{n_init * dt_med:.1f}s): "
        f"a_mean={a_mean.round(3)}, q0={q_init.round(3)}"
    )
    if calibrate_gyro_bias:
        gyro_bias = gyr[:n_init].mean(axis=0)
        gyr = gyr - gyro_bias
        logger.info(
            f"Estimated gyro bias from stationary init: "
            f"{np.rad2deg(gyro_bias).round(3)} deg/s — subtracted from all samples"
        )

    # --- Resample to uniform grid (savgol requires it)
    fs = float(resample_hz) if resample_hz is not None else 1.0 / dt_med
    n_uniform = int(np.floor((t_raw[-1] - t_raw[0]) * fs)) + 1
    t_uniform = t_raw[0] + np.arange(n_uniform) / fs
    acc_uniform = np.column_stack([
        np.interp(t_uniform, t_raw, acc[:, 0]),
        np.interp(t_uniform, t_raw, acc[:, 1]),
        np.interp(t_uniform, t_raw, acc[:, 2]),
    ])
    gyr_uniform = np.column_stack([
        np.interp(t_uniform, t_raw, gyr[:, 0]),
        np.interp(t_uniform, t_raw, gyr[:, 1]),
        np.interp(t_uniform, t_raw, gyr[:, 2]),
    ])

    # --- Pure gyro attitude integration
    quat_gyro = np.zeros((n_uniform, 4))
    quat_gyro[0] = q_init
    q = q_init.copy()
    dt_uniform = 1.0 / fs
    for i in range(1, n_uniform):
        wx, wy, wz = gyr_uniform[i]
        q0, q1, q2, q3 = q
        qdot = 0.5 * np.array([
            -q1 * wx - q2 * wy - q3 * wz,
            +q0 * wx + q2 * wz - q3 * wy,
            +q0 * wy - q1 * wz + q3 * wx,
            +q0 * wz + q1 * wy - q2 * wx,
        ])
        q = q + qdot * dt_uniform
        q /= np.linalg.norm(q)
        quat_gyro[i] = q

    # --- Extract gyro-only Euler angles
    rots_gyro = Rotation.from_quat(quat_gyro[:, [1, 2, 3, 0]])
    eul_gyro = rots_gyro.as_euler("ZYX")  # [yaw, pitch, roll]
    yaw_gyro = eul_gyro[:, 0]
    pitch_gyro = eul_gyro[:, 1]
    roll_gyro = eul_gyro[:, 2]

    # --- Savitzky-Golay detrend on each Euler component
    # Window must be odd and > polyorder. For very short recordings clamp to
    # something reasonable.
    win = int(round(savgol_window_seconds * fs))
    if win % 2 == 0:
        win += 1
    win = max(savgol_polyorder + 2, min(win, n_uniform - 1))
    if win % 2 == 0:
        win -= 1
    if win < savgol_polyorder + 2:
        raise ValueError(
            f"Recording too short for savgol_window_seconds={savgol_window_seconds}"
            f" at fs={fs:.1f} Hz with savgol_polyorder={savgol_polyorder}"
        )
    yaw_trend = savgol_filter(yaw_gyro, win, savgol_polyorder)
    pitch_trend = savgol_filter(pitch_gyro, win, savgol_polyorder)
    roll_trend = savgol_filter(roll_gyro, win, savgol_polyorder)
    yaw_clean = yaw_gyro - yaw_trend
    pitch_clean = pitch_gyro - pitch_trend
    roll_clean = roll_gyro - roll_trend

    logger.info(
        f"Savgol detrend: window={win} samples ({win/fs:.1f}s ≈ "
        f"1/{fs/win:.3f} Hz high-pass), polyorder={savgol_polyorder}; "
        f"removed trend std: roll={np.rad2deg(roll_trend.std()):.2f}°, "
        f"pitch={np.rad2deg(pitch_trend.std()):.2f}°, "
        f"yaw={np.rad2deg(yaw_trend.std()):.2f}°"
    )

    # --- Re-build quaternion from cleaned Euler
    rot_clean = Rotation.from_euler(
        "ZYX",
        np.column_stack([yaw_clean, pitch_clean, roll_clean]),
    )

    # --- Rotate body accel → world, subtract gravity, take z-up
    acc_world = rot_clean.apply(acc_uniform)
    acc_world_linear = acc_world + np.array([0.0, 0.0, G_STD])
    acc_z_up = -acc_world_linear[:, 2]

    # --- Detrend + low-pass for plotting + FFT bandpass-integrate
    if detrend_polynomial_order >= 0:
        acc_z_detrended = polynomial_detrend(acc_z_up, detrend_polynomial_order)
    else:
        acc_z_detrended = acc_z_up
    acc_z_bp = lowpass_zero_phase(acc_z_detrended, fs, high_hz, order=4)
    vel_z, disp_z = fft_double_integrate_bandpass(
        acc_z_detrended, fs, low_hz, high_hz,
        taper_fraction=fft_taper_fraction,
        cutoff_softness=cutoff_softness,
    )


    logger.success(
        f"Savgol-detrend vertical complete: {n_raw} IMU → {n_uniform} uniform "
        f"@ {fs:.1f} Hz; detrend window {win/fs:.1f}s; band [{low_hz}, {high_hz}] Hz; "
        f"σ_disp = {np.std(disp_z) * 1000:.1f} mm"
    )

    return VerticalAHRSResult(
        t=t_uniform - t_uniform[0],
        accel_z_up=acc_z_bp,
        accel_z_up_raw=acc_z_up,
        velocity_z_up=vel_z,
        displacement_z_up=disp_z,
        roll_deg=np.rad2deg(roll_clean),
        pitch_deg=np.rad2deg(pitch_clean),
        fs_hz=fs,
        low_hz=low_hz,
        high_hz=high_hz,
    )
