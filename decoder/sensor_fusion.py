"""15-state Error-State Extended Kalman Filter for IMU + GNSS + magnetometer fusion.

Produces a fused estimate of position (including vertical), velocity, attitude,
and sensor biases by combining the OLA logger's ICM-20948 (accelerometer +
gyroscope + magnetometer) with the MAX-M10S GNSS receiver.

Architecture
------------
This is a *loosely-coupled* ESKF — the GNSS receiver's PVT output (position,
velocity) is treated as a noisy measurement, not raw pseudoranges. This is the
standard architecture for consumer-grade GPS/INS systems and is what every
commercial unit at this price point uses.

State vector (nominal, 16D):
    p      (3) — position in local NED frame, metres (origin: first GNSS fix)
    v      (3) — velocity in NED frame, m/s
    q      (4) — attitude quaternion, body→NED (Hamilton convention)
    b_a    (3) — accelerometer bias, m/s²
    b_g    (3) — gyroscope bias, rad/s

Error state (15D, what the EKF actually propagates):
    δp     (3) — position error
    δv     (3) — velocity error
    δθ     (3) — small-angle attitude error (3D, axis-angle)
    δb_a   (3) — accel bias error
    δb_g   (3) — gyro bias error

Prediction (at IMU rate ~100-225 Hz):
    p_{k+1} = p_k + v_k·dt + 0.5·(R_k·(a_m-b_a) + g)·dt²
    v_{k+1} = v_k + (R_k·(a_m-b_a) + g)·dt
    q_{k+1} = q_k ⊗ Exp((ω_m-b_g)·dt)
    biases random-walk

Updates (loosely-coupled):
    GPS position (NED):   z = p,  R = diag(σ_xy², σ_xy², σ_z²)
    GPS velocity (NED):   z = v,  R = diag(σ_v², σ_v², σ_vz²)
    Magnetometer yaw:     z = ψ,  R = σ_ψ²   (1D update on heading only)

References
----------
- J. Solà, "Quaternion kinematics for the error-state KF", arXiv:1711.02508 (2017)
- P. Groves, "Principles of GNSS, Inertial, and Multisensor Integrated Navigation
  Systems", 2nd ed., Artech House 2013, ch. 13-14.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation


# ------------------------------- constants ---------------------------------

# Gravity vector in NED frame (z-down). |g| at 60°N sea level ≈ 9.819, common
# practice is to use the standard value and let bias estimation absorb the
# small residual.
G_NED = np.array([0.0, 0.0, 9.80665])
G_MAG = 9.80665

# Earth radii (WGS-84) for flat-earth lat/lon → NED conversion.
WGS84_A = 6378137.0  # semi-major axis
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)


# ----------------------- coordinate-frame helpers --------------------------


def lla_to_local_ned(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    alt_m: np.ndarray,
    ref_lat_deg: float,
    ref_lon_deg: float,
    ref_alt_m: float,
) -> np.ndarray:
    """Flat-earth lat/lon/alt → local NED offsets in metres.

    Uses local meridional / transverse radii of curvature. Accurate to ~cm at
    distances under a few kilometres, where the curvature of the Earth is
    negligible relative to GPS noise.

    Returns an (N, 3) array of [north, east, down] offsets from reference.
    """
    ref_lat_rad = np.deg2rad(ref_lat_deg)
    sin_lat = np.sin(ref_lat_rad)
    # Meridional / transverse radii at reference latitude
    rn = WGS84_A * (1 - WGS84_E2) / (1 - WGS84_E2 * sin_lat ** 2) ** 1.5
    re = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat ** 2)

    d_lat_rad = np.deg2rad(np.asarray(lat_deg) - ref_lat_deg)
    d_lon_rad = np.deg2rad(np.asarray(lon_deg) - ref_lon_deg)
    d_alt_m = np.asarray(alt_m) - ref_alt_m

    n = d_lat_rad * (rn + ref_alt_m)
    e = d_lon_rad * (re + ref_alt_m) * np.cos(ref_lat_rad)
    d = -d_alt_m  # NED: down is positive, altitude is up
    return np.column_stack([n, e, d])


def skew(v: np.ndarray) -> np.ndarray:
    """3x3 skew-symmetric (cross-product) matrix from a 3-vector."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


# ------------------------ initial attitude (AHRS) --------------------------


def attitude_from_accel_mag(
    accel_mean_body: np.ndarray,
    mag_mean_body: np.ndarray,
) -> Rotation:
    """Compute body→NED rotation from averaged accel + mag while stationary.

    Uses the TRIAD-like construction:
      - Gravity in NED is +z (down). Sensor measures specific force, so at rest
        the accel reading points up in body frame (opposite of g). We negate
        to recover the body-frame "down" direction.
      - The horizontal component of the magnetic field defines the heading
        (north). Vertical mag component (inclination) is discarded.

    This is the standard initialisation used by Madgwick/Mahony at startup.
    Caller should pass several seconds of stationary mean readings — noisy or
    moving data will give a poor initial attitude (filter will then take longer
    to converge).
    """
    # Body-frame "down" = direction the accelerometer's negative reading points
    down_b = -accel_mean_body / np.linalg.norm(accel_mean_body)

    # Remove gravity-aligned component from mag to get horizontal projection
    mag_n = mag_mean_body / np.linalg.norm(mag_mean_body)
    mag_horiz_b = mag_n - down_b * (mag_n @ down_b)
    mag_horiz_b /= np.linalg.norm(mag_horiz_b)

    # East = down × north (right-hand NED). north here is the horizontal mag.
    east_b = np.cross(down_b, mag_horiz_b)
    east_b /= np.linalg.norm(east_b)

    # Body-frame columns of NED basis (north, east, down)
    # R_n_b columns are body-frame components of NED axes ⇒ R_n_b = R_body→NED^T
    r_nav_axes_in_body = np.column_stack([mag_horiz_b, east_b, down_b])
    # We want R_body→NED, the rotation that takes a body vector to NED frame
    # That is the inverse (transpose) of the matrix whose columns are nav-axes-
    # expressed-in-body.
    rot_body_to_ned = r_nav_axes_in_body.T
    return Rotation.from_matrix(rot_body_to_ned)


# --------------------------------- ESKF -----------------------------------


@dataclass
class EKFNoiseParams:
    """Process and measurement noise parameters.

    Defaults are tuned for ICM-20948 + MAX-M10S. Tune up sigma_acc_noise /
    sigma_gyro_noise if you see the filter chasing IMU noise; tune up the GPS
    sigmas if the filter snaps to bad fixes.
    """

    # IMU white noise (continuous-time PSD → discrete via sqrt(dt))
    sigma_acc_noise: float = 0.05     # m/s²/√Hz
    sigma_gyro_noise: float = 0.005   # rad/s/√Hz

    # Bias random-walk (drives bias estimation; tune up if biases drift)
    sigma_acc_bias_walk: float = 1e-4   # m/s²/√s
    sigma_gyro_bias_walk: float = 1e-5  # rad/s/√s

    # GPS measurement noise (1-sigma)
    sigma_gps_pos_horiz: float = 2.0   # m
    sigma_gps_pos_vert: float = 5.0    # m (typically 2-3× worse than horizontal)
    sigma_gps_vel_horiz: float = 0.1   # m/s
    sigma_gps_vel_vert: float = 0.2    # m/s

    # Magnetometer yaw noise (after tilt compensation)
    sigma_mag_yaw_rad: float = np.deg2rad(5.0)


@dataclass
class EKFState:
    """Nominal-state container."""

    p: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    q: Rotation = field(default_factory=lambda: Rotation.identity())
    b_a: np.ndarray = field(default_factory=lambda: np.zeros(3))
    b_g: np.ndarray = field(default_factory=lambda: np.zeros(3))


class ErrorStateEKF:
    """15-state error-state Extended Kalman Filter.

    Usage:
        ekf = ErrorStateEKF()
        ekf.initialize(initial_state, P0)
        for sample in imu_stream:
            ekf.predict(accel_body, gyro_body, dt)
            # ... possibly fold in measurements:
            if new_gps_pos_available:
                ekf.update_gps_position(p_meas_ned)
            if new_gps_vel_available:
                ekf.update_gps_velocity(v_meas_ned)
            if new_mag_yaw_available:
                ekf.update_mag_yaw(mag_body)
    """

    # Error-state index layout
    IDX_P, IDX_V, IDX_TH, IDX_BA, IDX_BG = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 15)
    N_ERR = 15

    def __init__(self, noise: EKFNoiseParams | None = None):
        self.noise = noise or EKFNoiseParams()
        self.state = EKFState()
        # Initial covariance — set in initialize()
        self.P = np.eye(self.N_ERR) * 1e-3
        # For diagnostics
        self.last_gps_innovation: np.ndarray | None = None
        self.last_mag_innovation: float | None = None

    # ----- initialisation -----

    def initialize(self, state: EKFState, P0: np.ndarray) -> None:
        """Set initial nominal state and covariance."""
        self.state = state
        assert P0.shape == (self.N_ERR, self.N_ERR), \
            f"P0 must be {self.N_ERR}x{self.N_ERR}, got {P0.shape}"
        self.P = P0.copy()

    # ----- prediction -----

    def predict(
        self,
        accel_body: np.ndarray,
        gyro_body: np.ndarray,
        dt: float,
    ) -> None:
        """Propagate nominal state and covariance one IMU step.

        accel_body : measured specific force in body frame (m/s²)
        gyro_body  : measured angular rate in body frame (rad/s)
        dt         : time step (s)
        """
        s = self.state
        # Bias-corrected IMU
        a_hat = accel_body - s.b_a
        w_hat = gyro_body - s.b_g

        R_bn = s.q.as_matrix()  # body→NED
        a_ned = R_bn @ a_hat + G_NED  # specific force + gravity = inertial accel

        # --- nominal-state integration (mid-point on velocity, RK1 on the rest)
        p_new = s.p + s.v * dt + 0.5 * a_ned * dt * dt
        v_new = s.v + a_ned * dt
        # Attitude: q ⊗ Exp(ω dt). Using axis-angle directly.
        dtheta = w_hat * dt
        dq = Rotation.from_rotvec(dtheta)
        q_new = s.q * dq
        # Biases are random-walk; nominal value unchanged.

        # --- error-state transition (linearised, first-order)
        # F is the 15x15 state transition for the error state, derived from
        # the nominal dynamics. See Sola 2017 §7.
        F = np.eye(self.N_ERR)
        F[self.IDX_P, self.IDX_V] = np.eye(3) * dt
        F[self.IDX_V, self.IDX_TH] = -R_bn @ skew(a_hat) * dt
        F[self.IDX_V, self.IDX_BA] = -R_bn * dt
        F[self.IDX_TH, self.IDX_TH] = dq.inv().as_matrix()  # exp(-ωdt)
        F[self.IDX_TH, self.IDX_BG] = -np.eye(3) * dt

        # Process noise input matrix G (15x12): IMU white noise → vel/att,
        # bias random walks → biases.
        Q = np.zeros((self.N_ERR, self.N_ERR))
        # Velocity gets accel white noise rotated into NED, scaled by dt²
        Q[self.IDX_V, self.IDX_V] = (R_bn @ R_bn.T) * (self.noise.sigma_acc_noise ** 2) * dt
        # Attitude gets gyro white noise, scaled by dt²
        Q[self.IDX_TH, self.IDX_TH] = np.eye(3) * (self.noise.sigma_gyro_noise ** 2) * dt
        # Bias random walks
        Q[self.IDX_BA, self.IDX_BA] = np.eye(3) * (self.noise.sigma_acc_bias_walk ** 2) * dt
        Q[self.IDX_BG, self.IDX_BG] = np.eye(3) * (self.noise.sigma_gyro_bias_walk ** 2) * dt

        self.P = F @ self.P @ F.T + Q
        # Force symmetry; numerical drift bites otherwise
        self.P = 0.5 * (self.P + self.P.T)

        # Commit nominal
        s.p, s.v, s.q = p_new, v_new, q_new

    # ----- update steps -----

    def update_gps_position(self, p_meas_ned: np.ndarray, horizontal_only: bool = False) -> None:
        """Fold in a GPS position measurement (NED, metres).

        horizontal_only=True drops the Down component (typical when indoor /
        poor sky view makes GPS altitude unreliable). The vertical state is
        then only constrained by IMU integration (drifts; pair with a band-
        pass filter on the output if you only care about the AC component).
        """
        if horizontal_only:
            H = np.zeros((2, self.N_ERR))
            H[:, self.IDX_P.start:self.IDX_P.start + 2] = np.eye(2)
            R = np.diag([self.noise.sigma_gps_pos_horiz ** 2] * 2)
            innov = p_meas_ned[:2] - self.state.p[:2]
        else:
            H = np.zeros((3, self.N_ERR))
            H[:, self.IDX_P] = np.eye(3)
            R = np.diag([
                self.noise.sigma_gps_pos_horiz ** 2,
                self.noise.sigma_gps_pos_horiz ** 2,
                self.noise.sigma_gps_pos_vert ** 2,
            ])
            innov = p_meas_ned - self.state.p
        self._apply_update(H, R, innov)
        self.last_gps_innovation = innov

    def update_gps_velocity(self, v_meas_ned: np.ndarray, horizontal_only: bool = False) -> None:
        """Fold in a GPS velocity measurement (NED, m/s).

        horizontal_only=True drops the Down component (vel_down comes from
        the same satellite geometry as altitude, so when altitude is bad
        vel_down typically is too).
        """
        if horizontal_only:
            H = np.zeros((2, self.N_ERR))
            H[:, self.IDX_V.start:self.IDX_V.start + 2] = np.eye(2)
            R = np.diag([self.noise.sigma_gps_vel_horiz ** 2] * 2)
            innov = v_meas_ned[:2] - self.state.v[:2]
        else:
            H = np.zeros((3, self.N_ERR))
            H[:, self.IDX_V] = np.eye(3)
            R = np.diag([
                self.noise.sigma_gps_vel_horiz ** 2,
                self.noise.sigma_gps_vel_horiz ** 2,
                self.noise.sigma_gps_vel_vert ** 2,
            ])
            innov = v_meas_ned - self.state.v
        self._apply_update(H, R, innov)

    def update_static_anchor(
        self,
        sigma_horiz_m: float,
        sigma_vert_m: float | None = None,
        anchor_ned: np.ndarray | None = None,
    ) -> None:
        """Inject a "static-position pseudo-measurement" — tell the EKF that
        the device is approximately at `anchor_ned` (default = origin) with
        the given per-axis uncertainty.

        Used for GNSS-denied deployments where the device's mean position is
        approximately known and bounded (a moored buoy, a stationary indoor
        sensor, a hand-held device staying in one spot). Without something
        like this, the EKF's position state is unobservable from IMU alone
        and runs away at ω²·t² growth as integrated accel bias accumulates.

        Choosing the sigmas:
          - Tight (e.g. 0.1 m) → strong anchor; suppresses motion in the
            wave band. Use only for genuinely-stationary deployments.
          - Loose (e.g. 5 m) → allows wave-scale motion through unhindered
            while still bounding drift at ~5 m. Recommended default for
            wave buoys and human-scale-motion tests.
          - Per-axis: pass a smaller `sigma_horiz_m` and larger
            `sigma_vert_m` for a buoy where vertical motion exceeds
            horizontal drift.

        This is mathematically identical to a GPS-position update; we just
        call it through a separate method to make the *intent* (synthetic
        anchor vs. real-sky GPS) explicit at the call site.
        """
        if anchor_ned is None:
            anchor_ned = np.zeros(3)
        sv = sigma_vert_m if sigma_vert_m is not None else sigma_horiz_m
        H = np.zeros((3, self.N_ERR))
        H[:, self.IDX_P] = np.eye(3)
        R = np.diag([sigma_horiz_m ** 2, sigma_horiz_m ** 2, sv ** 2])
        innov = np.asarray(anchor_ned, dtype=np.float64) - self.state.p
        self._apply_update(H, R, innov)

    def update_mag_yaw(self, mag_body_uT: np.ndarray) -> None:
        """1D yaw update from magnetometer.

        We don't model magnetic inclination explicitly; we project the mag
        vector onto the local horizontal plane (using current attitude) and
        compare its heading to the predicted heading. This keeps the update
        linearisable and avoids hard-iron/soft-iron bias coupling into roll/
        pitch (which are already well-observed by accel).
        """
        R_bn = self.state.q.as_matrix()
        mag_ned = R_bn @ mag_body_uT
        # Measured heading from the horizontal mag projection
        psi_meas = np.arctan2(mag_ned[1], mag_ned[0])
        # Current heading from attitude (yaw of Z-Y-X Euler decomposition)
        psi_pred = self.state.q.as_euler("ZYX")[0]
        innov = wrap_to_pi(psi_meas - psi_pred)

        # Heading observation: H selects the yaw-component of δθ in NED.
        # For small-angle errors, ψ_meas - ψ_pred ≈ δθ_z (z = down).
        H = np.zeros((1, self.N_ERR))
        H[0, self.IDX_TH.start + 2] = 1.0
        R = np.array([[self.noise.sigma_mag_yaw_rad ** 2]])
        self._apply_update(H, R, np.array([innov]))
        self.last_mag_innovation = innov

    def _apply_update(self, H: np.ndarray, R: np.ndarray, innov: np.ndarray) -> None:
        """Compute Kalman gain, update covariance, inject error into nominal."""
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ innov  # 15-vector

        # Joseph-form covariance update for numerical stability
        I_KH = np.eye(self.N_ERR) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        # Inject into nominal
        self.state.p += dx[self.IDX_P]
        self.state.v += dx[self.IDX_V]
        dtheta = dx[self.IDX_TH]
        self.state.q = self.state.q * Rotation.from_rotvec(dtheta)
        self.state.b_a += dx[self.IDX_BA]
        self.state.b_g += dx[self.IDX_BG]


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle (rad) to the half-open interval [-π, π)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


# ---------------------- high-level driver routine --------------------------


@dataclass
class FusionResult:
    """Output of run_fusion."""

    t: np.ndarray            # (N,) time in seconds since first IMU sample
    p_ned: np.ndarray        # (N, 3) position in local NED frame (m)
    v_ned: np.ndarray        # (N, 3) velocity in NED frame (m/s)
    euler_zyx: np.ndarray    # (N, 3) Z-Y-X intrinsic Euler (yaw, pitch, roll) in rad
    b_a: np.ndarray          # (N, 3) accel bias estimate (m/s²)
    b_g: np.ndarray          # (N, 3) gyro bias estimate (rad/s)
    sigma_p: np.ndarray      # (N, 3) position 1-sigma uncertainty (m)
    ref_lat_deg: float       # reference for NED origin
    ref_lon_deg: float
    ref_alt_m: float
    n_gps_updates: int
    n_mag_updates: int
    n_static_anchors: int = 0

    @property
    def altitude_m(self) -> np.ndarray:
        """Convenience: altitude above NED origin (m, +up). Equals -p_d."""
        return -self.p_ned[:, 2] + self.ref_alt_m


def run_fusion(
    data: dict,
    noise: EKFNoiseParams | None = None,
    stationary_init_seconds: float = 2.0,
    gnss_min_fix_type: int = 3,
    apply_mag_updates: bool = True,
    max_substep_dt: float = 0.05,
    gap_skip_dt: float = 1.0,
    use_outlier_flags: bool = True,
    gps_horizontal_only: bool = False,
    static_anchor_sigma_horiz_m: float | None = None,
    static_anchor_sigma_vert_m: float | None = None,
    static_anchor_cadence_hz: float = 1.0,
) -> FusionResult:
    """Run the EKF end-to-end on a `load_data_as_arrays` dict.

    Required arrays in `data`:
        imu_micros_unwrapped, imu_acc_x/y/z (mg), imu_gyr_x/y/z (mdps)

    Optional (folded in when present + non-trivial):
        imu_mag_x/y/z (uT), gnss_micros_unwrapped, gnss_latitude/longitude,
        gnss_altitude_msl, gnss_vel_north/east/down (mm/s), gnss_fix_type

    `stationary_init_seconds` : how many seconds at the start to average for
        initial attitude. Set higher if the device wobbles at boot. If no
        magnetometer is present, yaw is initialised to zero (filter will then
        drift in yaw until GPS velocity provides an observation).

    `gnss_min_fix_type` : minimum u-blox fix type to accept (3 = 3D fix).

    `apply_mag_updates` : set False to disable mag yaw updates (e.g. if you
        suspect strong magnetic interference).

    `max_substep_dt` : whenever the IMU dt exceeds this (typically due to SD
        flush gaps), the predict step is broken into substeps no larger than
        this duration. Keeps quaternion integration and covariance propagation
        well-conditioned across gaps. Set to a large value (e.g. 1.0) to
        disable substepping.

    `gap_skip_dt` : if the IMU dt exceeds this, the predict step is skipped
        entirely for that sample — the device has been "dark" too long to
        trust open-loop integration. A subsequent GNSS update will re-anchor
        position/velocity. Set to math.inf to never skip.

    `use_outlier_flags` : when True (default), IMU samples flagged as outliers
        by the decoder (imu_acc_*_outlier / imu_gyr_*_outlier) are replaced
        with the last good reading before being fed to the predict step. This
        is the cheapest, most effective defence against gap-edge spikes.

    `gps_horizontal_only` : when True, only the horizontal (N, E) components
        of GPS position and velocity measurements are folded into the filter.
        Useful for indoor / weak-signal recordings where GPS altitude /
        vertical velocity are dominated by multipath noise. Vertical position
        is then driven by IMU integration only — apply `bandpass_result()`
        with a band centred on your motion-of-interest to extract the useful
        AC component.

    `static_anchor_sigma_horiz_m`, `static_anchor_sigma_vert_m`,
    `static_anchor_cadence_hz` : enable a synthetic "you're near the origin"
        pseudo-measurement, for GNSS-denied deployments. Pass any non-None
        sigma to switch on. The EKF treats it identically to a noisy GPS fix
        at (0, 0, 0) injected at `static_anchor_cadence_hz` Hz (default 1 Hz).
        Choose σ larger than the expected motion radius so the filter still
        tracks real motion — for human-scale / hand-test work σ=1 m is a
        sensible default; for wave buoys σ=5–10 m horizontal, 2–3 m vertical.
        When `static_anchor_sigma_vert_m` is None, the horizontal sigma is
        used vertically too. Tuning σ trades off anchor strength against
        responsiveness; cadence trades off bias-observability against
        between-update drift.
    """
    noise = noise or EKFNoiseParams()
    ekf = ErrorStateEKF(noise=noise)

    # --- 1. Set up time base
    imu_us = np.asarray(data["imu_micros_unwrapped"], dtype=np.float64)
    if len(imu_us) < 100:
        raise ValueError(f"Not enough IMU samples to run fusion ({len(imu_us)})")
    t_imu = (imu_us - imu_us[0]) * 1e-6  # seconds since first IMU sample

    # --- 2. Pull arrays into SI units
    # mg → m/s², mdps → rad/s, μT → μT (mag used direction-only for yaw)
    mg_to_ms2 = G_MAG / 1000.0
    mdps_to_radps = (np.pi / 180.0) / 1000.0
    acc_xyz = np.column_stack([
        data["imu_acc_x"] * mg_to_ms2,
        data["imu_acc_y"] * mg_to_ms2,
        data["imu_acc_z"] * mg_to_ms2,
    ])
    gyr_xyz = np.column_stack([
        data["imu_gyr_x"] * mdps_to_radps,
        data["imu_gyr_y"] * mdps_to_radps,
        data["imu_gyr_z"] * mdps_to_radps,
    ])
    has_mag = (
        "imu_mag_x" in data
        and len(data["imu_mag_x"]) == len(t_imu)
        and not np.all(data["imu_mag_x"] == 0)
    )
    if has_mag:
        # The firmware logs mag in the MAG-PCB-silkscreen frame, which on the
        # OLA has its Y and Z arrows opposite to the ACCEL/GYRO silkscreen
        # cross (verified BOOT_000362 orientation test). Flip Y and Z here so
        # the mag vector is expressed in the SAME body frame the accel/gyro
        # already live in — that's what the EKF's R_bn @ mag_body math assumes.
        mag_xyz = np.column_stack([
             np.asarray(data["imu_mag_x"]),
            -np.asarray(data["imu_mag_y"]),
            -np.asarray(data["imu_mag_z"]),
        ])
        logger.info("Magnetometer y/z flipped into accel/gyro body frame "
                    "(mag silkscreen vs accel/gyro silkscreen)")
    else:
        mag_xyz = None
        logger.warning("No magnetometer data — initial yaw set to 0 and no mag updates")

    # --- 3. Determine GNSS availability and pick NED origin
    gnss_t = None
    gnss_p_ned = None
    gnss_v_ned = None
    gnss_fix = None
    ref_lat = ref_lon = ref_alt = 0.0
    if (
        "gnss_micros_unwrapped" in data
        and len(data["gnss_micros_unwrapped"]) > 0
        and "gnss_altitude_msl" in data
        and not np.all(data["gnss_altitude_msl"] == 0)
    ):
        g_us = np.asarray(data["gnss_micros_unwrapped"], dtype=np.float64)
        gnss_t = (g_us - imu_us[0]) * 1e-6
        fix_arr = np.asarray(data["gnss_fix_type"])
        good = fix_arr >= gnss_min_fix_type
        if good.any():
            i_first = int(np.argmax(good))
            ref_lat = float(data["gnss_latitude"][i_first])
            ref_lon = float(data["gnss_longitude"][i_first])
            ref_alt = float(data["gnss_altitude_msl"][i_first])
            gnss_p_ned = lla_to_local_ned(
                data["gnss_latitude"], data["gnss_longitude"], data["gnss_altitude_msl"],
                ref_lat, ref_lon, ref_alt,
            )
            gnss_v_ned = np.column_stack([
                data["gnss_vel_north"] * 1e-3,
                data["gnss_vel_east"] * 1e-3,
                data["gnss_vel_down"] * 1e-3,
            ])
            gnss_fix = fix_arr
            logger.info(
                f"GNSS reference origin: lat={ref_lat:.6f}°, lon={ref_lon:.6f}°, "
                f"alt={ref_alt:.1f} m MSL (first 3D fix at t={gnss_t[i_first]:.1f}s)"
            )
        else:
            logger.warning(
                f"No GNSS fix ≥{gnss_min_fix_type}D in recording — running IMU-only "
                "(position/velocity will drift)"
            )
    else:
        logger.warning(
            "No GNSS altitude in recording — running IMU-only (this is normal "
            "for indoor recordings or pre-altitude firmware versions)"
        )

    # --- 4. Initial attitude from stationary average
    n_init = max(1, int(stationary_init_seconds * (len(t_imu) / max(t_imu[-1], 1e-6))))
    n_init = min(n_init, len(t_imu))
    a_mean = acc_xyz[:n_init].mean(axis=0)
    if has_mag:
        m_mean = mag_xyz[:n_init].mean(axis=0)
        q0 = attitude_from_accel_mag(a_mean, m_mean)
    else:
        # Roll/pitch from accel only; yaw = 0
        down_b = -a_mean / np.linalg.norm(a_mean)
        # Pick an arbitrary "north" orthogonal to down. Yaw is unobservable
        # here — will be corrected by GPS velocity if/when device moves.
        if abs(down_b[0]) < 0.9:
            north_b = np.cross(down_b, np.array([1.0, 0.0, 0.0]))
        else:
            north_b = np.cross(down_b, np.array([0.0, 1.0, 0.0]))
        north_b /= np.linalg.norm(north_b)
        east_b = np.cross(down_b, north_b)
        r_nav_in_body = np.column_stack([north_b, east_b, down_b])
        q0 = Rotation.from_matrix(r_nav_in_body.T)

    # Initial covariance: known position (0), reasonable velocity, looser attitude/bias
    P0 = np.diag([
        1.0, 1.0, 2.0,             # position (we *are* at origin, so small)
        0.5, 0.5, 1.0,             # velocity
        np.deg2rad(10) ** 2,
        np.deg2rad(10) ** 2,
        np.deg2rad(30) ** 2,       # yaw uncertainty larger when no mag
        0.5 ** 2, 0.5 ** 2, 0.5 ** 2,   # accel bias
        np.deg2rad(2) ** 2, np.deg2rad(2) ** 2, np.deg2rad(2) ** 2,  # gyro bias
    ])
    if not has_mag:
        P0[8, 8] = np.deg2rad(90) ** 2  # yaw completely unknown

    state0 = EKFState(p=np.zeros(3), v=np.zeros(3), q=q0)
    ekf.initialize(state0, P0)

    # --- 5. Main loop: predict at IMU rate, update on GNSS / mag arrival
    n = len(t_imu)
    p_log = np.zeros((n, 3))
    v_log = np.zeros((n, 3))
    euler_log = np.zeros((n, 3))
    ba_log = np.zeros((n, 3))
    bg_log = np.zeros((n, 3))
    sigma_p_log = np.zeros((n, 3))

    g_idx = 0
    n_gps_upd = 0
    n_mag_upd = 0
    n_substeps_used = 0
    n_skipped_gaps = 0
    n_held_outliers = 0
    # Mag updates: fold in at fixed rate (every ~10 IMU samples ≈ 10 Hz at 100Hz IMU)
    mag_update_stride = max(1, int(round(len(t_imu) / max(t_imu[-1], 1.0) / 10)))

    # Build a per-sample outlier mask — any axis flagged disqualifies the sample.
    # Decoder may or may not have populated these depending on segment length.
    if use_outlier_flags:
        acc_mask_keys = ("imu_acc_x_outlier", "imu_acc_y_outlier", "imu_acc_z_outlier")
        gyr_mask_keys = ("imu_gyr_x_outlier", "imu_gyr_y_outlier", "imu_gyr_z_outlier")
        try:
            outlier_mask = np.zeros(n, dtype=bool)
            for k in acc_mask_keys + gyr_mask_keys:
                if k in data and len(data[k]) == n:
                    outlier_mask |= np.asarray(data[k], dtype=bool)
        except Exception:
            outlier_mask = np.zeros(n, dtype=bool)
    else:
        outlier_mask = np.zeros(n, dtype=bool)

    # Working copies of the IMU streams with outliers replaced by last-good.
    # This is the spike-mitigation step: when an outlier sample lines up with
    # a gap-edge dt, the EKF would otherwise integrate a huge (a, dt) pair.
    acc_clean = acc_xyz.copy()
    gyr_clean = gyr_xyz.copy()
    last_good_acc = acc_xyz[0].copy()
    last_good_gyr = gyr_xyz[0].copy()
    for i in range(n):
        if outlier_mask[i]:
            acc_clean[i] = last_good_acc
            gyr_clean[i] = last_good_gyr
            n_held_outliers += 1
        else:
            last_good_acc = acc_xyz[i]
            last_good_gyr = gyr_xyz[i]

    # Static-anchor scheduler. We don't track the integer count of injections
    # because IMU dt can drift; instead use wall-clock-style timestamps and
    # inject whenever t_imu has crossed the next scheduled anchor time.
    static_anchor_enabled = static_anchor_sigma_horiz_m is not None
    static_anchor_period_s = 1.0 / max(static_anchor_cadence_hz, 1e-6)
    next_anchor_t = 0.0  # inject the first one immediately at sample 0
    n_static_anchors = 0

    for i in range(n):
        if i > 0:
            dt = t_imu[i] - t_imu[i - 1]
            if dt <= 0 or dt > gap_skip_dt:
                # Bad timestamp or gap too large — skip predict; covariance
                # stays as-is and a subsequent GPS update will re-anchor.
                n_skipped_gaps += 1
            elif dt > max_substep_dt:
                # Substep through the gap to keep quaternion + covariance
                # integration well-conditioned. acc/gyr held constant across
                # the gap (best we can do without IMU samples during the gap).
                n_sub = int(np.ceil(dt / max_substep_dt))
                sub_dt = dt / n_sub
                for _ in range(n_sub):
                    ekf.predict(acc_clean[i], gyr_clean[i], sub_dt)
                n_substeps_used += n_sub
            else:
                ekf.predict(acc_clean[i], gyr_clean[i], dt)

        # GPS updates: fold in any GNSS samples whose time has passed
        # Skip entirely if no usable fix was ever obtained (gnss_fix=None means
        # GNSS samples exist but none reached `gnss_min_fix_type`).
        if gnss_t is not None and gnss_fix is not None:
            while g_idx < len(gnss_t) and gnss_t[g_idx] <= t_imu[i]:
                if gnss_fix[g_idx] >= gnss_min_fix_type:
                    ekf.update_gps_position(gnss_p_ned[g_idx], horizontal_only=gps_horizontal_only)
                    ekf.update_gps_velocity(gnss_v_ned[g_idx], horizontal_only=gps_horizontal_only)
                    n_gps_upd += 1
                g_idx += 1

        # Static-anchor pseudo-measurement (GNSS-denied substitute). Drives the
        # accel-bias state observable in the absence of a real position source.
        while static_anchor_enabled and t_imu[i] >= next_anchor_t:
            ekf.update_static_anchor(
                sigma_horiz_m=static_anchor_sigma_horiz_m,
                sigma_vert_m=static_anchor_sigma_vert_m,
            )
            n_static_anchors += 1
            next_anchor_t += static_anchor_period_s

        # Mag update at reduced rate
        if has_mag and apply_mag_updates and (i % mag_update_stride == 0):
            ekf.update_mag_yaw(mag_xyz[i])
            n_mag_upd += 1

        # Log
        p_log[i] = ekf.state.p
        v_log[i] = ekf.state.v
        euler_log[i] = ekf.state.q.as_euler("ZYX")
        ba_log[i] = ekf.state.b_a
        bg_log[i] = ekf.state.b_g
        sigma_p_log[i] = np.sqrt(np.diag(ekf.P)[ekf.IDX_P])

    logger.success(
        f"EKF complete: {n} steps, {n_gps_upd} GPS updates, {n_mag_upd} mag "
        f"updates, {n_static_anchors} static anchors"
    )
    if n_held_outliers or n_substeps_used or n_skipped_gaps:
        logger.info(
            f"Gap/spike mitigation: held {n_held_outliers} outlier IMU samples, "
            f"substepped {n_substeps_used} sub-steps across large dts, "
            f"skipped {n_skipped_gaps} too-large gaps (> {gap_skip_dt}s)"
        )

    return FusionResult(
        t=t_imu,
        p_ned=p_log,
        v_ned=v_log,
        euler_zyx=euler_log,
        b_a=ba_log,
        b_g=bg_log,
        sigma_p=sigma_p_log,
        ref_lat_deg=ref_lat,
        ref_lon_deg=ref_lon,
        ref_alt_m=ref_alt,
        n_gps_updates=n_gps_upd,
        n_mag_updates=n_mag_upd,
        n_static_anchors=n_static_anchors,
    )


# ----------------------- post-processing helpers ---------------------------


def _bandpass_zero_phase(
    x: np.ndarray,
    t: np.ndarray,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass along axis 0 of x.

    Assumes (and lightly verifies) that t is approximately uniformly sampled.
    Uses forward-backward filtering (sosfiltfilt) so there is no phase delay
    in the output — useful when you want to overlay the filtered signal on
    the raw signal without time-shifting.
    """
    from scipy.signal import butter, sosfiltfilt

    dt = np.diff(t)
    dt_med = float(np.median(dt))
    if dt_med <= 0:
        raise ValueError("Cannot bandpass: median dt is non-positive")
    fs = 1.0 / dt_med
    nyq = 0.5 * fs
    if not (0 < low_hz < high_hz < nyq):
        raise ValueError(
            f"Band [{low_hz}, {high_hz}] Hz out of range for fs={fs:.1f} Hz "
            f"(Nyquist {nyq:.1f}). Need 0 < low < high < Nyquist."
        )
    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    if x.ndim == 1:
        return sosfiltfilt(sos, x)
    return np.column_stack([sosfiltfilt(sos, x[:, k]) for k in range(x.shape[1])])


def bandpass_result(
    result: FusionResult,
    low_hz: float = 0.05,
    high_hz: float = 2.5,
    order: int = 4,
) -> FusionResult:
    """Return a copy of `result` with position and velocity band-pass filtered.

    Default band [0.05, 2.5] Hz matches typical wave / motion-of-interest
    frequencies. The output has *no* DC component — it represents oscillation
    around the slow mean, which is what you usually want when waves or
    vibrations are the signal of interest.

    Bias and attitude streams are passed through unchanged.

    `ref_alt_m` is forced to 0 in the output so that `altitude_m` simply
    returns the band-pass-filtered vertical displacement (in metres) about
    zero, rather than around the absolute MSL reference.
    """
    p_bp = _bandpass_zero_phase(result.p_ned, result.t, low_hz, high_hz, order)
    v_bp = _bandpass_zero_phase(result.v_ned, result.t, low_hz, high_hz, order)

    from dataclasses import replace

    return replace(
        result,
        p_ned=p_bp,
        v_ned=v_bp,
        # No meaningful absolute reference after band-pass
        ref_alt_m=0.0,
        # sigma_p is no longer a 1σ band of the filtered signal — set to NaN
        # so plots that try to use it won't silently mislead. Users who want
        # filtered uncertainty should run the EKF in a frequency-aware mode.
        sigma_p=np.full_like(result.sigma_p, np.nan),
    )
