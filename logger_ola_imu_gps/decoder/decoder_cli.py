#!/usr/bin/env python3
"""Command-line tool for decoding and visualizing OLA logger data files.

This CLI tool provides comprehensive visualization of decoded data with 8 plots.
It produces the same .npz output files as the programmatic API.

**Usage:**

    python decoder_cli.py -p DATA_BOOT_0000_TIME_20260204T193000.dat

**Output:**

1. Decodes the binary .dat file → creates .npz file  
2. Generates 8 matplotlib plots:
   - 3-axis acceleration (mg) with outlier detection
   - 3-axis gyroscope (mdps) with outlier detection
   - GNSS coordinates (lat/lon) with outlier detection
   - GNSS velocities (NED frame) with outlier detection
   - Time differences between consecutive entries
   - PPS mismatch analysis (regression quality)
   - IMU counter analysis (unwrapping and anomalies)
   - Raw vs cleaned micros timestamps

The .npz files created by the CLI are identical to those created by
decode_file() in the Python API, and can be loaded with load_data_as_arrays().

**Alternative: Programmatic Usage**

For custom analysis or integration into your own scripts, use the Python API:

    from decoder import decode_file, load_data_as_arrays
    result = decode_file(Path("data.dat"))
    data = load_data_as_arrays(result['file'])
    
See simple_example.py for a complete working example.
"""

from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from decoder import decode_file, load_and_combine_segments


def compute_plot_limits(data: np.ndarray, outliers: np.ndarray, margin: float = 0.1) -> tuple[float, float]:
    """Compute plot axis limits based on valid (non-outlier) data.
    
    Args:
        data: Full data array
        outliers: Boolean array indicating outliers (True = outlier)
        margin: Fractional margin to add around valid data range (default 0.1 = 10%)
        
    Returns:
        Tuple of (min_limit, max_limit) for axis range
    """
    # Get valid (non-outlier) data
    valid_data = data[~outliers]
    
    if len(valid_data) == 0:
        # Fallback to full data if no valid points
        valid_data = data
    
    if len(valid_data) == 0:
        # No data at all
        return (0, 1)
    
    # Compute min/max of valid data
    data_min = np.min(valid_data)
    data_max = np.max(valid_data)
    
    # Add margin
    data_range = data_max - data_min
    if data_range == 0:
        # All values are the same, add a small margin
        data_range = abs(data_min) * 0.1 if data_min != 0 else 1.0
    
    margin_size = data_range * margin
    return (data_min - margin_size, data_max + margin_size)


@click.command()
@click.option(
    "-p",
    "--path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the data file to decode",
)
@click.option(
    "--allow-no-pps",
    is_flag=True,
    default=False,
    help=(
        "Keep segments that have IMU/GNSS data but no PPS. Use this for indoor / "
        "debug recordings, or when the GNSS module's PPS pin isn't wired up. "
        "Timestamps will not be UTC-synced via PPS regression (only raw micros + "
        "GNSS posix_timestamp from the 10 Hz fixes are available)."
    ),
)
def main(path: Path, allow_no_pps: bool) -> None:
    """Decode OLA logger data file and create matplotlib visualizations.

    This CLI tool decodes the specified data file and generates several plots:
    - 3-axis acceleration (mg)
    - 3-axis gyroscope (mdps)
    - GNSS coordinates (latitude/longitude)
    - Time differences between consecutive IMU and GNSS entries
    - PPS mismatch (UTC regression vs closest second)
    - IMU counter vs entry number
    - Combined micros data (raw vs cleaned/unwrapped) with min/max stats
    """
    logger.info(f"Decoding file: {path}")

    # Decode the file (no ASCII plots)
    output_files = decode_file(path, show_plots=False, allow_no_pps=allow_no_pps)

    # Extract unwrap stats
    unwrap_stats = output_files.get("unwrap_stats", None)

    # Load the decoded data from compressed archive
    logger.info("Loading decoded data for visualization...")
    npz_file = output_files["file"]
    combined = load_and_combine_segments(npz_file)
    pps_data = combined["pps"]
    gnss_data = combined["gnss"]
    imu_data = combined["imu"]

    logger.info(f"Loaded {len(pps_data)} PPS, {len(gnss_data)} GNSS, {len(imu_data)} IMU entries")

    # Create visualizations
    logger.info("Creating matplotlib visualizations...")

    # Plot 1: 3-axis acceleration
    plot_imu_acceleration(imu_data)

    # Plot 2: 3-axis gyroscope
    plot_imu_gyroscope(imu_data)

    # Plot 3: GNSS coordinates
    plot_gnss_coordinates(gnss_data)
    
    # Plot 4: GNSS velocities (NED)
    plot_gnss_velocities(gnss_data)

    # Plot 5: Time differences between consecutive entries
    plot_time_differences(imu_data, gnss_data)

    # Plot 6: PPS mismatch
    plot_pps_mismatch(pps_data)

    # Plot 7: IMU counter vs entry number
    plot_imu_counter(imu_data, unwrap_stats)

    # Plot 8: Micros data - raw vs cleaned/unwrapped
    plot_micros_raw_vs_cleaned(pps_data, gnss_data, imu_data, unwrap_stats)

    logger.success("All plots created! Close plot windows to exit.")
    plt.show()


def plot_imu_acceleration(imu_data: np.ndarray) -> None:
    """Plot 3-axis acceleration data with outliers marked.

    Args:
        imu_data: Array of IMUReading objects
    """
    if len(imu_data) == 0:
        logger.warning("No IMU data to plot")
        return

    # Extract time and acceleration data
    time_utc = np.array([imu.utc_timestamp_from_pps_regression for imu in imu_data])
    time_relative = time_utc - time_utc[0]  # Relative time in seconds

    acc_x = np.array([imu.acc_x_mg for imu in imu_data])
    acc_y = np.array([imu.acc_y_mg for imu in imu_data])
    acc_z = np.array([imu.acc_z_mg for imu in imu_data])
    
    # Extract outlier flags
    acc_x_outliers = np.array([imu.acc_x_mg_stdchecked for imu in imu_data])
    acc_y_outliers = np.array([imu.acc_y_mg_stdchecked for imu in imu_data])
    acc_z_outliers = np.array([imu.acc_z_mg_stdchecked for imu in imu_data])

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("IMU Acceleration (3-axis)", fontsize=14, fontweight="bold")

    # Plot each axis
    axes[0].plot(time_relative, acc_x, "r-", linewidth=0.5, label="Acc X")
    if np.any(acc_x_outliers):
        axes[0].plot(time_relative[acc_x_outliers], acc_x[acc_x_outliers], 
                     "kx", markersize=8, markeredgewidth=2, label=f"Outliers ({np.sum(acc_x_outliers)})")
    axes[0].set_ylabel("Acc X (mg)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_ylim(compute_plot_limits(acc_x, acc_x_outliers))
    axes[0].autoscale(enable=False, axis='y')

    axes[1].plot(time_relative, acc_y, "g-", linewidth=0.5, label="Acc Y")
    if np.any(acc_y_outliers):
        axes[1].plot(time_relative[acc_y_outliers], acc_y[acc_y_outliers], 
                     "kx", markersize=8, markeredgewidth=2, label=f"Outliers ({np.sum(acc_y_outliers)})")
    axes[1].set_ylabel("Acc Y (mg)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_ylim(compute_plot_limits(acc_y, acc_y_outliers))
    axes[1].autoscale(enable=False, axis='y')

    axes[2].plot(time_relative, acc_z, "b-", linewidth=0.5, label="Acc Z")
    if np.any(acc_z_outliers):
        axes[2].plot(time_relative[acc_z_outliers], acc_z[acc_z_outliers], 
                     "kx", markersize=8, markeredgewidth=2, label=f"Outliers ({np.sum(acc_z_outliers)})")
    axes[2].set_ylabel("Acc Z (mg)")
    axes[2].set_xlabel("Time since start (seconds)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_ylim(compute_plot_limits(acc_z, acc_z_outliers))
    axes[2].autoscale(enable=False, axis='y')

    plt.tight_layout()
    logger.info("Created acceleration plot with outlier markers")


def plot_imu_gyroscope(imu_data: np.ndarray) -> None:
    """Plot 3-axis gyroscope data.

    Args:
        imu_data: Array of IMUReading objects
    """
    if len(imu_data) == 0:
        logger.warning("No IMU data to plot")
        return

    # Extract time and gyroscope data
    time_utc = np.array([imu.utc_timestamp_from_pps_regression for imu in imu_data])
    time_relative = time_utc - time_utc[0]  # Relative time in seconds

    gyr_x = np.array([imu.gyr_x_mdps for imu in imu_data])
    gyr_y = np.array([imu.gyr_y_mdps for imu in imu_data])
    gyr_z = np.array([imu.gyr_z_mdps for imu in imu_data])
    
    # Extract outlier flags
    gyr_x_outliers = np.array([imu.gyr_x_mdps_stdchecked for imu in imu_data])
    gyr_y_outliers = np.array([imu.gyr_y_mdps_stdchecked for imu in imu_data])
    gyr_z_outliers = np.array([imu.gyr_z_mdps_stdchecked for imu in imu_data])

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("IMU Gyroscope (3-axis)", fontsize=14, fontweight="bold")

    # Plot each axis
    axes[0].plot(time_relative, gyr_x, "r-", linewidth=0.5, label="Gyr X")
    if np.any(gyr_x_outliers):
        axes[0].plot(time_relative[gyr_x_outliers], gyr_x[gyr_x_outliers], 
                     "kx", markersize=8, markeredgewidth=2, label=f"Outliers ({np.sum(gyr_x_outliers)})")
    axes[0].set_ylabel("Gyr X (mdps)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_ylim(compute_plot_limits(gyr_x, gyr_x_outliers))
    axes[0].autoscale(enable=False, axis='y')

    axes[1].plot(time_relative, gyr_y, "g-", linewidth=0.5, label="Gyr Y")
    if np.any(gyr_y_outliers):
        axes[1].plot(time_relative[gyr_y_outliers], gyr_y[gyr_y_outliers], 
                     "kx", markersize=8, markeredgewidth=2, label=f"Outliers ({np.sum(gyr_y_outliers)})")
    axes[1].set_ylabel("Gyr Y (mdps)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_ylim(compute_plot_limits(gyr_y, gyr_y_outliers))
    axes[1].autoscale(enable=False, axis='y')

    axes[2].plot(time_relative, gyr_z, "b-", linewidth=0.5, label="Gyr Z")
    if np.any(gyr_z_outliers):
        axes[2].plot(time_relative[gyr_z_outliers], gyr_z[gyr_z_outliers], 
                     "kx", markersize=8, markeredgewidth=2, label=f"Outliers ({np.sum(gyr_z_outliers)})")
    axes[2].set_ylabel("Gyr Z (mdps)")
    axes[2].set_xlabel("Time since start (seconds)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_ylim(compute_plot_limits(gyr_z, gyr_z_outliers))
    axes[2].autoscale(enable=False, axis='y')

    plt.tight_layout()
    logger.info("Created gyroscope plot with outlier markers")


def plot_gnss_coordinates(gnss_data: np.ndarray) -> None:
    """Plot GNSS coordinates (latitude and longitude over time) with outliers marked.

    Args:
        gnss_data: Array of GNSSReading objects
    """
    if len(gnss_data) == 0:
        logger.warning("No GNSS data to plot")
        return

    # Extract time and position data
    time_utc = np.array([gnss.utc_timestamp_from_pps_regression for gnss in gnss_data])
    time_relative = time_utc - time_utc[0]  # Relative time in seconds

    latitude = np.array([gnss.latitude_dd for gnss in gnss_data])
    longitude = np.array([gnss.longitude_dd for gnss in gnss_data])
    
    # Extract outlier flags
    lat_outliers = np.array([gnss.latitude_dd_stdchecked for gnss in gnss_data])
    lon_outliers = np.array([gnss.longitude_dd_stdchecked for gnss in gnss_data])

    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("GNSS Coordinates", fontsize=14, fontweight="bold")

    # Plot latitude
    axes[0].plot(time_relative, latitude, "b-", linewidth=1, label="Latitude")
    if np.any(lat_outliers):
        axes[0].plot(time_relative[lat_outliers], latitude[lat_outliers], 
                     "kx", markersize=10, markeredgewidth=2, label=f"Outliers ({np.sum(lat_outliers)})")
    axes[0].set_ylabel("Latitude (degrees)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_ylim(compute_plot_limits(latitude, lat_outliers))
    axes[0].autoscale(enable=False, axis='y')

    # Plot longitude
    axes[1].plot(time_relative, longitude, "r-", linewidth=1, label="Longitude")
    if np.any(lon_outliers):
        axes[1].plot(time_relative[lon_outliers], longitude[lon_outliers], 
                     "kx", markersize=10, markeredgewidth=2, label=f"Outliers ({np.sum(lon_outliers)})")
    axes[1].set_ylabel("Longitude (degrees)")
    axes[1].set_xlabel("Time since start (seconds)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_ylim(compute_plot_limits(longitude, lon_outliers))
    axes[1].autoscale(enable=False, axis='y')

    plt.tight_layout()
    logger.info("Created GNSS coordinates plot with outlier markers")


def plot_gnss_velocities(gnss_data: np.ndarray) -> None:
    """Plot GNSS NED velocities over time with outliers marked.

    Args:
        gnss_data: Array of GNSSReading objects
    """
    if len(gnss_data) == 0:
        logger.warning("No GNSS data to plot")
        return

    # Extract time and velocity data
    time_utc = np.array([gnss.utc_timestamp_from_pps_regression for gnss in gnss_data])
    time_relative = time_utc - time_utc[0]  # Relative time in seconds

    vel_n = np.array([gnss.ned_vel_north_mmps for gnss in gnss_data])
    vel_e = np.array([gnss.ned_vel_east_mmps for gnss in gnss_data])
    vel_d = np.array([gnss.ned_vel_down_mmps for gnss in gnss_data])
    
    # Extract outlier flags
    vel_n_outliers = np.array([gnss.ned_vel_north_mmps_stdchecked for gnss in gnss_data])
    vel_e_outliers = np.array([gnss.ned_vel_east_mmps_stdchecked for gnss in gnss_data])
    vel_d_outliers = np.array([gnss.ned_vel_down_mmps_stdchecked for gnss in gnss_data])

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("GNSS NED Velocities", fontsize=14, fontweight="bold")

    # Plot North velocity
    axes[0].plot(time_relative, vel_n, "r-", linewidth=1, label="Vel North")
    if np.any(vel_n_outliers):
        axes[0].plot(time_relative[vel_n_outliers], vel_n[vel_n_outliers], 
                     "kx", markersize=10, markeredgewidth=2, label=f"Outliers ({np.sum(vel_n_outliers)})")
    axes[0].set_ylabel("Vel North (mm/s)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_ylim(compute_plot_limits(vel_n, vel_n_outliers))
    axes[0].autoscale(enable=False, axis='y')

    # Plot East velocity
    axes[1].plot(time_relative, vel_e, "g-", linewidth=1, label="Vel East")
    if np.any(vel_e_outliers):
        axes[1].plot(time_relative[vel_e_outliers], vel_e[vel_e_outliers], 
                     "kx", markersize=10, markeredgewidth=2, label=f"Outliers ({np.sum(vel_e_outliers)})")
    axes[1].set_ylabel("Vel East (mm/s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_ylim(compute_plot_limits(vel_e, vel_e_outliers))
    axes[1].autoscale(enable=False, axis='y')

    # Plot Down velocity
    axes[2].plot(time_relative, vel_d, "b-", linewidth=1, label="Vel Down")
    if np.any(vel_d_outliers):
        axes[2].plot(time_relative[vel_d_outliers], vel_d[vel_d_outliers], 
                     "kx", markersize=10, markeredgewidth=2, label=f"Outliers ({np.sum(vel_d_outliers)})")
    axes[2].set_ylabel("Vel Down (mm/s)")
    axes[2].set_xlabel("Time since start (seconds)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_ylim(compute_plot_limits(vel_d, vel_d_outliers))
    axes[2].autoscale(enable=False, axis='y')

    plt.tight_layout()
    logger.info("Created GNSS velocities plot with outlier markers")


def plot_time_differences(imu_data: np.ndarray, gnss_data: np.ndarray) -> None:
    """Plot time differences between consecutive entries.

    Computes and plots the time differences (dt) between consecutive
    measurements for both IMU and GNSS, using both micros timestamps
    and UTC timestamps from the PPS regression.

    Args:
        imu_data: Array of IMUReading objects
        gnss_data: Array of GNSSReading objects
    """
    if len(imu_data) < 2 and len(gnss_data) < 2:
        logger.warning("Not enough data to compute time differences")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Time Differences Between Consecutive Entries", fontsize=14, fontweight="bold")

    # IMU time differences from micros
    if len(imu_data) >= 2:
        micros_imu = np.array([imu.micros_reading for imu in imu_data])
        dt_micros_imu = np.diff(micros_imu) / 1000.0  # Convert to milliseconds

        axes[0, 0].plot(dt_micros_imu, "b.", markersize=1, alpha=0.5)
        axes[0, 0].set_ylabel("Δt (ms)")
        axes[0, 0].set_xlabel("Sample index")
        axes[0, 0].set_title("IMU: Time diff from micros()")
        axes[0, 0].grid(True, alpha=0.3)

        mean_dt = np.mean(dt_micros_imu)
        std_dt = np.std(dt_micros_imu)
        axes[0, 0].axhline(mean_dt, color="r", linestyle="--", linewidth=1,
                           label=f"Mean: {mean_dt:.3f} ms")
        axes[0, 0].legend()

        logger.info(f"IMU dt from micros: mean={mean_dt:.3f} ms, std={std_dt:.3f} ms")

    # IMU time differences from UTC regression
    if len(imu_data) >= 2:
        utc_imu = np.array([imu.utc_timestamp_from_pps_regression for imu in imu_data])
        dt_utc_imu = np.diff(utc_imu) * 1000  # Convert to milliseconds

        axes[0, 1].plot(dt_utc_imu, "b.", markersize=1, alpha=0.5)
        axes[0, 1].set_ylabel("Δt (ms)")
        axes[0, 1].set_xlabel("Sample index")
        axes[0, 1].set_title("IMU: Time diff from UTC regression")
        axes[0, 1].grid(True, alpha=0.3)

        mean_dt = np.mean(dt_utc_imu)
        std_dt = np.std(dt_utc_imu)
        axes[0, 1].axhline(mean_dt, color="r", linestyle="--", linewidth=1,
                           label=f"Mean: {mean_dt:.3f} ms")
        axes[0, 1].legend()

        logger.info(f"IMU dt from UTC: mean={mean_dt:.3f} ms, std={std_dt:.3f} ms")

    # GNSS time differences from micros
    if len(gnss_data) >= 2:
        micros_gnss = np.array([gnss.micros_reading for gnss in gnss_data])
        dt_micros_gnss = np.diff(micros_gnss) / 1000.0  # Convert to milliseconds

        axes[1, 0].plot(dt_micros_gnss, "g.", markersize=2, alpha=0.5)
        axes[1, 0].set_ylabel("Δt (ms)")
        axes[1, 0].set_xlabel("Sample index")
        axes[1, 0].set_title("GNSS: Time diff from micros()")
        axes[1, 0].grid(True, alpha=0.3)

        mean_dt = np.mean(dt_micros_gnss)
        std_dt = np.std(dt_micros_gnss)
        axes[1, 0].axhline(mean_dt, color="r", linestyle="--", linewidth=1,
                           label=f"Mean: {mean_dt:.1f} ms")
        axes[1, 0].legend()

        logger.info(f"GNSS dt from micros: mean={mean_dt:.1f} ms, std={std_dt:.1f} ms")

    # GNSS time differences from UTC regression
    if len(gnss_data) >= 2:
        utc_gnss = np.array([gnss.utc_timestamp_from_pps_regression for gnss in gnss_data])
        dt_utc_gnss = np.diff(utc_gnss) * 1000  # Convert to milliseconds

        axes[1, 1].plot(dt_utc_gnss, "g.", markersize=2, alpha=0.5)
        axes[1, 1].set_ylabel("Δt (ms)")
        axes[1, 1].set_xlabel("Sample index")
        axes[1, 1].set_title("GNSS: Time diff from UTC regression")
        axes[1, 1].grid(True, alpha=0.3)

        mean_dt = np.mean(dt_utc_gnss)
        std_dt = np.std(dt_utc_gnss)
        axes[1, 1].axhline(mean_dt, color="r", linestyle="--", linewidth=1,
                           label=f"Mean: {mean_dt:.1f} ms")
        axes[1, 1].legend()

        logger.info(f"GNSS dt from UTC: mean={mean_dt:.1f} ms, std={std_dt:.1f} ms")

    plt.tight_layout()
    logger.info("Created time differences plot")


def plot_imu_counter(imu_data: np.ndarray, unwrap_stats: dict | None = None) -> None:
    """Plot IMU counter as a function of entry number.

    Creates two subplots:
    - Left: Raw counter values
    - Right: Unwrapped counter with anomalies highlighted

    Args:
        imu_data: Array of IMUReading objects
        unwrap_stats: Optional unwrap statistics from decoder
    """
    if len(imu_data) == 0:
        logger.warning("No IMU data to plot")
        return

    # Extract counter values (raw and unwrapped)
    counters = np.array([imu.counter for imu in imu_data], dtype=np.int64)
    unwrapped_counters = np.array([imu.counter_unwrapped for imu in imu_data], dtype=np.int64)
    entry_numbers = np.arange(len(imu_data))

    # Get wrap and jump counts from unwrap_stats if available
    if unwrap_stats and "IMU" in unwrap_stats and "counter" in unwrap_stats["IMU"]:
        num_wraps = unwrap_stats["IMU"]["counter"]["wraps"]
        wrap_indices_array = unwrap_stats["IMU"]["counter"].get("wrap_indices")
        jump_indices_array = unwrap_stats["IMU"]["counter"].get("jump_indices")
        wrap_indices = list(wrap_indices_array) if wrap_indices_array is not None else []
        anomaly_indices = list(jump_indices_array) if jump_indices_array is not None else []
        num_anomalies = len(anomaly_indices)
    else:
        # Fallback: detect anomalies from unwrapped data
        num_wraps = 0
        wrap_indices = []
        anomaly_indices = []
        for i in range(1, len(unwrapped_counters)):
            increment = unwrapped_counters[i] - unwrapped_counters[i-1]
            if increment != 1:
                anomaly_indices.append(i)
                # Detect wraps: large jumps in unwrapped (but this is approximate)
                if increment > 60000:
                    num_wraps += 1
                    wrap_indices.append(i)
        num_anomalies = len(anomaly_indices)

    logger.info(f"IMU counter: {num_wraps} wraps detected, "
                f"{num_anomalies} anomalous samples (missed data)")

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("IMU Counter vs Entry Number", fontsize=14, fontweight="bold")

    # Left plot: raw counter
    axes[0].plot(entry_numbers, counters, "b-", linewidth=0.8, label="Raw Counter")
    axes[0].set_xlabel("Entry Number")
    axes[0].set_ylabel("Counter Value")
    axes[0].set_title(f"Raw Counter (uint16): {num_wraps} wraps, {num_anomalies} jumps (anomalous)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Right plot: unwrapped counter with wraps and anomalies
    axes[1].plot(entry_numbers, unwrapped_counters, "b-", linewidth=0.8,
                 label="Unwrapped Counter")

    # Highlight wraps with green crosses
    if wrap_indices:
        axes[1].plot(entry_numbers[wrap_indices],
                     unwrapped_counters[wrap_indices],
                     "gx", markersize=10, markeredgewidth=2,
                     label=f"Wraps ({num_wraps})")

    # Highlight anomalies with red crosses
    if anomaly_indices:
        axes[1].plot(entry_numbers[anomaly_indices],
                     unwrapped_counters[anomaly_indices],
                     "rx", markersize=8, markeredgewidth=2,
                     label=f"Anomalies ({num_anomalies})")

    axes[1].set_xlabel("Entry Number")
    axes[1].set_ylabel("Unwrapped Counter Value")
    axes[1].set_title(f"Unwrapped Counter: {num_wraps} wraps, {num_anomalies} jumps (anomalous)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    logger.info("Created IMU counter plot")


def plot_pps_mismatch(pps_data: np.ndarray) -> None:
    """Plot PPS mismatch between UTC regression and closest second.

    Computes and plots the mismatch between the UTC datetime from linear
    regression and the closest UTC second for each PPS entry.

    Args:
        pps_data: Array of PPSFix objects
    """
    if len(pps_data) == 0:
        logger.warning("No PPS data to plot")
        return

    # Check if regression was computed
    if pps_data[0].utc_timestamp_from_pps_regression is None:
        logger.warning("PPS regression not computed, skipping mismatch plot")
        return

    # Compute mismatch for each PPS entry
    mismatches = []
    time_points = []

    first_pps_utc = pps_data[0].utc_timestamp_from_pps_regression

    for pps in pps_data:
        utc_timestamp = pps.utc_timestamp_from_pps_regression
        closest_second = round(utc_timestamp)
        mismatch = utc_timestamp - closest_second
        mismatches.append(mismatch * 1000)  # Convert to milliseconds
        time_points.append(utc_timestamp - first_pps_utc)  # Time since start

    mismatches = np.array(mismatches)
    time_points = np.array(time_points)

    # Compute statistics
    max_mismatch = np.max(np.abs(mismatches))
    mean_mismatch = np.mean(mismatches)
    rms_mismatch = np.sqrt(np.mean(mismatches ** 2))

    logger.info(f"PPS Mismatch: max={max_mismatch:.3f} ms, "
                f"mean={mean_mismatch:.3f} ms, rms={rms_mismatch:.3f} ms")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("PPS UTC Mismatch: Regression vs Closest Second",
                 fontsize=14, fontweight="bold")

    ax.plot(time_points, mismatches, "b-", linewidth=0.8, label="UTC mismatch")
    ax.axhline(0, color="k", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Time since first PPS (seconds)")
    ax.set_ylabel("UTC mismatch (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add statistics text
    stats_text = (f"Max: {max_mismatch:.3f} ms\n"
                  f"Mean: {mean_mismatch:.3f} ms\n"
                  f"RMS: {rms_mismatch:.3f} ms")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    logger.info("Created PPS mismatch plot")


def plot_micros_raw_vs_cleaned(
    pps_data: np.ndarray,
    gnss_data: np.ndarray,
    imu_data: np.ndarray,
    unwrap_stats: dict | None = None
) -> None:
    """Plot raw micros vs PPS regression-based micros for PPS, GNSS, and IMU.

    Creates one figure with two subplots showing all three data types:
    - Left: Raw micros_reading vs normalized index for PPS, GNSS, IMU
    - Right: PPS regression-based micros (UTC timestamp * 1e6) vs normalized index

    Args:
        pps_data: Array of PPSFix objects
        gnss_data: Array of GNSSReading objects
        imu_data: Array of IMUReading objects
        unwrap_stats: Optional unwrap statistics from decoder (unused, kept for compatibility)
    """
    # Create single figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Micros Data: Raw vs PPS Regression-Based (PPS, GNSS, IMU)",
                 fontsize=14, fontweight="bold")

    # Plot data for each type
    data_types = [
        ("PPS", pps_data, "blue", "o"),
        ("GNSS", gnss_data, "green", "s"),
        ("IMU", imu_data, "red", "."),
    ]

    any_data_plotted = False

    for data_type, data, color, marker in data_types:
        if len(data) == 0:
            logger.warning(f"No {data_type} data to plot")
            continue

        any_data_plotted = True

        # Extract raw micros and PPS regression-based micros
        micros_raw = np.array([entry.micros_reading for entry in data], dtype=np.float64)
        
        # Extract UTC timestamps from PPS regression and convert to microseconds
        # utc_timestamp_from_pps_regression is in seconds, multiply by 1e6 for micros
        utc_timestamps = np.array([entry.utc_timestamp_from_pps_regression for entry in data], dtype=np.float64)
        
        # Filter out None/NaN values (entries without PPS regression data)
        valid_mask = ~np.isnan(utc_timestamps)
        utc_micros = utc_timestamps * 1e6  # Convert seconds to microseconds
        
        # Create normalized indices (0 to 1)
        n_entries = len(data)
        normalized_indices = np.arange(n_entries) / n_entries

        # Compute min/max for legend
        raw_min = np.min(micros_raw)
        raw_max = np.max(micros_raw)
        
        if np.any(valid_mask):
            utc_micros_valid = utc_micros[valid_mask]
            utc_min = np.min(utc_micros_valid)
            utc_max = np.max(utc_micros_valid)
            n_valid = np.sum(valid_mask)
        else:
            utc_min = 0
            utc_max = 0
            n_valid = 0

        # Left plot: Raw micros
        markersize = 4 if marker == "o" else (3 if marker == "s" else 0.5)
        axes[0].plot(normalized_indices, micros_raw, marker=marker, color=color,
                     linewidth=0, markersize=markersize, alpha=0.6,
                     label=f"{data_type} (n={n_entries:,}, min={raw_min:.0f}, max={raw_max:.0f})")

        # Right plot: PPS regression-based micros (only entries with valid UTC)
        if n_valid > 0:
            axes[1].plot(normalized_indices[valid_mask], utc_micros[valid_mask],
                         marker=marker, color=color, linewidth=0, markersize=markersize, alpha=0.6,
                         label=f"{data_type} (n={n_valid:,}, min={utc_min:.0f}, max={utc_max:.0f})")
        else:
            logger.warning(f"No valid PPS regression data for {data_type}")

    if not any_data_plotted:
        logger.warning("No data to plot for any data type")
        plt.close(fig)
        return

    # Configure left plot
    axes[0].set_xlabel("Entry Index / Total Entries")
    axes[0].set_ylabel("Raw Micros Reading (µs)")
    axes[0].set_title("Raw micros_reading")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=9)

    # Configure right plot
    axes[1].set_xlabel("Entry Index / Total Entries")
    axes[1].set_ylabel("PPS Regression-Based Micros (µs)")
    axes[1].set_title("UTC timestamp from PPS regression (× 1e6)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=9)

    plt.tight_layout()
    logger.info("Created combined micros raw vs PPS regression-based plot for PPS, GNSS, and IMU")


if __name__ == "__main__":
    main()
