#!/usr/bin/env python3
"""Simple example showing the recommended workflow for end users.

This script demonstrates the easiest way to decode OLA logger data files:
1. Decode the binary file
2. Load the decoded data as numpy arrays
3. Plot or analyze the data

Creates up to 5 plots showing:
- 3-axis acceleration with outlier detection
- 3-axis gyroscope with outlier detection
- 3-axis magnetometer (only if firmware logged magnetometer)
- GPS track (lat/lon) with position outliers
- GNSS velocities (NED) with velocity outliers

Usage:
    python simple_example.py [path/to/file.dat]

If no path is given, the script picks the first DATA_BOOT_*.dat file in the
current directory.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from decoder import decode_file, load_data_as_arrays


def find_default_data_file() -> Path | None:
    """Pick the first DATA_BOOT_*.dat file in the script directory."""
    here = Path(__file__).parent
    candidates = sorted(here.glob("DATA_BOOT_*.dat"))
    return candidates[0] if candidates else None


def main():
    """Decode and plot OLA logger data."""

    # Step 1: Specify the data file to decode (CLI arg, or auto-pick from current dir)
    if len(sys.argv) > 1:
        data_file = Path(sys.argv[1])
    else:
        data_file = find_default_data_file()
        if data_file is None:
            print("No data file given and no DATA_BOOT_*.dat found in script directory.")
            print("Usage: python simple_example.py [path/to/file.dat]")
            return
        print(f"No path argument given, auto-selected: {data_file.name}")

    if not data_file.exists():
        print(f"Error: {data_file} not found")
        return

    print(f"Decoding: {data_file}")
    print("=" * 60)

    # Step 2: Decode the file (creates .npz file).
    # allow_no_pps=True so indoor / debug recordings still decode. When the
    # PPS hardware wire is connected to OLA pad 11 and the GNSS module has a
    # fix, you can leave this off to get microsecond-accurate UTC timestamps.
    result = decode_file(data_file, allow_no_pps=True)
    npz_file = result['file']
    print(f"\nDecoded data saved to: {npz_file}")

    # Step 3: Load decoded data as numpy arrays
    data = load_data_as_arrays(npz_file)

    # Detect whether the magnetometer was logged for this file.
    has_mag = ('imu_mag_x' in data and len(data['imu_mag_x']) > 0
               and not np.all(data['imu_mag_x'] == 0))

    # Step 4: Display summary information
    print("\n" + "=" * 60)
    print("DECODED DATA SUMMARY")
    print("=" * 60)
    print(f"Firmware:   {data.get('firmware_commit', 'unknown')}")
    print(f"IMU ODR:    {data.get('imu_odr', 'unknown')} Hz")
    print(f"GNSS Rate:  {data.get('gnss_rate', 'unknown')} Hz")
    print(f"Acc sens:   {data.get('acc_sensitivity', '?')} mg/LSB")
    print(f"Gyr sens:   {data.get('gyr_sensitivity', '?')} mdps/LSB")
    if has_mag:
        print(f"Mag sens:   {data.get('mag_sensitivity', '?')} uT/LSB  (ICM-20948 AK09916)")
    else:
        print("Mag sens:   N/A  (firmware did not log magnetometer)")
    print(f"Segments:   {data.get('number_of_segments', '?')}")
    print("\nData points:")
    print(f"  - PPS:  {len(data['pps_micros'])} entries")
    print(f"  - GNSS: {len(data['gnss_micros'])} entries")
    print(f"  - IMU:  {len(data['imu_micros'])} entries")

    # Step 5: Simple data analysis
    if len(data['imu_micros']) > 0:
        # Filter out NaN values for duration calculation
        valid_times = data['imu_utc'][~np.isnan(data['imu_utc'])]
        if len(valid_times) > 1:
            duration_s = (valid_times[-1] - valid_times[0])
            print(f"\nRecording duration (UTC): {duration_s:.1f} seconds ({duration_s/60:.1f} minutes)")
        else:
            # Fall back to raw micros if no UTC sync (no PPS available).
            micros = data['imu_micros_unwrapped']
            valid_us = micros[~np.isnan(micros)] if micros.dtype.kind == 'f' else micros
            if len(valid_us) > 1:
                duration_s = (valid_us[-1] - valid_us[0]) / 1e6
                print(f"\nRecording duration (raw micros): {duration_s:.1f} seconds (~{duration_s/60:.1f} min, no UTC sync)")

        # Count outliers
        n_acc_outliers = sum(data['imu_acc_x_outlier']) + sum(data['imu_acc_y_outlier']) + sum(data['imu_acc_z_outlier'])
        n_gyr_outliers = sum(data['imu_gyr_x_outlier']) + sum(data['imu_gyr_y_outlier']) + sum(data['imu_gyr_z_outlier'])
        print(f"IMU outliers: {n_acc_outliers} acceleration, {n_gyr_outliers} gyroscope")
        if has_mag:
            n_mag_outliers = sum(data['imu_mag_x_outlier']) + sum(data['imu_mag_y_outlier']) + sum(data['imu_mag_z_outlier'])
            print(f"Mag outliers: {n_mag_outliers}")

    if len(data['gnss_micros']) > 0:
        n_pos_outliers = sum(data['gnss_latitude_outlier']) + sum(data['gnss_longitude_outlier'])
        n_vel_outliers = sum(data['gnss_vel_north_outlier']) + sum(data['gnss_vel_east_outlier']) + sum(data['gnss_vel_down_outlier'])
        print(f"GNSS outliers: {n_pos_outliers} position, {n_vel_outliers} velocity")

    # Step 6: Pick the x-axis source. Prefer UTC-synced timestamps; fall back to
    # raw unwrapped micros (in seconds) when PPS sync isn't available.
    if len(data['imu_micros']) > 0 and not np.all(np.isnan(data['imu_utc'])):
        imu_t = data['imu_utc']
        imu_xlabel = 'UTC time (s since epoch)'
    elif len(data['imu_micros']) > 0:
        imu_t = data['imu_micros_unwrapped'] / 1e6
        imu_xlabel = 'micros() unwrapped (s since boot)'
    else:
        imu_t = np.array([])
        imu_xlabel = ''

    if len(data['gnss_micros']) > 0 and not np.all(np.isnan(data['gnss_utc'])):
        gnss_t = data['gnss_utc']
        gnss_xlabel = 'UTC time (s since epoch)'
    elif len(data['gnss_micros']) > 0:
        gnss_t = data['gnss_micros_unwrapped'] / 1e6
        gnss_xlabel = 'micros() unwrapped (s since boot)'
    else:
        gnss_t = np.array([])
        gnss_xlabel = ''

    # Step 7: Create plots
    print("\n" + "=" * 60)
    print("CREATING PLOTS")
    print("=" * 60)

    # Plot 1: IMU Acceleration
    if len(imu_t) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(imu_t, data['imu_acc_x'], 'b-', linewidth=0.5, label='X', alpha=0.7)
        ax.plot(imu_t, data['imu_acc_y'], 'g-', linewidth=0.5, label='Y', alpha=0.7)
        ax.plot(imu_t, data['imu_acc_z'], 'r-', linewidth=0.5, label='Z', alpha=0.7)
        for axis in ('x', 'y', 'z'):
            mask = data[f'imu_acc_{axis}_outlier']
            if mask.any():
                ax.plot(imu_t[mask], data[f'imu_acc_{axis}'][mask], 'kx', markersize=6, alpha=0.8)
        ax.set_xlabel(imu_xlabel)
        ax.set_ylabel('Acceleration (mg)')
        ax.set_title(f'3-Axis Acceleration - {data_file.name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        print("  ok  acceleration plot")

    # Plot 2: IMU Gyroscope
    if len(imu_t) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(imu_t, data['imu_gyr_x'], 'b-', linewidth=0.5, label='X', alpha=0.7)
        ax.plot(imu_t, data['imu_gyr_y'], 'g-', linewidth=0.5, label='Y', alpha=0.7)
        ax.plot(imu_t, data['imu_gyr_z'], 'r-', linewidth=0.5, label='Z', alpha=0.7)
        for axis in ('x', 'y', 'z'):
            mask = data[f'imu_gyr_{axis}_outlier']
            if mask.any():
                ax.plot(imu_t[mask], data[f'imu_gyr_{axis}'][mask], 'kx', markersize=6, alpha=0.8)
        ax.set_xlabel(imu_xlabel)
        ax.set_ylabel('Angular velocity (mdps)')
        ax.set_title(f'3-Axis Gyroscope - {data_file.name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        print("  ok  gyroscope plot")

    # Plot 3: Magnetometer (only if logged)
    if has_mag and len(imu_t) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(imu_t, data['imu_mag_x'], 'b-', linewidth=0.5, label='X', alpha=0.7)
        ax.plot(imu_t, data['imu_mag_y'], 'g-', linewidth=0.5, label='Y', alpha=0.7)
        ax.plot(imu_t, data['imu_mag_z'], 'r-', linewidth=0.5, label='Z', alpha=0.7)
        for axis in ('x', 'y', 'z'):
            mask = data[f'imu_mag_{axis}_outlier']
            if mask.any():
                ax.plot(imu_t[mask], data[f'imu_mag_{axis}'][mask], 'kx', markersize=6, alpha=0.8)
        ax.set_xlabel(imu_xlabel)
        ax.set_ylabel('Magnetic field (uT)')
        ax.set_title(f'3-Axis Magnetometer (AK09916, ~100 Hz) - {data_file.name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        print("  ok  magnetometer plot")

    # Plot 4: GPS Track
    if len(data['gnss_micros']) > 0 and len(data['gnss_latitude']) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(data['gnss_longitude'], data['gnss_latitude'], 'b-', linewidth=1, alpha=0.6)
        ax.plot(data['gnss_longitude'][0], data['gnss_latitude'][0], 'go', markersize=10, label='Start')
        ax.plot(data['gnss_longitude'][-1], data['gnss_latitude'][-1], 'ro', markersize=10, label='End')
        outlier_mask = data['gnss_latitude_outlier'] | data['gnss_longitude_outlier']
        if outlier_mask.any():
            ax.plot(data['gnss_longitude'][outlier_mask], data['gnss_latitude'][outlier_mask],
                    'kx', markersize=8, label='Outliers')
        ax.set_xlabel('Longitude (deg)')
        ax.set_ylabel('Latitude (deg)')
        ax.set_title(f'GPS Track - {data_file.name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        print("  ok  GPS track plot")

    # Plot 5: GNSS Velocities (NED frame)
    if len(gnss_t) > 0 and len(data['gnss_vel_north']) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(gnss_t, data['gnss_vel_north'], 'b-', linewidth=1, label='North', alpha=0.7)
        ax.plot(gnss_t, data['gnss_vel_east'], 'g-', linewidth=1, label='East', alpha=0.7)
        ax.plot(gnss_t, data['gnss_vel_down'], 'r-', linewidth=1, label='Down', alpha=0.7)
        for direction in ('north', 'east', 'down'):
            mask = data[f'gnss_vel_{direction}_outlier']
            if mask.any():
                ax.plot(gnss_t[mask], data[f'gnss_vel_{direction}'][mask], 'kx', markersize=6, alpha=0.8)
        ax.set_xlabel(gnss_xlabel)
        ax.set_ylabel('Velocity (mm/s)')
        ax.set_title(f'GNSS Velocities (NED) - {data_file.name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        print("  ok  GNSS velocity plot")

    print("\nDisplaying plots. Close windows to exit.")
    plt.show()

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
