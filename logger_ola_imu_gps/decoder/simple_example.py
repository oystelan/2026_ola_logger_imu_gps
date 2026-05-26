#!/usr/bin/env python3
"""Simple example showing the recommended workflow for end users.

This script demonstrates the easiest way to decode OLA logger data files:
1. Decode the binary file
2. Load the decoded data as numpy arrays
3. Plot or analyze the data

Creates 4 plots showing:
- 3-axis acceleration with outlier detection
- 3-axis gyroscope with outlier detection
- GPS track (lat/lon) with position outliers
- GNSS velocities (NED) with velocity outliers
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from decoder import decode_file, load_data_as_arrays


def main():
    """Decode and plot OLA logger data."""
    
    # Step 1: Specify the data file to decode
    data_file = Path("DATA_BOOT_0000_TIME_20260204T193000.dat")
    
    if not data_file.exists():
        print(f"Error: {data_file} not found")
        print("Please provide a valid .dat file")
        return
    
    print(f"Decoding: {data_file}")
    print("=" * 60)
    
    # Step 2: Decode the file (creates .npz file)
    result = decode_file(data_file)
    npz_file = result['file']
    print(f"\nDecoded data saved to: {npz_file}")
    
    # Step 3: Load decoded data as numpy arrays
    data = load_data_as_arrays(npz_file)
    
    # Step 4: Display summary information
    print("\n" + "=" * 60)
    print("DECODED DATA SUMMARY")
    print("=" * 60)
    print(f"Firmware: {data['firmware_commit']}")
    print(f"IMU ODR: {data['imu_odr']} Hz")
    print(f"GNSS Rate: {data['gnss_rate']} Hz")
    print(f"Segments: {data['number_of_segments']}")
    print(f"\nData points:")
    print(f"  - PPS:  {len(data['pps_micros'])} entries")
    print(f"  - GNSS: {len(data['gnss_micros'])} entries")
    print(f"  - IMU:  {len(data['imu_micros'])} entries")
    
    # Step 5: Simple data analysis
    if len(data['imu_micros']) > 0:
        # Filter out NaN values for duration calculation
        valid_times = data['imu_utc'][~np.isnan(data['imu_utc'])]
        if len(valid_times) > 1:
            duration_s = (valid_times[-1] - valid_times[0])
            print(f"\nRecording duration: {duration_s:.1f} seconds ({duration_s/60:.1f} minutes)")
        
        # Count outliers
        n_acc_outliers = sum(data['imu_acc_x_outlier']) + sum(data['imu_acc_y_outlier']) + sum(data['imu_acc_z_outlier'])
        n_gyr_outliers = sum(data['imu_gyr_x_outlier']) + sum(data['imu_gyr_y_outlier']) + sum(data['imu_gyr_z_outlier'])
        print(f"IMU outliers: {n_acc_outliers} acceleration, {n_gyr_outliers} gyroscope")
    
    if len(data['gnss_micros']) > 0:
        n_pos_outliers = sum(data['gnss_latitude_outlier']) + sum(data['gnss_longitude_outlier'])
        n_vel_outliers = sum(data['gnss_vel_north_outlier']) + sum(data['gnss_vel_east_outlier']) + sum(data['gnss_vel_down_outlier'])
        print(f"GNSS outliers: {n_pos_outliers} position, {n_vel_outliers} velocity")
    
    # Step 6: Create plots
    print("\n" + "=" * 60)
    print("CREATING PLOTS")
    print("=" * 60)
    
    # Plot 1: IMU Acceleration
    if len(data['imu_micros']) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data['imu_utc'], data['imu_acc_x'], 'b-', linewidth=0.5, label='X', alpha=0.7)
        ax.plot(data['imu_utc'], data['imu_acc_y'], 'g-', linewidth=0.5, label='Y', alpha=0.7)
        ax.plot(data['imu_utc'], data['imu_acc_z'], 'r-', linewidth=0.5, label='Z', alpha=0.7)
        
        # Mark outliers for all axes
        outlier_mask = data['imu_acc_x_outlier'] | data['imu_acc_y_outlier'] | data['imu_acc_z_outlier']
        if sum(outlier_mask) > 0:
            # Plot outliers for each axis
            if sum(data['imu_acc_x_outlier']) > 0:
                ax.plot(data['imu_utc'][data['imu_acc_x_outlier']], 
                       data['imu_acc_x'][data['imu_acc_x_outlier']], 
                       'kx', markersize=6, alpha=0.8)
            if sum(data['imu_acc_y_outlier']) > 0:
                ax.plot(data['imu_utc'][data['imu_acc_y_outlier']], 
                       data['imu_acc_y'][data['imu_acc_y_outlier']], 
                       'kx', markersize=6, alpha=0.8)
            if sum(data['imu_acc_z_outlier']) > 0:
                ax.plot(data['imu_utc'][data['imu_acc_z_outlier']], 
                       data['imu_acc_z'][data['imu_acc_z_outlier']], 
                       'kx', markersize=6, alpha=0.8, label='Outliers')
        
        ax.set_xlabel('UTC Time (seconds since epoch)')
        ax.set_ylabel('Acceleration (mg)')
        ax.set_title(f'3-Axis Acceleration - {data_file.name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        print("  ✓ Created acceleration plot")
    
    # Plot 2: IMU Gyroscope
    if len(data['imu_micros']) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data['imu_utc'], data['imu_gyr_x'], 'b-', linewidth=0.5, label='X', alpha=0.7)
        ax.plot(data['imu_utc'], data['imu_gyr_y'], 'g-', linewidth=0.5, label='Y', alpha=0.7)
        ax.plot(data['imu_utc'], data['imu_gyr_z'], 'r-', linewidth=0.5, label='Z', alpha=0.7)
        
        # Mark outliers for all axes
        outlier_mask = data['imu_gyr_x_outlier'] | data['imu_gyr_y_outlier'] | data['imu_gyr_z_outlier']
        if sum(outlier_mask) > 0:
            # Plot outliers for each axis
            if sum(data['imu_gyr_x_outlier']) > 0:
                ax.plot(data['imu_utc'][data['imu_gyr_x_outlier']], 
                       data['imu_gyr_x'][data['imu_gyr_x_outlier']], 
                       'kx', markersize=6, alpha=0.8)
            if sum(data['imu_gyr_y_outlier']) > 0:
                ax.plot(data['imu_utc'][data['imu_gyr_y_outlier']], 
                       data['imu_gyr_y'][data['imu_gyr_y_outlier']], 
                       'kx', markersize=6, alpha=0.8)
            if sum(data['imu_gyr_z_outlier']) > 0:
                ax.plot(data['imu_utc'][data['imu_gyr_z_outlier']], 
                       data['imu_gyr_z'][data['imu_gyr_z_outlier']], 
                       'kx', markersize=6, alpha=0.8, label='Outliers')
        
        ax.set_xlabel('UTC Time (seconds since epoch)')
        ax.set_ylabel('Angular Velocity (mdps)')
        ax.set_title(f'3-Axis Gyroscope - {data_file.name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        print("  ✓ Created gyroscope plot")
    
    # Plot 3: GPS Track
    if len(data['gnss_micros']) > 0 and len(data['gnss_latitude']) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot trajectory
        ax.plot(data['gnss_longitude'], data['gnss_latitude'], 'b-', linewidth=1, alpha=0.6)
        
        # Mark start and end
        ax.plot(data['gnss_longitude'][0], data['gnss_latitude'][0], 
               'go', markersize=10, label='Start')
        ax.plot(data['gnss_longitude'][-1], data['gnss_latitude'][-1], 
               'ro', markersize=10, label='End')
        
        # Mark outliers
        outlier_mask = data['gnss_latitude_outlier'] | data['gnss_longitude_outlier']
        if sum(outlier_mask) > 0:
            ax.plot(data['gnss_longitude'][outlier_mask], 
                   data['gnss_latitude'][outlier_mask],
                   'kx', markersize=8, label='Outliers')
        
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        ax.set_title(f'GPS Track - {data_file.name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        print("  ✓ Created GPS track plot")
    
    # Plot 4: GNSS Velocities (NED frame)
    if len(data['gnss_micros']) > 0 and len(data['gnss_vel_north']) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data['gnss_utc'], data['gnss_vel_north'], 'b-', linewidth=1, label='North', alpha=0.7)
        ax.plot(data['gnss_utc'], data['gnss_vel_east'], 'g-', linewidth=1, label='East', alpha=0.7)
        ax.plot(data['gnss_utc'], data['gnss_vel_down'], 'r-', linewidth=1, label='Down', alpha=0.7)
        
        # Mark outliers for all velocity components
        outlier_mask = (data['gnss_vel_north_outlier'] | 
                       data['gnss_vel_east_outlier'] | 
                       data['gnss_vel_down_outlier'])
        if sum(outlier_mask) > 0:
            # Plot outliers for each component
            if sum(data['gnss_vel_north_outlier']) > 0:
                ax.plot(data['gnss_utc'][data['gnss_vel_north_outlier']], 
                       data['gnss_vel_north'][data['gnss_vel_north_outlier']], 
                       'kx', markersize=6, alpha=0.8)
            if sum(data['gnss_vel_east_outlier']) > 0:
                ax.plot(data['gnss_utc'][data['gnss_vel_east_outlier']], 
                       data['gnss_vel_east'][data['gnss_vel_east_outlier']], 
                       'kx', markersize=6, alpha=0.8)
            if sum(data['gnss_vel_down_outlier']) > 0:
                ax.plot(data['gnss_utc'][data['gnss_vel_down_outlier']], 
                       data['gnss_vel_down'][data['gnss_vel_down_outlier']], 
                       'kx', markersize=6, alpha=0.8, label='Outliers')
        
        ax.set_xlabel('UTC Time (seconds since epoch)')
        ax.set_ylabel('Velocity (mm/s)')
        ax.set_title(f'GNSS Velocities (NED) - {data_file.name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        print("  ✓ Created GNSS velocity plot")
    
    # Show all plots
    print("\nDisplaying plots. Close windows to exit.")
    plt.show()
    
    print("\n" + "=" * 60)
    print("✓ Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
