#!/usr/bin/env python3
"""Example demonstrating segment-based data processing.

This example shows how to:
1. Decode a data file into segments
2. Access individual segments
3. Load and combine all segments
"""

from pathlib import Path
import numpy as np
from decoder import decode_file, load_and_combine_segments


def main():
    """Demonstrate segment-based decoding."""
    # Find a test data file
    test_files = list(Path(".").glob("DATA_*.dat"))
    if not test_files:
        print("No data files found. Please run with a DATA_*.dat file in the current directory.")
        return
    
    input_file = test_files[0]
    print(f"Processing: {input_file}")
    
    # Decode the file - this creates segments automatically
    output_files = decode_file(input_file)
    npz_file = output_files["file"]
    print(f"\nDecoded to: {npz_file}")
    
    # Load the NPZ file and examine segments
    print("\n=== Examining Segments ===")
    data = np.load(npz_file, allow_pickle=True)
    n_segments = int(data['number_of_segments'][0])
    print(f"Number of segments: {n_segments}")
    
    # Show info for each segment
    for seg_idx in range(n_segments):
        seg_suffix = f"_segment{seg_idx:03d}"
        pps_seg = data[f"pps{seg_suffix}"]
        gnss_seg = data[f"gnss{seg_suffix}"]
        imu_seg = data[f"imu{seg_suffix}"]
        
        print(f"  Segment {seg_idx:03d}: {len(pps_seg):3d} PPS, "
              f"{len(gnss_seg):4d} GNSS, {len(imu_seg):5d} IMU")
    
    # Load and combine all segments
    print("\n=== Loading Combined Data ===")
    combined = load_and_combine_segments(npz_file)
    print(f"Total: {len(combined['pps'])} PPS, "
          f"{len(combined['gnss'])} GNSS, {len(combined['imu'])} IMU entries")
    
    # Access header info
    print("\n=== Header Info ===")
    if 'imu_odr' in combined:
        print(f"IMU ODR: {combined['imu_odr'][0]} Hz")
    if 'gnss_rate' in combined:
        print(f"GNSS Rate: {combined['gnss_rate'][0]} Hz")
    if 'acc_sensitivity' in combined:
        print(f"Acc Sensitivity: {combined['acc_sensitivity'][0]} mg/LSB")
    if 'gyr_sensitivity' in combined:
        print(f"Gyr Sensitivity: {combined['gyr_sensitivity'][0]} mdps/LSB")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
