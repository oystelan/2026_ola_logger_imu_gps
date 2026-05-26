"""Regression test for DATA_BOOT_000008_TIME_20260209T004502.dat.

This file previously failed with micros consistency check errors due to
improper handling of anomalous single-entry segments in the unwrapping logic.
The IMU timestamps were offset by 2^32 microseconds (~4295 seconds) from 
PPS/GNSS due to double-wrapping.

This test ensures the fix remains effective.
"""

from pathlib import Path

import numpy as np
import pytest

from decoder import decode_file


def test_boot8_file_decodes_successfully():
    """Test that DATA_BOOT_000008 decodes without micros consistency errors.
    
    This is a regression test for a bug where:
    - Segment 1 had only 1 IMU entry with an anomalous timestamp
    - This caused IMU to get double-wrapped (2*2^32 offset)
    - While PPS/GNSS got single-wrapped (1*2^32 offset)
    - Result: IMU was ~4295 seconds offset from PPS/GNSS
    
    The fix involves:
    1. Lowered wrap_threshold from 0.75 to 0.25
    2. Don't carry forward unwrapping state from segments with < 10 entries
    3. Reset both prev_raw_value AND offset together for consistency
    """
    dat_file = Path(__file__).parent / "DATA_BOOT_000008_TIME_20260209T004502.dat"
    
    if not dat_file.exists():
        pytest.skip(f"Test file not found: {dat_file}")
    
    # This should not raise any exceptions, particularly no micros consistency errors
    result = decode_file(dat_file, show_plots=False)
    
    # Verify we got the output file
    assert "file" in result
    npz_file = result["file"]
    assert npz_file.exists()
    
    # Load the npz file to check segments
    data = np.load(npz_file, allow_pickle=True)
    
    # Verify we have the expected number of segments
    assert "number_of_segments" in data
    num_segments = int(data["number_of_segments"][0])
    assert num_segments >= 17  # Should have 17 valid segments (segment 1 is skipped)
    
    # Check segment 2 (the problematic segment after the anomalous segment 1)
    # Data is stored as separate arrays: imu_segment002, pps_segment002, gnss_segment002
    assert "imu_segment002" in data
    assert "pps_segment002" in data
    assert "gnss_segment002" in data
    
    imu_seg2 = data["imu_segment002"]
    pps_seg2 = data["pps_segment002"]
    gnss_seg2 = data["gnss_segment002"]
    
    assert len(imu_seg2) > 0
    assert len(pps_seg2) > 0
    assert len(gnss_seg2) > 0
    
    # Verify IMU, PPS, and GNSS timestamps are aligned (within 10 seconds)
    # This is the key check - before the fix, IMU would be offset by ~4295 seconds
    imu_micros = np.array([entry.micros_reading for entry in imu_seg2])
    pps_micros = np.array([entry.micros_reading for entry in pps_seg2])
    gnss_micros = np.array([entry.micros_reading for entry in gnss_seg2])
    
    imu_min = imu_micros.min()
    pps_min = pps_micros.min()
    gnss_min = gnss_micros.min()
    
    # Convert to seconds for readability
    imu_min_s = imu_min / 1e6
    pps_min_s = pps_min / 1e6
    gnss_min_s = gnss_min / 1e6
    
    # Check alignment: all mins should be within 10 seconds of each other
    min_vals = [imu_min_s, pps_min_s, gnss_min_s]
    min_of_mins = min(min_vals)
    max_of_mins = max(min_vals)
    deviation = max_of_mins - min_of_mins
    
    assert deviation < 10.0, (
        f"Micros timestamps not aligned in segment 2! "
        f"IMU min: {imu_min_s:.3f}s, PPS min: {pps_min_s:.3f}s, "
        f"GNSS min: {gnss_min_s:.3f}s, deviation: {deviation:.3f}s"
    )
    
    # Also check segment 3 for good measure
    imu_seg3 = data["imu_segment003"]
    pps_seg3 = data["pps_segment003"]
    gnss_seg3 = data["gnss_segment003"]
    
    imu_micros_3 = np.array([entry.micros_reading for entry in imu_seg3])
    pps_micros_3 = np.array([entry.micros_reading for entry in pps_seg3])
    gnss_micros_3 = np.array([entry.micros_reading for entry in gnss_seg3])
    
    imu_min_3 = imu_micros_3.min() / 1e6
    pps_min_3 = pps_micros_3.min() / 1e6
    gnss_min_3 = gnss_micros_3.min() / 1e6
    
    min_vals_3 = [imu_min_3, pps_min_3, gnss_min_3]
    deviation_3 = max(min_vals_3) - min(min_vals_3)
    
    assert deviation_3 < 10.0, (
        f"Micros timestamps not aligned in segment 3! "
        f"Deviation: {deviation_3:.3f}s"
    )


def test_boot8_unwrapping_stats():
    """Verify unwrapping statistics are reasonable for BOOT8 file."""
    dat_file = Path(__file__).parent / "DATA_BOOT_000008_TIME_20260209T004502.dat"
    
    if not dat_file.exists():
        pytest.skip(f"Test file not found: {dat_file}")
    
    result = decode_file(dat_file, show_plots=False)
    
    # Check unwrap stats if available
    if "unwrap_stats" in result:
        unwrap_stats = result["unwrap_stats"]
        
        # Unwrap stats are organized by segment, not by data type
        # Each segment has PPS, GNSS, and IMU stats
        for seg_key in ["segment_000", "segment_002", "segment_003"]:
            if seg_key in unwrap_stats:
                seg_stats = unwrap_stats[seg_key]
                
                # Check that we have stats for each data type in this segment
                for data_type in ["PPS", "GNSS", "IMU"]:
                    if data_type in seg_stats:
                        # micros_reading should exist
                        if "micros_reading" in seg_stats[data_type]:
                            wraps = seg_stats[data_type]["micros_reading"]["wraps"]
                            # Wraps should be reasonable (typically 0-2 per segment)
                            assert 0 <= wraps <= 5, (
                                f"{seg_key} {data_type} micros wraps seems unreasonable: {wraps}"
                            )
