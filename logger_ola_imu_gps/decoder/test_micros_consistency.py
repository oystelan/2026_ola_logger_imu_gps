"""Tests for the micros timestamp consistency check."""

from datetime import datetime, timezone

import pytest

from decoder import (
    GNSSReading,
    IMUReading,
    PPSFix,
    check_micros_consistency,
)


def make_gnss(micros, unwrapped_micros):
    """Helper to create a GNSSReading with minimal fields."""
    return GNSSReading(
        micros_reading=micros,
        micros_reading_unwrapped=unwrapped_micros,
        latitude=0, longitude=0, posix_timestamp=0, microseconds=0,
        ned_vel_north=0, ned_vel_east=0, ned_vel_down=0, fix_type=3,
        latitude_dd=0.0, longitude_dd=0.0,
        ned_vel_north_mmps=0, ned_vel_east_mmps=0, ned_vel_down_mmps=0,
        datetime_utc=datetime.now(timezone.utc)
    )


def make_imu(micros, unwrapped_micros, counter=0):
    """Helper to create an IMUReading with minimal fields."""
    return IMUReading(
        micros_reading=micros,
        micros_reading_unwrapped=unwrapped_micros,
        counter=counter, acc_x=0, acc_y=0, acc_z=0,
        gyr_x=0, gyr_y=0, gyr_z=0,
        acc_x_mg=0.0, acc_y_mg=0.0, acc_z_mg=0.0,
        gyr_x_mdps=0.0, gyr_y_mdps=0.0, gyr_z_mdps=0.0
    )


def test_micros_consistency_check_passes():
    """Test that consistency check passes when timestamps are aligned."""
    # Create test data with consistent timestamps (all within 1 second)
    base_micros = 1_000_000_000  # 1 second
    
    pps_list = [
        PPSFix(micros_reading=base_micros, micros_reading_unwrapped=base_micros),
        PPSFix(micros_reading=base_micros + 1_000_000, micros_reading_unwrapped=base_micros + 1_000_000),
    ]
    
    gnss_list = [make_gnss(base_micros + 100_000, base_micros + 100_000)]
    imu_list = [
        make_imu(base_micros + 50_000, base_micros + 50_000, 0),
        make_imu(base_micros + 500_000, base_micros + 500_000, 1),
    ]
    
    # Should not raise an exception
    check_micros_consistency(pps_list, gnss_list, imu_list, segment_idx=0)


def test_micros_consistency_check_fails_min_deviation():
    """Test that consistency check fails when min timestamps deviate too much."""
    # Create test data where PPS starts 15 seconds before GNSS/IMU
    base_micros = 1_000_000_000
    
    pps_list = [
        PPSFix(micros_reading=base_micros, micros_reading_unwrapped=base_micros),
    ]
    
    gnss_list = [make_gnss(base_micros + 15_000_000, base_micros + 15_000_000)]
    imu_list = [make_imu(base_micros + 14_500_000, base_micros + 14_500_000)]
    
    # Should raise ValueError
    with pytest.raises(ValueError, match="failed micros consistency check"):
        check_micros_consistency(pps_list, gnss_list, imu_list, segment_idx=0)


def test_micros_consistency_check_fails_max_deviation():
    """Test that consistency check fails when max timestamps deviate too much."""
    # Create test data where PPS ends 15 seconds after GNSS/IMU
    base_micros = 1_000_000_000
    
    pps_list = [
        PPSFix(micros_reading=base_micros, micros_reading_unwrapped=base_micros),
        PPSFix(micros_reading=base_micros + 20_000_000, micros_reading_unwrapped=base_micros + 20_000_000),
    ]
    
    gnss_list = [make_gnss(base_micros + 5_000_000, base_micros + 5_000_000)]
    imu_list = [make_imu(base_micros + 4_000_000, base_micros + 4_000_000)]
    
    # Should raise ValueError (max deviation is 16s)
    with pytest.raises(ValueError, match="failed micros consistency check"):
        check_micros_consistency(pps_list, gnss_list, imu_list, segment_idx=0)


def test_micros_consistency_check_with_one_data_type():
    """Test that consistency check is skipped with only one data type."""
    # Create test data with only IMU
    base_micros = 1_000_000_000
    
    imu_list = [make_imu(base_micros, base_micros)]
    
    # Should not raise an exception (check is skipped)
    check_micros_consistency([], [], imu_list, segment_idx=0)


def test_micros_consistency_check_at_threshold():
    """Test that consistency check passes exactly at the 10s threshold."""
    # Create test data with exactly 10 seconds deviation
    base_micros = 1_000_000_000
    
    pps_list = [
        PPSFix(micros_reading=base_micros, micros_reading_unwrapped=base_micros),
    ]
    
    gnss_list = [make_gnss(base_micros + 10_000_000, base_micros + 10_000_000)]
    imu_list = [make_imu(base_micros + 5_000_000, base_micros + 5_000_000)]
    
    # Should pass (exactly at threshold)
    check_micros_consistency(pps_list, gnss_list, imu_list, segment_idx=0)


def test_micros_consistency_check_just_over_threshold():
    """Test that consistency check fails just over the 10s threshold."""
    # Create test data with 10.001 seconds deviation
    base_micros = 1_000_000_000
    
    pps_list = [
        PPSFix(micros_reading=base_micros, micros_reading_unwrapped=base_micros),
    ]
    
    gnss_list = [make_gnss(base_micros + 10_001_000, base_micros + 10_001_000)]
    imu_list = [make_imu(base_micros + 5_000_000, base_micros + 5_000_000)]
    
    # Should fail (just over threshold)
    with pytest.raises(ValueError, match="failed micros consistency check"):
        check_micros_consistency(pps_list, gnss_list, imu_list, segment_idx=0)


def test_micros_consistency_check_with_none_values():
    """Test that consistency check handles None unwrapped values."""
    # Create test data where some entries don't have unwrapped values
    base_micros = 1_000_000_000
    
    pps_list = [
        PPSFix(micros_reading=base_micros, micros_reading_unwrapped=base_micros),
        PPSFix(micros_reading=base_micros + 1_000_000, micros_reading_unwrapped=None),  # None
    ]
    
    gnss_list = [make_gnss(base_micros + 500_000, base_micros + 500_000)]
    
    # Should not raise an exception (None values are filtered out)
    check_micros_consistency(pps_list, gnss_list, [], segment_idx=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
