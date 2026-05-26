"""Tests for the decoder module."""

import os
import struct
from pathlib import Path

import numpy as np
import pytest

from decoder import (
    GNSSReading,
    IMUReading,
    PPSFix,
    compute_pps_regression,
    decode_file,
    load_and_combine_segments,
    parse_gnss_entry,
    parse_header,
    parse_imu_entry,
    parse_pps_entry,
    unwrap_array,
)


def test_parse_pps_entry():
    """Test parsing a PPS entry."""
    data = struct.pack("<I", 12345)
    pps = parse_pps_entry(data)
    assert isinstance(pps, PPSFix)
    assert pps.micros_reading == 12345


def test_parse_gnss_entry():
    """Test parsing a GNSS entry."""
    packed = struct.pack(
        "<IiiiIiiiB",
        10000,
        123456789,
        -987654321,
        1705000000,
        500000,
        100,
        -50,
        25,
        3,
    ) + b"\x00\x00\x00"  # Add 3 bytes padding to reach 36 bytes
    gnss = parse_gnss_entry(packed)
    assert isinstance(gnss, GNSSReading)
    assert gnss.micros_reading == 10000
    assert gnss.latitude == 123456789
    assert gnss.longitude == -987654321
    assert gnss.posix_timestamp == 1705000000
    assert gnss.microseconds == 500000
    assert gnss.ned_vel_north == 100
    assert gnss.ned_vel_east == -50
    assert gnss.ned_vel_down == 25
    assert gnss.fix_type == 3


def test_parse_imu_entry():
    """Test parsing an IMU entry."""
    data = struct.pack("<IHhhhhhh", 5000, 42, 100, -200, 300, 1000, -2000, 3000)
    imu = parse_imu_entry(data, acc_sensitivity=0.061, gyr_sensitivity=4.375)
    assert isinstance(imu, IMUReading)
    assert imu.micros_reading == 5000
    assert imu.counter == 42
    assert imu.acc_x == 100
    assert imu.acc_y == -200
    assert imu.acc_z == 300
    assert imu.gyr_x == 1000
    assert imu.gyr_y == -2000
    assert imu.gyr_z == 3000
    assert abs(imu.acc_x_mg - 100 * 0.061) < 1e-6
    assert abs(imu.acc_y_mg - (-200) * 0.061) < 1e-6
    assert abs(imu.acc_z_mg - 300 * 0.061) < 1e-6
    assert abs(imu.gyr_x_mdps - 1000 * 4.375) < 1e-6
    assert abs(imu.gyr_y_mdps - (-2000) * 4.375) < 1e-6
    assert abs(imu.gyr_z_mdps - 3000 * 4.375) < 1e-6


def test_parse_header(tmp_path):
    """Test parsing the file header."""
    test_file = tmp_path / "test_header.dat"
    header_content = """Log start OLA ISM330DHCX SAM-M10Q logger

Firmware commit ID: 391b428a3e869543ebd2caf1626f845730858f8b
ISM330DHCX Acc sensitivity (mg/LSB): 0.061000
ISM330DHCX Gyr sensitivity (mdps/LSB): 4.375000
ISM330DHCX ODR (Hz): 417.00
GNSS update rate (Hz): 10
"""
    test_file.write_text(header_content)

    header_info, header_text = parse_header(test_file)
    assert header_info["acc_sensitivity"] == 0.061
    assert header_info["gyr_sensitivity"] == 4.375
    assert header_info["imu_odr"] == 417.0
    assert header_info["gnss_rate"] == 10.0
    assert header_info["firmware_commit"] == "391b428a3e869543ebd2caf1626f845730858f8b"
    assert isinstance(header_text, str)
    assert "Log start OLA ISM330DHCX SAM-M10Q logger" in header_text
    assert "Firmware commit ID" in header_text


def test_decode_file_with_real_data():
    """Test decoding with the real data file if it exists."""
    test_file = Path("DATA_BOOT_0000_TIME_20260204T193000.dat")
    if not test_file.exists():
        pytest.skip("Real data file not found")

    output_files = decode_file(test_file)

    assert "file" in output_files
    assert "unwrap_stats" in output_files

    # Load from compressed archive using helper function
    combined_data = load_and_combine_segments(output_files["file"])
    pps_data = combined_data["pps"]
    gnss_data = combined_data["gnss"]
    imu_data = combined_data["imu"]

    assert len(pps_data) > 0
    assert len(gnss_data) > 0
    assert len(imu_data) > 0

    assert isinstance(pps_data[0], PPSFix)
    assert isinstance(gnss_data[0], GNSSReading)
    assert isinstance(imu_data[0], IMUReading)
    
    # Verify new format fields exist
    assert hasattr(pps_data[0], 'micros_reading')
    assert hasattr(gnss_data[0], 'micros_reading')
    assert hasattr(imu_data[0], 'micros_reading')
    assert hasattr(imu_data[0], 'counter')
    
    # Verify counter increments (check first 10 entries)
    for i in range(1, min(10, len(imu_data))):
        # Allow for wrapping at uint16 max
        expected = (imu_data[i-1].counter + 1) % (2**16)
        assert imu_data[i].counter == expected or imu_data[i].counter == 0, \
            f"Counter not incrementing correctly at index {i}"
    
    # Verify micros timestamps are increasing
    for i in range(1, min(100, len(imu_data))):
        assert imu_data[i].micros_reading >= imu_data[i-1].micros_reading, \
            f"Micros not increasing at index {i}"

    for key, output_file in output_files.items():
        if key != "unwrap_stats" and output_file.exists():
            output_file.unlink()


def test_decode_file_synthetic(tmp_path):
    """Test decoding with synthetic data."""
    test_file = tmp_path / "test_data.dat"

    header = """Log start OLA ISM330DHCX SAM-M10Q logger

Firmware commit ID: test123
ISM330DHCX Acc sensitivity (mg/LSB): 0.061000
ISM330DHCX Gyr sensitivity (mdps/LSB): 4.375000
ISM330DHCX ODR (Hz): 417.00
GNSS update rate (Hz): 10
"""

    pps_entry = b"\nPPS" + struct.pack("<I", 1000)
    gnss_entry = (
        b"\nGPS"
        + struct.pack(
            "<IiiiIiiiB", 2000, 12345678, -87654321, 1705000000, 0, 10, 20, 5, 3
        )
        + b"\x00\x00\x00"
    )  # pad to 36 bytes (33 struct + 3 padding)
    # IMU with padding: 18-byte struct + 2-byte padding
    imu_entry = b"\nIMU" + struct.pack("<IHhhhhhh", 3000, 0, 100, 200, 300, 50, 100, 150) + b"\x00\x00"
    footer = b"\n\nLog stop OLA ISM330DHCX SAM-M10Q logger\n"

    with open(test_file, "wb") as f:
        f.write(header.encode("utf-8"))
        f.write(pps_entry)
        f.write(gnss_entry)
        f.write(imu_entry)
        f.write(footer)

    output_files = decode_file(test_file, output_dir=tmp_path)

    assert "file" in output_files
    assert "unwrap_stats" in output_files

    # Load from compressed archive using helper function
    combined_data = load_and_combine_segments(output_files["file"])
    pps_data = combined_data["pps"]
    gnss_data = combined_data["gnss"]
    imu_data = combined_data["imu"]

    assert len(pps_data) == 1
    assert len(gnss_data) == 1
    assert len(imu_data) == 1

    assert pps_data[0].micros_reading == 1000
    assert gnss_data[0].micros_reading == 2000
    assert gnss_data[0].latitude == 12345678
    assert imu_data[0].micros_reading == 3000
    assert imu_data[0].counter == 0
    assert imu_data[0].acc_x == 100


def test_parse_entry_assertions():
    """Test that assertions catch incorrect data sizes."""
    # Test PPS assertion
    with pytest.raises(AssertionError, match="PPS data size mismatch"):
        parse_pps_entry(b"123")  # Too short

    with pytest.raises(AssertionError, match="PPS data size mismatch"):
        parse_pps_entry(b"12345")  # Too long

    # Test GNSS assertion - should expect exactly 36 bytes (33 struct + 3 padding)
    with pytest.raises(AssertionError, match="GNSS data size mismatch"):
        parse_gnss_entry(b"1" * 33)  # Too short (missing padding)

    with pytest.raises(AssertionError, match="GNSS data size mismatch"):
        parse_gnss_entry(b"1" * 40)  # Too long

    # Test IMU assertion - now expects 18 bytes (4+2+6*2)
    with pytest.raises(AssertionError, match="IMU data size mismatch"):
        parse_imu_entry(b"1" * 10)  # Too short

    with pytest.raises(AssertionError, match="IMU data size mismatch"):
        parse_imu_entry(b"1" * 20)  # Too long


def test_compute_pps_regression():
    """Test PPS to UTC regression computation."""
    # Create synthetic PPS and GNSS data with known relationship
    # micros = slope * utc + intercept
    # For simplicity: 1000000 us = 1 second UTC starting at t=1000000
    pps_list = [
        PPSFix(micros_reading=1000000),
        PPSFix(micros_reading=2000000),
        PPSFix(micros_reading=3000000),
    ]

    gnss_list = [
        GNSSReading(
            micros_reading=1000000, latitude=0, longitude=0,
            posix_timestamp=1, microseconds=0,
            ned_vel_north=0, ned_vel_east=0, ned_vel_down=0, fix_type=3,
            latitude_dd=0.0, longitude_dd=0.0,
            ned_vel_north_mmps=0, ned_vel_east_mmps=0, ned_vel_down_mmps=0,
            datetime_utc=None
        ),
        GNSSReading(
            micros_reading=2000000, latitude=0, longitude=0,
            posix_timestamp=2, microseconds=0,
            ned_vel_north=0, ned_vel_east=0, ned_vel_down=0, fix_type=3,
            latitude_dd=0.0, longitude_dd=0.0,
            ned_vel_north_mmps=0, ned_vel_east_mmps=0, ned_vel_down_mmps=0,
            datetime_utc=None
        ),
        GNSSReading(
            micros_reading=3000000, latitude=0, longitude=0,
            posix_timestamp=3, microseconds=0,
            ned_vel_north=0, ned_vel_east=0, ned_vel_down=0, fix_type=3,
            latitude_dd=0.0, longitude_dd=0.0,
            ned_vel_north_mmps=0, ned_vel_east_mmps=0, ned_vel_down_mmps=0,
            datetime_utc=None
        ),
    ]

    regression = compute_pps_regression(pps_list, gnss_list)
    assert regression is not None
    slope, intercept, r_squared = regression

    # Expected: utc = slope * micros + intercept
    # With our data: utc=1 at micros=1000000, utc=2 at micros=2000000, etc.
    # slope should be 1e-6 (1 second per 1000000 micros)
    assert abs(slope - 1e-6) < 1e-9
    # intercept should be 0 (utc = 1e-6 * micros + 0)
    assert abs(intercept - 0.0) < 1e-6
    # R² should be perfect for this synthetic data
    assert r_squared > 0.999


def test_imu_padding_handling(tmp_path):
    """Test that IMU entries with 2-byte padding are parsed correctly."""
    test_file = tmp_path / "test_padding.dat"
    
    header = """Log start OLA ISM330DHCX SAM-M10Q logger

Firmware commit ID: test_padding
ISM330DHCX Acc sensitivity (mg/LSB): 0.061000
ISM330DHCX Gyr sensitivity (mdps/LSB): 4.375000
ISM330DHCX ODR (Hz): 208.00
GNSS update rate (Hz): 10
"""
    
    # Create multiple IMU entries with padding to test sequential parsing
    imu_entries = []
    for i in range(5):
        micros = 1000000 + i * 5000  # ~5ms between samples
        counter = 100 + i
        # 18-byte struct + 2-byte padding
        imu_entry = b"\nIMU" + struct.pack("<IHhhhhhh", 
                                           micros, counter,
                                           100+i, 200+i, 300+i,
                                           50+i, 100+i, 150+i) + b"\x00\x00"
        imu_entries.append(imu_entry)
    
    footer = b"\n\nLog stop OLA ISM330DHCX SAM-M10Q logger\n"
    
    with open(test_file, "wb") as f:
        f.write(header.encode("utf-8"))
        for entry in imu_entries:
            f.write(entry)
        f.write(footer)
    
    # Decode the file
    output_files = decode_file(test_file, output_dir=tmp_path)
    combined_data = load_and_combine_segments(output_files["file"])
    imu_data = combined_data["imu"]
    
    # Should have parsed all 5 entries
    assert len(imu_data) == 5
    
    # Verify each entry
    for i in range(5):
        assert imu_data[i].micros_reading == 1000000 + i * 5000
        assert imu_data[i].counter == 100 + i
        assert imu_data[i].acc_x == 100 + i
        assert imu_data[i].acc_y == 200 + i
        assert imu_data[i].acc_z == 300 + i
        assert imu_data[i].gyr_x == 50 + i
        assert imu_data[i].gyr_y == 100 + i
        assert imu_data[i].gyr_z == 150 + i


def test_unwrap_array_no_wrapping():
    """Test unwrap_array with no wrapping or jumps."""
    values = np.array([1000, 2000, 3000, 4000])
    unwrapped, wraps, jumps, final_offset, last_raw = unwrap_array(values, max_value=2**32)
    
    assert np.array_equal(unwrapped, values)
    assert wraps is None
    assert jumps is None
    assert final_offset == 0
    assert last_raw == 4000


def test_unwrap_array_with_wrapping():
    """Test unwrap_array with wrapping at uint32_t boundary."""
    UINT32_MAX = 2**32
    # Simulate wrap: values go from near max to near zero
    values = np.array([UINT32_MAX - 1000, UINT32_MAX - 500, 100, 500])
    unwrapped, wraps, jumps, final_offset, last_raw = unwrap_array(values, max_value=UINT32_MAX)
    
    # After unwrapping, values should be monotonic
    assert unwrapped[0] == UINT32_MAX - 1000
    assert unwrapped[1] == UINT32_MAX - 500
    assert unwrapped[2] == UINT32_MAX + 100
    assert unwrapped[3] == UINT32_MAX + 500
    
    # Should detect one wrap at index 2
    assert wraps is not None
    assert len(wraps) == 1
    assert wraps[0] == 2
    
    # No jumps (monotonic after unwrapping)
    assert jumps is None
    
    # Final offset should be UINT32_MAX
    assert final_offset == UINT32_MAX
    assert last_raw == 500


def test_unwrap_array_with_negative_jump():
    """Test unwrap_array detecting negative jumps (backwards in time)."""
    values = np.array([1000, 2000, 3000, 2500, 4000])  # value[3] goes backwards
    unwrapped, wraps, jumps, final_offset, last_raw = unwrap_array(values, max_value=2**32)
    
    # No wraps (no large backwards jumps)
    assert wraps is None
    
    # Should detect negative jump at index 3
    assert jumps is not None
    assert 3 in jumps


def test_unwrap_array_with_large_forward_jump():
    """Test unwrap_array detecting large forward jumps."""
    UINT32_MAX = 2**32
    jump_threshold = 0.1 * UINT32_MAX
    
    values = np.array([1000, 2000, 3000, 3000 + jump_threshold + 1000])
    unwrapped, wraps, jumps, final_offset, last_raw = unwrap_array(values, max_value=UINT32_MAX)
    
    # No wraps
    assert wraps is None
    
    # Should detect large jump at index 3
    assert jumps is not None
    assert 3 in jumps


def test_unwrap_array_counter():
    """Test unwrap_array with uint16 counter and jump detection."""
    UINT16_MAX = 2**16

    # Counter wraps at 65536 and has one missed sample (jump by 2 instead of 1)
    values = np.array([65534, 65535, 0, 1, 3, 4])  # Jump at index 4 (1->3, diff=2 > threshold 1)
    unwrapped, wraps, jumps, final_offset, last_raw = unwrap_array(
        values, max_value=UINT16_MAX, jump_threshold=1
    )

    # Should detect wrap at index 2
    assert wraps is not None
    assert 2 in wraps

    # After unwrapping: [65534, 65535, 65536, 65537, 65539, 65540]
    # Should detect jump at index 4 (diff=2, exceeds threshold of 1)
    assert jumps is not None
    assert 4 in jumps
    
    # Final offset should be UINT16_MAX
    assert final_offset == UINT16_MAX
    assert last_raw == 4


def test_unwrap_array_empty():
    """Test unwrap_array with empty array."""
    values = np.array([])
    unwrapped, wraps, jumps, final_offset, last_raw = unwrap_array(values, max_value=2**32)
    
    assert len(unwrapped) == 0
    assert wraps is None
    assert jumps is None
    assert final_offset == 0
    assert last_raw is None


def test_unwrap_offset_carryover():
    """Test that unwrap offsets carry over correctly between segments."""
    # Simulate two segments where second segment continues from first
    
    # Segment 1: ends near a wrap boundary (use 900 to be close to 1000)
    seg1_values = np.array([700, 800, 900])
    seg1_unwrapped, _, _, seg1_final_offset, seg1_last_raw = unwrap_array(seg1_values, max_value=1000)
    
    assert seg1_unwrapped[0] == 700
    assert seg1_unwrapped[2] == 900
    assert seg1_final_offset == 0  # No wraps yet
    assert seg1_last_raw == 900
    
    # Segment 2: wraps to low values (900 -> 10 is diff of -890, exceeds threshold of -750)
    seg2_values = np.array([10, 100, 200])
    seg2_unwrapped, wraps, _, seg2_final_offset, seg2_last_raw = unwrap_array(
        seg2_values, max_value=1000, initial_offset=seg1_final_offset, prev_raw_value=seg1_last_raw
    )
    
    # With proper detection, should be [1010, 1100, 1200]
    assert seg2_unwrapped[0] == 1010, f"Expected 1010, got {seg2_unwrapped[0]}"
    assert seg2_unwrapped[2] == 1200, f"Expected 1200, got {seg2_unwrapped[2]}"
    assert wraps is not None, "Should detect wrap"
    assert len(wraps) == 1, "Should detect one wrap"
    assert wraps[0] == 0, "Wrap should be at first element (segment boundary)"
    assert seg2_final_offset == 1000, f"Expected offset 1000, got {seg2_final_offset}"
    assert seg2_last_raw == 200
    
    # Segment 3: continues with carried offset, no wrap within segment
    seg3_values = np.array([250, 300, 350])
    seg3_unwrapped, _, _, seg3_final_offset, _ = unwrap_array(
        seg3_values, max_value=1000, initial_offset=seg2_final_offset, prev_raw_value=seg2_last_raw
    )
    
    assert seg3_unwrapped[0] == 1250, f"Expected 1250, got {seg3_unwrapped[0]}"
    assert seg3_unwrapped[2] == 1350, f"Expected 1350, got {seg3_unwrapped[2]}"


def test_unwrap_counter_carryover():
    """Test that IMU counter unwrap offsets carry over between segments."""
    UINT16_MAX = 2**16
    
    # Segment 1: counter goes from 65530 to 65535
    seg1_values = np.array([65530, 65531, 65532, 65533, 65534, 65535])
    seg1_unwrapped, _, _, seg1_offset, seg1_last_raw = unwrap_array(seg1_values, max_value=UINT16_MAX, jump_threshold=1)
    
    assert seg1_unwrapped[-1] == 65535
    assert seg1_offset == 0  # No wrap yet
    assert seg1_last_raw == 65535
    
    # Segment 2: counter wraps to 0, 1, 2, ...
    seg2_values = np.array([0, 1, 2, 3, 4, 5])
    seg2_unwrapped, wraps, _, seg2_offset, _ = unwrap_array(
        seg2_values, max_value=UINT16_MAX, jump_threshold=1, initial_offset=seg1_offset, prev_raw_value=seg1_last_raw
    )
    
    # Should detect wrap and continue monotonically
    assert wraps is not None
    assert len(wraps) == 1
    assert wraps[0] == 0  # Wrap at first element (segment boundary)
    assert seg2_unwrapped[0] == 65536, f"Expected 65536, got {seg2_unwrapped[0]}"
    assert seg2_unwrapped[-1] == 65541, f"Expected 65541, got {seg2_unwrapped[-1]}"
    assert seg2_offset == UINT16_MAX
    
    # Verify monotonic increase from seg1 to seg2
    assert seg2_unwrapped[0] > seg1_unwrapped[-1]
    assert seg2_unwrapped[0] - seg1_unwrapped[-1] == 1  # Should increment by 1


def test_small_segment_handling():
    """Test that small segments (insufficient PPS for regression) are skipped gracefully.
    
    Uses the real file DATA_BOOT_000093_TIME_20260208T133001.dat which has a small
    final segment with only 1 PPS entry (insufficient for regression).
    """
    from pathlib import Path
    from decoder import decode_file, load_and_combine_segments
    
    # Test with real file that has a small segment
    test_file = Path("DATA_BOOT_000093_TIME_20260208T133001.dat")
    if not test_file.exists():
        pytest.skip("Test file with small segment not available")
    
    # This file has 17 segments total, with segment 16 being too small (1 PPS)
    # Decode should succeed by skipping the small segment
    from tempfile import TemporaryDirectory
    
    with TemporaryDirectory() as tmpdir:
        result = decode_file(test_file, output_dir=Path(tmpdir))
        assert result is not None, "Decode should succeed even with small segment"
        
        # Extract the NPZ file path from the result dict
        npz_file = result['file']
        
        # Load the data
        data = load_and_combine_segments(npz_file)
        
        # Verify data was loaded successfully
        assert len(data['pps']) > 0, "Should have PPS data"
        assert len(data['gnss']) > 0, "Should have GNSS data"
        assert len(data['imu']) > 0, "Should have IMU data"
        
        # Verify all IMU entries have valid UTC timestamps
        # (small segment without regression was skipped)
        none_count = sum(
            1 for imu in data['imu'] 
            if imu.utc_timestamp_from_pps_regression is None
        )
        assert none_count == 0, \
            f"All IMU entries should have UTC timestamps (small segments skipped), " \
            f"but {none_count} entries have None"
        
        # Verify timestamps are reasonable
        valid_timestamps = [
            imu.utc_timestamp_from_pps_regression 
            for imu in data['imu'] 
            if imu.utc_timestamp_from_pps_regression is not None
        ]
        assert all(ts > 1.7e9 for ts in valid_timestamps), \
            "All timestamps should be reasonable (> year 2023)"


def test_bad_regression_filtering():
    """Test that segments with poor PPS regression quality are filtered out.
    
    This test verifies the fix for a real-world issue where segment 11 of
    DATA_BOOT_000093_TIME_20260208T141501.dat previously had R²=0.56 and
    max mismatch=498ms. After implementing outlier filtering in PPS regression,
    this segment should now have good quality (R² > 0.99, mismatch < 200ms).
    
    This test ensures no regression in PPS regression quality.
    """
    from pathlib import Path
    from decoder import decode_file, load_and_combine_segments
    from tempfile import TemporaryDirectory
    
    # Test with the file that previously had a bad segment 11
    test_file = Path("DATA_BOOT_000093_TIME_20260208T141501.dat")
    if not test_file.exists():
        pytest.skip("Test file with historically bad segment not available")
    
    with TemporaryDirectory() as tmpdir:
        result = decode_file(test_file, output_dir=Path(tmpdir))
        assert result is not None, "Decode should succeed"
        
        npz_file = result['file']
        data = load_and_combine_segments(npz_file)
        
        # Verify we have data
        assert len(data['pps']) > 0, "Should have PPS data"
        assert len(data['gnss']) > 0, "Should have GNSS data"
        assert len(data['imu']) > 0, "Should have IMU data"
        
        # Verify all IMU entries have valid UTC timestamps
        # (segments with bad regression would have been filtered out)
        none_count = sum(
            1 for imu in data['imu'] 
            if imu.utc_timestamp_from_pps_regression is None
        )
        assert none_count == 0, \
            f"All IMU entries should have UTC timestamps, but {none_count} have None"
        
        # Verify all timestamps are reasonable
        valid_timestamps = [
            imu.utc_timestamp_from_pps_regression 
            for imu in data['imu'] 
            if imu.utc_timestamp_from_pps_regression is not None
        ]
        assert len(valid_timestamps) > 0, "Should have valid timestamps"
        assert all(ts > 1.7e9 for ts in valid_timestamps), \
            "All timestamps should be reasonable (> year 2023)"
        
        # The key test: verify the file was fully processed without
        # discarding segments due to bad regression
        # (with the fix, segment 11 should be good now)
        # We expect all 16-17 segments to be valid
        assert len(data['imu']) > 150000, \
            f"Should have most IMU data (~198K expected), got {len(data['imu'])}"


def test_simple_example_script():
    """Test that simple_example.py runs without errors.
    
    This ensures the recommended user-facing example remains functional.
    """
    from pathlib import Path
    import subprocess
    import sys
    
    # Check if the example data file exists
    test_file = Path("DATA_BOOT_0000_TIME_20260204T193000.dat")
    if not test_file.exists():
        pytest.skip("Test data file not available")
    
    # Run the simple_example.py script
    # Use subprocess to run it in a separate process
    script_path = Path(__file__).parent / "simple_example.py"
    
    # Set MPLBACKEND to non-interactive backend to avoid hanging on plt.show()
    env = {'MPLBACKEND': 'Agg', 'PATH': os.environ.get('PATH', '')}
    
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=120,  # 2 minute timeout
        env=env,
        cwd=Path(__file__).parent
    )
    
    # Check that the script ran successfully
    assert result.returncode == 0, \
        f"simple_example.py failed with return code {result.returncode}.\n" \
        f"STDOUT:\n{result.stdout}\n" \
        f"STDERR:\n{result.stderr}"
    
    # Check for expected output strings
    assert "Decoding:" in result.stdout or "Decoding:" in result.stderr, \
        "Expected 'Decoding:' in output"
    assert "DECODED DATA SUMMARY" in result.stdout, \
        "Expected 'DECODED DATA SUMMARY' in output"
    assert "CREATING PLOTS" in result.stdout, \
        "Expected 'CREATING PLOTS' in output"
    assert "Done!" in result.stdout, \
        "Expected 'Done!' in output"



