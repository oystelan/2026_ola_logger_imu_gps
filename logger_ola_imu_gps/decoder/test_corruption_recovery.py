"""Tests for corruption recovery in the decoder."""

import struct
from pathlib import Path

import numpy as np
import pytest

from decoder import decode_file, load_and_combine_segments


def test_corruption_recovery_with_gap(tmp_path):
    """Test decoder recovers when corruption creates a gap in data."""
    test_file = tmp_path / "test_corruption_gap.dat"

    header = """Log start OLA ISM330DHCX SAM-M10Q logger

Firmware commit ID: test_corruption
ISM330DHCX Acc sensitivity (mg/LSB): 0.061000
ISM330DHCX Gyr sensitivity (mdps/LSB): 4.375000
ISM330DHCX ODR (Hz): 417.00
GNSS update rate (Hz): 10
"""

    # Create valid entries
    imu_entry_1 = b"\nIMU" + struct.pack("<IHhhhhhh", 1000, 0, 100, 200, 300, 50, 100, 150) + b"\x00\x00"
    imu_entry_2 = b"\nIMU" + struct.pack("<IHhhhhhh", 2000, 1, 101, 201, 301, 51, 101, 151) + b"\x00\x00"

    # Create corruption: random garbage bytes
    corruption = b"GARBAGE_DATA_CORRUPT"

    # Create more valid entries after corruption
    imu_entry_3 = b"\nIMU" + struct.pack("<IHhhhhhh", 3000, 2, 102, 202, 302, 52, 102, 152) + b"\x00\x00"
    imu_entry_4 = b"\nIMU" + struct.pack("<IHhhhhhh", 4000, 3, 103, 203, 303, 53, 103, 153) + b"\x00\x00"

    footer = b"\n\nLog stop OLA ISM330DHCX SAM-M10Q logger\n"

    with open(test_file, "wb") as f:
        f.write(header.encode("utf-8"))
        f.write(imu_entry_1)
        f.write(imu_entry_2)
        f.write(corruption)  # Corruption here
        f.write(imu_entry_3)
        f.write(imu_entry_4)
        f.write(footer)

    # Decode should succeed and recover
    output_files = decode_file(test_file, output_dir=tmp_path)

    combined = load_and_combine_segments(output_files["file"])
    imu_data = combined["imu"]

    # Should have recovered all 4 IMU entries
    assert len(imu_data) == 4, f"Expected 4 IMU entries, got {len(imu_data)}"
    assert imu_data[0].micros_reading == 1000
    assert imu_data[1].micros_reading == 2000
    assert imu_data[2].micros_reading == 3000
    assert imu_data[3].micros_reading == 4000


def test_truncated_file_mid_entry(tmp_path):
    """Test decoder handles file truncated in the middle of an entry."""
    test_file = tmp_path / "test_truncated.dat"

    header = """Log start OLA ISM330DHCX SAM-M10Q logger

Firmware commit ID: test_truncated
ISM330DHCX Acc sensitivity (mg/LSB): 0.061000
ISM330DHCX Gyr sensitivity (mdps/LSB): 4.375000
ISM330DHCX ODR (Hz): 417.00
GNSS update rate (Hz): 10
"""

    # Create valid entries
    imu_entry_1 = b"\nIMU" + struct.pack("<IHhhhhhh", 1000, 0, 100, 200, 300, 50, 100, 150) + b"\x00\x00"
    imu_entry_2 = b"\nIMU" + struct.pack("<IHhhhhhh", 2000, 1, 101, 201, 301, 51, 101, 151) + b"\x00\x00"

    # Start of third entry but truncate it (only marker + partial data)
    imu_entry_3_partial = b"\nIMU" + struct.pack("<IH", 3000, 2)  # Only 6 of 18 bytes

    with open(test_file, "wb") as f:
        f.write(header.encode("utf-8"))
        f.write(imu_entry_1)
        f.write(imu_entry_2)
        f.write(imu_entry_3_partial)  # Truncated here

    # Decode should succeed with what it has
    output_files = decode_file(test_file, output_dir=tmp_path)

    combined = load_and_combine_segments(output_files["file"])
    imu_data = combined["imu"]

    # Should have only 2 complete entries
    assert len(imu_data) == 2, f"Expected 2 IMU entries, got {len(imu_data)}"
    assert imu_data[0].micros_reading == 1000
    assert imu_data[1].micros_reading == 2000


def test_corruption_no_recovery_possible(tmp_path):
    """Test decoder stops when no valid markers found after corruption."""
    test_file = tmp_path / "test_no_recovery.dat"

    header = """Log start OLA ISM330DHCX SAM-M10Q logger

Firmware commit ID: test_no_recovery
ISM330DHCX Acc sensitivity (mg/LSB): 0.061000
ISM330DHCX Gyr sensitivity (mdps/LSB): 4.375000
ISM330DHCX ODR (Hz): 417.00
GNSS update rate (Hz): 10
"""

    # Create valid entries
    imu_entry_1 = b"\nIMU" + struct.pack("<IHhhhhhh", 1000, 0, 100, 200, 300, 50, 100, 150) + b"\x00\x00"

    # Large corruption with no valid markers within scan range
    corruption = b"X" * 2000  # More than CORRUPTION_SCAN_BYTES (1024)

    with open(test_file, "wb") as f:
        f.write(header.encode("utf-8"))
        f.write(imu_entry_1)
        f.write(corruption)

    # Decode should succeed with what it has before corruption
    output_files = decode_file(test_file, output_dir=tmp_path)

    combined = load_and_combine_segments(output_files["file"])
    imu_data = combined["imu"]

    # Should have only 1 entry (before corruption)
    assert len(imu_data) == 1
    assert imu_data[0].micros_reading == 1000


def test_corruption_recovery_mixed_types(tmp_path):
    """Test decoder recovers with different data types around corruption."""
    test_file = tmp_path / "test_mixed_corruption.dat"

    header = """Log start OLA ISM330DHCX SAM-M10Q logger

Firmware commit ID: test_mixed
ISM330DHCX Acc sensitivity (mg/LSB): 0.061000
ISM330DHCX Gyr sensitivity (mdps/LSB): 4.375000
ISM330DHCX ODR (Hz): 417.00
GNSS update rate (Hz): 10
"""

    # Mix of entry types
    pps_entry_1 = b"\nPPS" + struct.pack("<I", 1000)
    gnss_entry_1 = (
        b"\nGPS"
        + struct.pack("<IiiiIiiiB", 1500, 12345678, -87654321, 1705000000, 0, 10, 20, 5, 3)
        + b"\x00\x00\x00"
    )
    imu_entry_1 = b"\nIMU" + struct.pack("<IHhhhhhh", 2000, 0, 100, 200, 300, 50, 100, 150) + b"\x00\x00"

    corruption = b"CORRUPT!"

    pps_entry_2 = b"\nPPS" + struct.pack("<I", 3000)
    imu_entry_2 = b"\nIMU" + struct.pack("<IHhhhhhh", 4000, 1, 101, 201, 301, 51, 101, 151) + b"\x00\x00"

    footer = b"\n\nLog stop OLA ISM330DHCX SAM-M10Q logger\n"

    with open(test_file, "wb") as f:
        f.write(header.encode("utf-8"))
        f.write(pps_entry_1)
        f.write(gnss_entry_1)
        f.write(imu_entry_1)
        f.write(corruption)
        f.write(pps_entry_2)
        f.write(imu_entry_2)
        f.write(footer)

    output_files = decode_file(test_file, output_dir=tmp_path)

    combined = load_and_combine_segments(output_files["file"])
    pps_data = combined["pps"]
    gnss_data = combined["gnss"]
    imu_data = combined["imu"]

    # Should have recovered all entries
    assert len(pps_data) == 2
    assert len(gnss_data) == 1
    assert len(imu_data) == 2

    assert pps_data[0].micros_reading == 1000
    assert pps_data[1].micros_reading == 3000
    assert gnss_data[0].micros_reading == 1500
    assert imu_data[0].micros_reading == 2000
    assert imu_data[1].micros_reading == 4000


def test_real_world_aborted_file():
    """Test decoder handles real-world file that was aborted during logging.
    
    This tests the actual file DATA_BOOT_0003_TIME_20260205T151500.dat
    which was interrupted (power loss, abrupt restart, etc.) mid-logging.
    The decoder should handle this gracefully, recovering all valid data.
    """
    test_file = Path("DATA_BOOT_0003_TIME_20260205T151500.dat")
    
    if not test_file.exists():
        pytest.skip(f"Real-world corrupted file not found: {test_file}")
    
    # Decode should succeed without crashing
    output_files = decode_file(test_file)
    
    # Should have extracted some data (file isn't completely corrupt)
    combined = load_and_combine_segments(output_files["file"])
    pps_data = combined["pps"]
    gnss_data = combined["gnss"]
    imu_data = combined["imu"]
    
    # Basic sanity checks - we should have gotten some data
    total_entries = len(pps_data) + len(gnss_data) + len(imu_data)
    assert total_entries > 0, "Should have extracted at least some data from the file"
    
    # IMU data should have the most entries (highest sample rate ~437 Hz)
    # For a 10MB file, we'd expect tens or hundreds of thousands of IMU samples
    # if most of the file is valid
    assert len(imu_data) > 0, "Should have IMU data"
    
    # If we have data, verify it has the expected structure
    if len(pps_data) > 0:
        assert hasattr(pps_data[0], 'micros_reading')
        assert hasattr(pps_data[0], 'micros_reading_unwrapped')
    
    if len(gnss_data) > 0:
        assert hasattr(gnss_data[0], 'micros_reading')
        assert hasattr(gnss_data[0], 'latitude')
        assert hasattr(gnss_data[0], 'longitude')
    
    if len(imu_data) > 0:
        assert hasattr(imu_data[0], 'micros_reading')
        assert hasattr(imu_data[0], 'counter')
        assert hasattr(imu_data[0], 'acc_x_mg')
        assert hasattr(imu_data[0], 'gyr_x_mdps')
    
    # Log what we recovered for visibility
    print(f"\nRecovered from aborted file {test_file}:")
    print(f"  PPS entries:  {len(pps_data):,}")
    print(f"  GNSS entries: {len(gnss_data):,}")
    print(f"  IMU entries:  {len(imu_data):,}")
    print(f"  Total:        {total_entries:,} entries")
    
    # Clean up output files
    for key, output_file in output_files.items():
        if isinstance(output_file, Path) and output_file.exists():
            output_file.unlink()
