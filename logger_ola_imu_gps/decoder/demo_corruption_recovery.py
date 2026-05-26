#!/usr/bin/env python3
"""Demonstration script showing corruption recovery capabilities."""

import struct
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from loguru import logger

from decoder import decode_file


def create_corrupted_file(filepath: Path) -> None:
    """Create a test file with corruption in the middle."""
    header = """Log start OLA ISM330DHCX SAM-M10Q logger

Firmware commit ID: demo_corruption_recovery
ISM330DHCX Acc sensitivity (mg/LSB): 0.061000
ISM330DHCX Gyr sensitivity (mdps/LSB): 4.375000
ISM330DHCX ODR (Hz): 417.00
GNSS update rate (Hz): 10
"""

    logger.info("Creating demo file with intentional corruption...")

    # Create 10 valid IMU entries before corruption
    valid_entries_before = []
    for i in range(10):
        entry = b"\nIMU" + struct.pack("<IHhhhhhh", 
                                       1000 + i * 1000, i, 
                                       100 + i, 200 + i, 300 + i,
                                       50 + i, 100 + i, 150 + i) + b"\x00\x00"
        valid_entries_before.append(entry)

    # Simulate corruption (e.g., from power loss)
    corruption = b"XXXX_POWER_LOSS_CORRUPTED_DATA_XXXX" * 10

    # Create 10 valid IMU entries after corruption
    valid_entries_after = []
    for i in range(10, 20):
        entry = b"\nIMU" + struct.pack("<IHhhhhhh",
                                       1000 + i * 1000, i,
                                       100 + i, 200 + i, 300 + i,
                                       50 + i, 100 + i, 150 + i) + b"\x00\x00"
        valid_entries_after.append(entry)

    footer = b"\n\nLog stop OLA ISM330DHCX SAM-M10Q logger\n"

    with open(filepath, "wb") as f:
        f.write(header.encode("utf-8"))
        for entry in valid_entries_before:
            f.write(entry)
        logger.info(f"  Wrote {len(valid_entries_before)} valid entries")
        
        f.write(corruption)
        logger.info(f"  Inserted {len(corruption)} bytes of corruption")
        
        for entry in valid_entries_after:
            f.write(entry)
        logger.info(f"  Wrote {len(valid_entries_after)} more valid entries after corruption")
        
        f.write(footer)

    logger.info(f"Demo file created: {filepath}")


def create_truncated_file(filepath: Path) -> None:
    """Create a test file that is truncated mid-entry."""
    header = """Log start OLA ISM330DHCX SAM-M10Q logger

Firmware commit ID: demo_truncation
ISM330DHCX Acc sensitivity (mg/LSB): 0.061000
ISM330DHCX Gyr sensitivity (mdps/LSB): 4.375000
ISM330DHCX ODR (Hz): 417.00
GNSS update rate (Hz): 10
"""

    logger.info("Creating demo file with truncation...")

    # Create 5 valid IMU entries
    valid_entries = []
    for i in range(5):
        entry = b"\nIMU" + struct.pack("<IHhhhhhh",
                                       1000 + i * 1000, i,
                                       100 + i, 200 + i, 300 + i,
                                       50 + i, 100 + i, 150 + i) + b"\x00\x00"
        valid_entries.append(entry)

    # Partial entry (simulating abrupt power loss mid-write)
    partial_entry = b"\nIMU" + struct.pack("<IH", 6000, 5)  # Only marker + 6 bytes of 18

    with open(filepath, "wb") as f:
        f.write(header.encode("utf-8"))
        for entry in valid_entries:
            f.write(entry)
        logger.info(f"  Wrote {len(valid_entries)} complete entries")
        
        f.write(partial_entry)
        logger.info(f"  Wrote partial entry (truncated at {len(partial_entry)} bytes)")

    logger.info(f"Demo file created: {filepath}")


def main():
    """Demonstrate corruption recovery."""
    logger.info("=" * 80)
    logger.info("CORRUPTION RECOVERY DEMONSTRATION")
    logger.info("=" * 80)
    
    with TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Test 1: Recovery from corruption
        logger.info("\n" + "=" * 80)
        logger.info("TEST 1: File with corruption in the middle")
        logger.info("=" * 80)
        
        corrupted_file = tmpdir_path / "corrupted_demo.dat"
        create_corrupted_file(corrupted_file)
        
        logger.info("\nDecoding corrupted file...")
        output_files = decode_file(corrupted_file, output_dir=tmpdir_path)
        
        imu_data = np.load(output_files["imu"], allow_pickle=True)
        logger.info(f"\nResult: Successfully recovered {len(imu_data)} IMU entries")
        logger.info(f"  First entry micros: {imu_data[0].micros_reading}")
        logger.info(f"  Last entry micros:  {imu_data[-1].micros_reading}")
        
        if len(imu_data) == 20:
            logger.success("✓ All 20 entries recovered successfully!")
        else:
            logger.warning(f"✗ Expected 20 entries, got {len(imu_data)}")
        
        # Test 2: Truncated file
        logger.info("\n" + "=" * 80)
        logger.info("TEST 2: File truncated mid-entry (power loss during write)")
        logger.info("=" * 80)
        
        truncated_file = tmpdir_path / "truncated_demo.dat"
        create_truncated_file(truncated_file)
        
        logger.info("\nDecoding truncated file...")
        output_files = decode_file(truncated_file, output_dir=tmpdir_path)
        
        imu_data = np.load(output_files["imu"], allow_pickle=True)
        logger.info(f"\nResult: Successfully parsed {len(imu_data)} complete IMU entries")
        logger.info(f"  First entry micros: {imu_data[0].micros_reading}")
        logger.info(f"  Last entry micros:  {imu_data[-1].micros_reading}")
        
        if len(imu_data) == 5:
            logger.success("✓ All complete entries recovered, partial entry correctly skipped!")
        else:
            logger.warning(f"✗ Expected 5 entries, got {len(imu_data)}")
    
    logger.info("\n" + "=" * 80)
    logger.success("DEMONSTRATION COMPLETE")
    logger.info("=" * 80)
    logger.info("\nKey takeaways:")
    logger.info("  • Decoder can recover from corruption by scanning ahead for valid markers")
    logger.info("  • Truncated files are handled gracefully - all complete data is preserved")
    logger.info("  • Detailed error messages explain what happened and where")
    logger.info("  • No data loss before corruption point")


if __name__ == "__main__":
    main()
