#!/usr/bin/env python3
"""Simple test script for the new format data file (no pytest required)."""

import struct
from pathlib import Path

print("=" * 80)
print("TESTING DECODER WITH NEW FORMAT DATA")
print("=" * 80)

# Test 1: Check file exists
print("\n1. Checking for data file...")
data_file = Path("DATA_BOOT_0000_TIME_20260204T193000.dat")
if not data_file.exists():
    print(f"❌ FAILED: File not found: {data_file}")
    exit(1)
print(f"✅ PASSED: File exists ({data_file.stat().st_size / 1024 / 1024:.2f} MB)")

# Test 2: Verify file structure
print("\n2. Verifying file structure...")
with open(data_file, "rb") as f:
    content = f.read(4096)

# Check header
if b"ISM330DHCX Acc sensitivity" in content:
    print("✅ PASSED: Header contains sensitivity info")
else:
    print("❌ FAILED: Header missing sensitivity info")
    exit(1)

# Test 3: Parse first IMU entry manually
print("\n3. Parsing first IMU entry...")
imu_idx = content.find(b"\nIMU")
if imu_idx == -1:
    print("❌ FAILED: No IMU marker found")
    exit(1)

imu_data = content[imu_idx+4:imu_idx+4+18]
padding = content[imu_idx+4+18:imu_idx+4+20]
next_byte = content[imu_idx+4+20]

try:
    values = struct.unpack("<IHhhhhhh", imu_data)
    print(f"  micros:  {values[0]:,}")
    print(f"  counter: {values[1]}")
    print(f"  acc:     ({values[2]}, {values[3]}, {values[4]})")
    print(f"  gyr:     ({values[5]}, {values[6]}, {values[7]})")
    print(f"  padding: {padding.hex()} (expected: 0000)")
    print(f"  next:    0x{next_byte:02x} (expected: 0x0a)")
    
    if padding == b"\x00\x00" and next_byte == 0x0a:
        print("✅ PASSED: IMU structure with padding is correct")
    else:
        print("❌ FAILED: Unexpected padding or next byte")
        exit(1)
except Exception as e:
    print(f"❌ FAILED: Error parsing IMU: {e}")
    exit(1)

# Test 4: Try importing decoder module
print("\n4. Testing decoder module import...")
try:
    from decoder import (
        IMU_STRUCT_SIZE, IMU_PADDING, IMU_LINE_SIZE,
        parse_imu_entry, parse_header
    )
    print(f"  IMU_STRUCT_SIZE: {IMU_STRUCT_SIZE} bytes")
    print(f"  IMU_PADDING:     {IMU_PADDING} bytes")
    print(f"  IMU_LINE_SIZE:   {IMU_LINE_SIZE} bytes")
    
    if IMU_STRUCT_SIZE == 18 and IMU_PADDING == 2 and IMU_LINE_SIZE == 24:
        print("✅ PASSED: Decoder constants are correct")
    else:
        print("❌ FAILED: Decoder constants incorrect")
        exit(1)
except ImportError as e:
    print(f"⚠️  WARNING: Cannot import decoder (missing dependencies: {e})")
    print("   Basic structure tests passed, but full decoder test skipped")
    print("\n" + "=" * 80)
    print("✅ BASIC TESTS PASSED")
    print("=" * 80)
    print("\nTo run full tests, install dependencies:")
    print("  conda create -n decoder python=3.12")
    print("  conda activate decoder")
    print("  pip install numpy scipy loguru pytest")
    print("  pytest -v test_decoder.py::test_decode_file_with_real_data")
    exit(0)

# Test 5: Parse header
print("\n5. Parsing file header...")
try:
    header_info = parse_header(data_file)
    print(f"  acc_sensitivity: {header_info.get('acc_sensitivity')}")
    print(f"  gyr_sensitivity: {header_info.get('gyr_sensitivity')}")
    print(f"  imu_odr:         {header_info.get('imu_odr')}")
    print(f"  gnss_rate:       {header_info.get('gnss_rate')}")
    print("✅ PASSED: Header parsed successfully")
except Exception as e:
    print(f"❌ FAILED: Error parsing header: {e}")
    exit(1)

# Test 6: Parse single IMU entry
print("\n6. Parsing single IMU entry with decoder...")
try:
    imu = parse_imu_entry(imu_data, 
                          acc_sensitivity=0.061, 
                          gyr_sensitivity=4.375)
    print(f"  micros_reading: {imu.micros_reading}")
    print(f"  counter:        {imu.counter}")
    print(f"  acc_x_mg:       {imu.acc_x_mg:.2f} mg")
    print(f"  gyr_x_mdps:     {imu.gyr_x_mdps:.2f} mdps")
    print("✅ PASSED: IMU entry parsed successfully")
except Exception as e:
    print(f"❌ FAILED: Error parsing IMU entry: {e}")
    exit(1)

# Test 7: Full file decode (if numpy available)
print("\n7. Running full file decode...")
try:
    from decoder import decode_file
    import numpy as np
    
    output_files = decode_file(data_file, show_plots=False)
    
    pps_data = np.load(output_files["pps"], allow_pickle=True)
    gnss_data = np.load(output_files["gnss"], allow_pickle=True)
    imu_data = np.load(output_files["imu"], allow_pickle=True)
    
    print(f"  PPS entries:  {len(pps_data):,}")
    print(f"  GNSS entries: {len(gnss_data):,}")
    print(f"  IMU entries:  {len(imu_data):,}")
    
    # Verify first IMU entry
    print("\n  First IMU entry:")
    print(f"    micros:  {imu_data[0].micros_reading:,}")
    print(f"    counter: {imu_data[0].counter}")
    
    # Verify counter increments
    counter_ok = True
    for i in range(1, min(100, len(imu_data))):
        expected = (imu_data[i-1].counter + 1) % (2**16)
        if imu_data[i].counter != expected and imu_data[i].counter != 0:
            counter_ok = False
            break
    
    if counter_ok:
        print("  ✅ Counter increments correctly")
    else:
        print("  ⚠️  Counter has unexpected jumps")
    
    # Verify micros timestamps increase
    micros_ok = True
    for i in range(1, min(100, len(imu_data))):
        if imu_data[i].micros_reading < imu_data[i-1].micros_reading:
            micros_ok = False
            break
    
    if micros_ok:
        print("  ✅ Micros timestamps increase monotonically")
    else:
        print("  ❌ Micros timestamps not monotonic")
        exit(1)
    
    print("✅ PASSED: Full file decode successful")
    
    # Clean up
    for output_file in output_files.values():
        if output_file.exists():
            output_file.unlink()
    
except Exception as e:
    print(f"❌ FAILED: Error during full decode: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)
print("\nThe decoder is working correctly with the new format data file.")
print("Next step: Run full test suite with pytest:")
print("  pytest -v test_decoder.py")
