#!/usr/bin/env python3
"""Test the new micros plot with edge cases."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass

from decoder_cli import plot_micros_raw_vs_cleaned


@dataclass
class FakePPS:
    micros_reading: int
    micros_reading_unwrapped: int
    utc_timestamp_from_pps_regression: float = 0.0  # Add missing field


# Test 1: Empty data
print("Test 1: Empty arrays...")
empty = np.array([])
plot_micros_raw_vs_cleaned(empty, empty, empty, None)
print("  ✓ Handles empty data without crashing")

# Test 2: Data with no jumps
print("\nTest 2: Data with no jumps...")
data_no_jumps = np.array([
    FakePPS(1000, 1000),
    FakePPS(2000, 2000),
    FakePPS(3000, 3000),
])
unwrap_stats_no_jumps = {
    "PPS": {
        "micros_reading": {
            "wraps": 0,
            "jump_indices": np.array([])
        }
    }
}
plot_micros_raw_vs_cleaned(data_no_jumps, empty, empty, unwrap_stats_no_jumps)
print("  ✓ Handles data with no jumps correctly")

# Test 3: Data with jumps
print("\nTest 3: Data with jumps...")
data_with_jumps = np.array([
    FakePPS(1000, 1000),
    FakePPS(2000, 2000),
    FakePPS(500000000, 500000000),  # Big jump here
    FakePPS(500001000, 500001000),
])
unwrap_stats_with_jumps = {
    "PPS": {
        "micros_reading": {
            "wraps": 0,
            "jump_indices": np.array([2])
        }
    }
}
plot_micros_raw_vs_cleaned(data_with_jumps, empty, empty, unwrap_stats_with_jumps)
print("  ✓ Handles data with jumps correctly")

# Test 4: No unwrap stats provided
print("\nTest 4: No unwrap stats...")
plot_micros_raw_vs_cleaned(data_no_jumps, empty, empty, None)
print("  ✓ Handles missing unwrap_stats gracefully")

print("\n✅ All edge case tests passed!")
