"""Decoder for OLA ISM330DHCX + SAM-M10Q data logger files.

This module provides comprehensive decoding of binary data files from the 
OLA (OpenLogArtemis) logger with ISM330DHCX IMU and SAM-M10Q GNSS sensors.

**Recommended Usage for End Users:**

    from pathlib import Path
    from decoder import decode_file, load_data_as_arrays
    
    # Step 1: Decode the binary file
    result = decode_file(Path("DATA_BOOT_0000_TIME_20260204T193000.dat"))
    
    # Step 2: Load as numpy arrays (easiest way)
    data = load_data_as_arrays(result['file'])
    
    # Step 3: Use the data
    print(f"IMU rate: {data['imu_odr']} Hz")
    print(f"Recording: {len(data['imu_utc'])} IMU samples")
    
    # Plot acceleration
    import matplotlib.pyplot as plt
    plt.plot(data['imu_utc'], data['imu_acc_x'])
    plt.xlabel('UTC Time (s)')
    plt.ylabel('Acceleration (mg)')
    plt.show()

**Key Functions:**

- `decode_file()`: Main function to decode a .dat file → produces .npz file
- `load_data_as_arrays()`: Load .npz file as dict of numpy arrays (recommended)
- `load_and_combine_segments()`: Load .npz file as dict of dataclass arrays (advanced)

**Features:**

- Segment-based processing with automatic quality filtering
- GPS-synchronized UTC timestamps (microsecond accuracy via PPS regression)
- Statistical outlier detection for sensor readings
- Automatic unwrapping of overflow counters
- Robust corruption recovery
- Support for both new segmented and legacy file formats

See README.md for complete documentation and examples.
"""

import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from scipy import stats

# Try to import gnuplotlib, but don't fail if not available
try:
    import gnuplotlib as gp
    GNUPLOT_AVAILABLE = True
except ImportError:
    GNUPLOT_AVAILABLE = False
    logger.warning(
        "gnuplotlib not available - plots will be disabled. "
        "Install with: pip install gnuplotlib"
    )

# Magic constants
HEADER_SEARCH_BYTES = 64 * 1024
PPS_MARKER = b"\nPPS"
GPS_MARKER = b"\nGPS"
IMU_MARKER = b"\nIMU"
FOOTER_MARKER = b"Log stop OLA"
MARKER_SIZE = 4
PPS_STRUCT_SIZE = 4
GPS_STRUCT_SIZE = 36
# When firmware logs altitude_msl_mm (int32, +4 bytes): 33 data bytes + 3 pad = 36
# vs. 37 data bytes + 3 pad = 40 with altitude. Detection key in header:
# "GNSS includes altitude_msl".
GPS_STRUCT_SIZE_WITH_ALTITUDE = 40
IMU_STRUCT_SIZE = 18
IMU_PADDING = 2  # C struct alignment padding
# Two on-disk record sizes for the IMU struct depending on whether the
# firmware logs the magnetometer (mag_x/y/z, +6 bytes):
#   - 18-byte struct + 2 padding = 20 bytes per record (no mag, legacy)
#   - 24-byte struct + 0 padding = 24 bytes per record (with mag, since 24 is 4-aligned)
IMU_STRUCT_SIZE_WITH_MAG = 24
IMU_PADDING_WITH_MAG = 0
PPS_LINE_SIZE = MARKER_SIZE + PPS_STRUCT_SIZE  # 8 bytes
GPS_LINE_SIZE = MARKER_SIZE + GPS_STRUCT_SIZE  # 40 bytes (legacy, no altitude)
GPS_LINE_SIZE_WITH_ALTITUDE = MARKER_SIZE + GPS_STRUCT_SIZE_WITH_ALTITUDE  # 44 bytes
IMU_LINE_SIZE = MARKER_SIZE + IMU_STRUCT_SIZE + IMU_PADDING  # 24 bytes (4 + 18 + 2)
IMU_LINE_SIZE_WITH_MAG = MARKER_SIZE + IMU_STRUCT_SIZE_WITH_MAG + IMU_PADDING_WITH_MAG  # 28 bytes


@dataclass
class PPSFix:
    """PPS fix data structure."""

    micros_reading: int
    micros_reading_unwrapped: int | None = None
    utc_timestamp_from_pps_regression: float | None = None
    datetime_timestamp_from_pps_regression: datetime | None = None


@dataclass
class GNSSReading:
    """GNSS reading data structure."""

    micros_reading: int
    latitude: int
    longitude: int
    posix_timestamp: int
    microseconds: int
    ned_vel_north: int
    ned_vel_east: int
    ned_vel_down: int
    fix_type: int
    latitude_dd: float
    longitude_dd: float
    ned_vel_north_mmps: int
    ned_vel_east_mmps: int
    ned_vel_down_mmps: int
    datetime_utc: datetime
    # Altitude above mean sea level, only populated when firmware logs it.
    # 0.0 / 0 are the sentinel values for "no altitude in this record".
    altitude_msl_mm: int = 0
    altitude_msl_m: float = 0.0
    micros_reading_unwrapped: int | None = None
    utc_timestamp_from_pps_regression: float | None = None
    datetime_timestamp_from_pps_regression: datetime | None = None
    # Outlier detection flags
    latitude_dd_stdchecked: bool = False
    longitude_dd_stdchecked: bool = False
    ned_vel_north_mmps_stdchecked: bool = False
    ned_vel_east_mmps_stdchecked: bool = False
    ned_vel_down_mmps_stdchecked: bool = False
    altitude_msl_m_stdchecked: bool = False


@dataclass
class IMUReading:
    """IMU reading data structure.

    The magnetometer fields (mag_x/y/z, mag_*_uT) are set when the firmware
    logs the magnetometer (ICM-20948 with AK09916). On legacy data without
    mag, these stay at their default 0 values — check the presence of
    'mag_sensitivity' in the header to know whether the column is meaningful.
    """

    micros_reading: int
    counter: int
    acc_x: int
    acc_y: int
    acc_z: int
    gyr_x: int
    gyr_y: int
    gyr_z: int
    acc_x_mg: float
    acc_y_mg: float
    acc_z_mg: float
    gyr_x_mdps: float
    gyr_y_mdps: float
    gyr_z_mdps: float
    # Magnetometer (AK09916, only populated when firmware logs mag)
    mag_x: int = 0
    mag_y: int = 0
    mag_z: int = 0
    mag_x_uT: float = 0.0
    mag_y_uT: float = 0.0
    mag_z_uT: float = 0.0
    micros_reading_unwrapped: int | None = None
    counter_unwrapped: int | None = None
    utc_timestamp_from_pps_regression: float | None = None
    datetime_timestamp_from_pps_regression: datetime | None = None
    # Outlier detection flags
    acc_x_mg_stdchecked: bool = False
    acc_y_mg_stdchecked: bool = False
    acc_z_mg_stdchecked: bool = False
    gyr_x_mdps_stdchecked: bool = False
    gyr_y_mdps_stdchecked: bool = False
    gyr_z_mdps_stdchecked: bool = False
    mag_x_uT_stdchecked: bool = False
    mag_y_uT_stdchecked: bool = False
    mag_z_uT_stdchecked: bool = False


def parse_header(
    file_path: Path,
    markers: tuple[bytes, bytes, bytes] = (PPS_MARKER, GPS_MARKER, IMU_MARKER),
    search_bytes: int = HEADER_SEARCH_BYTES,
) -> tuple[dict[str, Any], str]:
    """Parse the header of the data file and extract metadata.

    Args:
        file_path: Path to the data file
        markers: Tuple of markers that indicate start of data section
        search_bytes: Number of bytes to scan from start of file for header

    Returns:
        Tuple of (header_info dict, header_text string)
    """
    header_info = {}

    with open(file_path, "rb") as f:
        content = f.read(search_bytes)

    marker_positions = [
        pos for m in markers if (pos := content.find(m)) != -1
    ]
    header_end = min(marker_positions) if marker_positions else len(content)
    header_text = content[:header_end].decode("utf-8", errors="ignore")
    header_lines = header_text.splitlines()

    # Match the sensitivity/ODR lines by their SUFFIX (units / label), not by
    # the chip name prefix, so the parser keeps working when the IMU is swapped.
    # Original used "ISM330DHCX Acc sensitivity"; the OLA's built-in IMU emits
    # "ICM-20948 Acc sensitivity"; future chips will follow the same pattern.
    for line in header_lines:
        if "Acc sensitivity" in line:
            parts = line.split(":")
            if len(parts) == 2:
                header_info["acc_sensitivity"] = float(parts[1].strip())
        elif "Gyr sensitivity" in line:
            parts = line.split(":")
            if len(parts) == 2:
                header_info["gyr_sensitivity"] = float(parts[1].strip())
        elif "Mag sensitivity" in line:
            # AK09916 inside the ICM-20948. Header line example:
            #   "ICM-20948 Mag sensitivity (uT/LSB): 0.150000 (AK09916 fixed)"
            # Take the first numeric token after the colon.
            parts = line.split(":")
            if len(parts) >= 2:
                # Strip trailing "(AK09916 fixed)" annotation and parse.
                rhs = parts[1].strip().split()[0]
                try:
                    header_info["mag_sensitivity"] = float(rhs)
                except ValueError:
                    pass
        elif "ODR" in line and "Hz" in line:
            # "ICM-20948 ODR (Hz): 225.00" — chip-agnostic match on "ODR" + "Hz".
            parts = line.split(":")
            if len(parts) == 2:
                header_info["imu_odr"] = float(parts[1].strip())
        elif "GNSS update rate" in line:
            parts = line.split(":")
            if len(parts) == 2:
                header_info["gnss_rate"] = float(parts[1].strip())
        elif "GNSS includes altitude_msl" in line:
            # Marker line from firmware versions that log altitude in the
            # GNSS struct. Triggers the wider GPS record parsing path.
            header_info["gnss_has_altitude"] = True
        elif "Firmware commit ID" in line:
            parts = line.split(":")
            if len(parts) == 2:
                header_info["firmware_commit"] = parts[1].strip()

    logger.info(f"Parsed header: {header_info}")
    return header_info, header_text


def unwrap_array(
    values: np.ndarray,
    max_value: int,
    wrap_threshold: float | None = None,
    jump_threshold: float | None = None,
    initial_offset: int = 0,
    prev_raw_value: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, int, int | None]:
    """Unwrap potentially wrapping array and detect anomalous jumps.

    Handles overflow in fixed-width integer timestamps (uint32_t micros, uint16_t counters)
    by detecting wrap-around events and applying offset corrections. Also identifies
    anomalous jumps that indicate missed data or timing glitches.

    Algorithm:
    1. Scan array for large negative jumps (diff < -wrap_threshold) → wrap detected
    2. Apply cumulative offset (add max_value) to all values after each wrap
    3. On unwrapped data, detect anomalous jumps:
       - Any negative jump (should be monotonic after unwrapping)
       - Any positive jump > jump_threshold (unexpectedly large time gap)

    Args:
        values: Array of potentially wrapping values (e.g., micros_reading, counter)
        max_value: Maximum value before wrapping (e.g., 2**32 for uint32, 2**16 for uint16)
        wrap_threshold: Threshold for wrap detection as fraction of max_value
                       (default: 0.25 * max_value, meaning negative jumps > 25% are wraps)
        jump_threshold: Threshold for anomalous jump detection
                       (default: 0.1 * max_value for timestamps, 1 for counters)
        initial_offset: Initial unwrap offset from previous segment (default: 0)
        prev_raw_value: Last raw value from previous segment for wrap detection at boundary

    Returns:
        Tuple of:
        - unwrapped_array: Array with wrapping corrected (int64 to avoid overflow)
        - wrap_indices: Indices where wraps occurred (None if no wraps detected)
        - jump_indices: Indices where anomalous jumps occurred (None if no jumps detected)
        - final_offset: Final unwrap offset to pass to next segment
        - last_raw_value: Last raw value to pass to next segment

    Example:
        >>> values = np.array([2**32-1000, 2**32-500, 100])  # Wraps at index 2
        >>> unwrapped, wraps, jumps, final_offset, last_raw = unwrap_array(values, max_value=2**32)
        >>> unwrapped
        array([4294966296, 4294966796, 4294967396])  # Monotonic after unwrapping
        >>> wraps
        array([2])  # Wrap detected at index 2
    """
    if len(values) == 0:
        return np.array([]), None, None, initial_offset, prev_raw_value

    if wrap_threshold is None:
        wrap_threshold = 0.25 * max_value
    if jump_threshold is None:
        jump_threshold = 0.1 * max_value

    # Step 1: Detect wraps and unwrap
    unwrapped = np.zeros_like(values, dtype=np.int64)
    offset = initial_offset
    wrap_indices_list = []

    # Check for wrap at segment boundary (first value vs previous segment's last value)
    if prev_raw_value is not None:
        diff = values[0] - prev_raw_value
        if diff < 0 and abs(diff) > wrap_threshold:
            offset += max_value
            wrap_indices_list.append(0)
    
    unwrapped[0] = values[0] + offset

    for i in range(1, len(values)):
        current = values[i]
        prev = values[i - 1]
        diff = current - prev

        # Detect wrap: large negative jump
        if diff < 0 and abs(diff) > wrap_threshold:
            offset += max_value
            wrap_indices_list.append(i)

        unwrapped[i] = current + offset

    # Step 2: Detect anomalous jumps on unwrapped data
    jump_indices_list = []

    for i in range(1, len(unwrapped)):
        diff = unwrapped[i] - unwrapped[i - 1]

        # Any negative jump is anomalous (should be monotonic after unwrapping)
        # Any positive jump > threshold is anomalous
        if diff < 0 or diff > jump_threshold:
            jump_indices_list.append(i)

    # Convert to numpy arrays or None
    wrap_indices = (
        np.array(wrap_indices_list, dtype=np.int64)
        if wrap_indices_list
        else None
    )
    jump_indices = (
        np.array(jump_indices_list, dtype=np.int64)
        if jump_indices_list
        else None
    )
    
    # Return last raw value for next segment
    last_raw_value = int(values[-1]) if len(values) > 0 else None

    return unwrapped, wrap_indices, jump_indices, offset, last_raw_value


def detect_outliers_stdcheck(
    values: np.ndarray,
    n_neighbors: int = 6,
    n_sigma: float = 5.0
) -> np.ndarray:
    """Detect outliers in a time series using neighboring values statistics.
    
    For each point in the time series, computes statistics from its N nearest
    neighbors and flags points that deviate by more than n_sigma standard
    deviations from the local mean.
    
    Algorithm:
    1. For each index i, find N closest neighboring indices
       - Interior points: symmetric neighbors (e.g., i-3, i-2, i-1, i+1, i+2, i+3 for N=6)
       - Edge points: asymmetric neighbors (e.g., i-1, i+1, i+2, i+3, i+4, i+5)
    2. Compute mean and std from these neighbors (excluding point i itself)
    3. Flag point i if |value[i] - mean| > n_sigma * std
    
    Special cases:
    - NaN/inf values are flagged as outliers
    - When all neighbors are identical (std=0), flag if value differs
    - Empty arrays or arrays with all NaN return empty result
    
    Args:
        values: 1D array of time series values
        n_neighbors: Number of neighbors to use for statistics (default: 6)
        n_sigma: Number of standard deviations for outlier threshold (default: 5.0)
        
    Returns:
        Array of indices where outliers were detected (empty if none found)
        
    Example:
        >>> data = np.array([1.0, 1.1, 1.0, 10.0, 0.9, 1.1, 1.0])
        >>> outliers = detect_outliers_stdcheck(data, n_neighbors=6, n_sigma=5.0)
        >>> outliers
        array([3])  # Index 3 (value=10.0) is an outlier
    """
    # Input validation
    if len(values) == 0:
        return np.array([], dtype=np.int64)
    
    # Check for all-NaN or all-inf arrays
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        # All values are NaN/inf - return all indices as outliers
        return np.arange(len(values), dtype=np.int64)
    
    if len(values) < n_neighbors + 1:
        # Not enough data points for meaningful outlier detection
        # Flag NaN/inf values only
        return np.where(~finite_mask)[0].astype(np.int64)
    
    outlier_indices = []
    n = len(values)
    
    # Number of neighbors on each side (for symmetric case)
    half_neighbors = n_neighbors // 2
    
    for i in range(n):
        # Flag NaN/inf values immediately
        if not np.isfinite(values[i]):
            outlier_indices.append(i)
            continue
        
        # Determine neighbor indices based on position
        if i < half_neighbors:
            # Near start: take neighbors to the right
            neighbor_start = 0
            neighbor_end = min(n_neighbors + 1, n)
        elif i >= n - half_neighbors:
            # Near end: take neighbors to the left
            neighbor_start = max(0, n - n_neighbors - 1)
            neighbor_end = n
        else:
            # Interior: symmetric neighbors
            neighbor_start = i - half_neighbors
            neighbor_end = i + half_neighbors + 1
        
        # Get neighbor values (excluding the point itself)
        neighbor_indices = list(range(neighbor_start, neighbor_end))
        if i in neighbor_indices:
            neighbor_indices.remove(i)
        
        # Ensure we have exactly n_neighbors (or as many as possible)
        neighbor_indices = neighbor_indices[:n_neighbors]
        
        if len(neighbor_indices) < 2:
            # Need at least 2 neighbors to compute std
            continue
            
        neighbor_values = values[neighbor_indices]
        
        # Filter out NaN/inf from neighbors
        finite_neighbors = neighbor_values[np.isfinite(neighbor_values)]
        if len(finite_neighbors) < 2:
            # Not enough valid neighbors for statistics
            continue
        
        # Compute statistics from valid neighbors
        mean_val = np.mean(finite_neighbors)
        std_val = np.std(finite_neighbors, ddof=1)  # Use sample std
        
        # Check if current value is an outlier
        deviation = abs(values[i] - mean_val)
        if std_val > 0:
            # Normal case: check if deviation exceeds threshold
            if deviation > n_sigma * std_val:
                outlier_indices.append(i)
        else:
            # When std is 0 (all neighbors identical), flag if value differs from mean
            if deviation > 0:
                outlier_indices.append(i)
    
    return np.array(outlier_indices, dtype=np.int64)


def apply_outlier_flags(
    data_list: list,
    field_name: str,
    outlier_indices: np.ndarray
) -> None:
    """Apply outlier flags to a list of dataclass objects.
    
    This is a helper function to efficiently set boolean flags for detected outliers.
    
    Args:
        data_list: List of dataclass objects (IMUReading or GNSSReading)
        field_name: Name of the boolean flag field to set (e.g., 'acc_x_mg_stdchecked')
        outlier_indices: Array of indices where outliers were detected
        
    Example:
        >>> imu_list = [IMUReading(...), IMUReading(...), ...]
        >>> outliers = detect_outliers_stdcheck(acc_x_values)
        >>> apply_outlier_flags(imu_list, 'acc_x_mg_stdchecked', outliers)
    """
    for idx in outlier_indices:
        setattr(data_list[idx], field_name, True)


def compute_pps_regression(
    pps_list: list[PPSFix],
    gnss_list: list[GNSSReading],
    global_min_micros: int | None = None,
) -> tuple[float, float, float] | None:
    """Compute linear regression from PPS micros to UTC timestamps.

    This function synchronizes MCU microsecond timestamps to absolute UTC time
    by establishing a linear mapping between PPS events and GNSS-provided UTC
    timestamps. The regression allows sub-millisecond accuracy for all sensor
    data timestamps.

    Process:
    1. Uses unwrapped micros timestamps for both PPS and GNSS data
    2. For each PPS event, finds the temporally closest GNSS measurement
    3. Uses GNSS UTC time to determine which second boundary the PPS marks
    4. Applies outlier filtering with n_neighbors=4, n_sigma=1.0 (if ≥7 pairs)
    5. Performs linear regression with improved normalization:
       - Subtracts minimum from both micros and UTC timestamps
       - Converts micros offset to seconds
       - Normalizes both quantities to max value of 1.0
       - This improves numerical stability and precision of the regression
    6. Transforms coefficients back to original scale: UTC_time = slope × micros + intercept

    Args:
        pps_list: List of PPS fixes (must have micros_reading_unwrapped populated)
        gnss_list: List of GNSS readings (must have micros_reading_unwrapped populated)
        global_min_micros: Minimum micros value across all data types for numerical
                          stability (defaults to min of pps_list if not provided)

    Returns:
        Tuple of (slope, intercept, r_squared) for the linear regression
        Returns None if insufficient data (empty lists or fewer than 2 PPS entries)
    """
    if not pps_list or not gnss_list:
        logger.warning("Cannot compute PPS regression: empty PPS or GNSS data")
        return None

    if len(pps_list) < 2:
        logger.warning(
            f"Cannot compute PPS regression: need at least 2 PPS entries, "
            f"got {len(pps_list)}"
        )
        return None

    # Get unwrapped micros for both PPS and GNSS
    pps_micros_unwrapped = np.array([
        p.micros_reading_unwrapped if p.micros_reading_unwrapped is not None
        else p.micros_reading
        for p in pps_list
    ], dtype=np.int64)
    gnss_micros_unwrapped = np.array([
        g.micros_reading_unwrapped if g.micros_reading_unwrapped is not None
        else g.micros_reading
        for g in gnss_list
    ], dtype=np.int64)

    # For each PPS entry, find the closest GNSS entry by micros
    # and determine which UTC second boundary the PPS marks
    pps_matched_micros = []
    pps_matched_utc = []

    # Use binary search for efficient closest neighbor finding
    # Sort GNSS micros if not already sorted (should be in chronological order)
    gnss_sorted_indices = np.argsort(gnss_micros_unwrapped)
    gnss_micros_sorted = gnss_micros_unwrapped[gnss_sorted_indices]

    for pps_micros in pps_micros_unwrapped:
        # Find insertion point using binary search
        insert_idx = np.searchsorted(gnss_micros_sorted, pps_micros)
        
        # Check neighbors around insertion point to find closest
        candidates = []
        if insert_idx > 0:
            candidates.append(insert_idx - 1)
        if insert_idx < len(gnss_micros_sorted):
            candidates.append(insert_idx)
        
        # Find the closest candidate
        if not candidates:
            continue
            
        closest_sorted_idx = min(
            candidates,
            key=lambda idx: abs(gnss_micros_sorted[idx] - pps_micros)
        )
        
        # Map back to original GNSS list index
        closest_gnss_idx = gnss_sorted_indices[closest_sorted_idx]

        # Get the UTC timestamp from the matched GNSS entry
        gnss_entry = gnss_list[closest_gnss_idx]
        utc_timestamp = gnss_entry.posix_timestamp + gnss_entry.microseconds / 1e6

        # Determine which second boundary this PPS marks
        # The PPS marks the start of a second. We estimate which second
        # by looking at the UTC time of the closest GNSS and the micros offset
        micros_offset = pps_micros - gnss_micros_unwrapped[closest_gnss_idx]
        estimated_pps_utc = utc_timestamp + micros_offset / 1e6

        # The PPS second is the second boundary closest to the estimated time
        utc_second = round(estimated_pps_utc)

        pps_matched_micros.append(pps_micros)
        pps_matched_utc.append(float(utc_second))

    # Apply outlier filtering to matched pairs before regression
    # Use stricter thresholds (n_neighbors=4, n_sigma=1.0) for high accuracy
    # Only filter if we have enough data points
    pps_matched_micros_array = np.array(pps_matched_micros, dtype=np.float64)
    pps_matched_utc_array = np.array(pps_matched_utc, dtype=np.float64)
    
    n_pairs_before = len(pps_matched_micros_array)
    
    if n_pairs_before >= 7:  # Need at least 7 points for filtering with n_neighbors=4
        # Detect outliers in micros values
        micros_outliers = detect_outliers_stdcheck(
            pps_matched_micros_array, n_neighbors=4, n_sigma=1.0
        )
        
        # Detect outliers in UTC values
        utc_outliers = detect_outliers_stdcheck(
            pps_matched_utc_array, n_neighbors=4, n_sigma=1.0
        )
        
        # Combine outlier indices (union)
        all_outliers = np.unique(np.concatenate([micros_outliers, utc_outliers]))
        
        if len(all_outliers) > 0:
            # Create mask of valid (non-outlier) indices
            valid_mask = np.ones(n_pairs_before, dtype=bool)
            valid_mask[all_outliers] = False
            
            # Filter out outliers
            pps_matched_micros_filtered = pps_matched_micros_array[valid_mask].tolist()
            pps_matched_utc_filtered = pps_matched_utc_array[valid_mask].tolist()
            
            n_removed = len(all_outliers)
            logger.info(
                f"Filtered {n_removed} outlier(s) from PPS-GNSS pairs "
                f"({n_pairs_before} → {len(pps_matched_micros_filtered)} pairs)"
            )
            
            # Check if we still have enough data after filtering
            if len(pps_matched_micros_filtered) < 2:
                logger.warning(
                    f"Too few pairs remaining after outlier filtering "
                    f"({len(pps_matched_micros_filtered)}), using unfiltered data"
                )
                pps_matched_micros = pps_matched_micros
                pps_matched_utc = pps_matched_utc
            else:
                pps_matched_micros = pps_matched_micros_filtered
                pps_matched_utc = pps_matched_utc_filtered
        else:
            # No outliers detected, use original data
            pps_matched_micros = pps_matched_micros_array.tolist()
            pps_matched_utc = pps_matched_utc_array.tolist()
    else:
        # Too few points for meaningful outlier detection, skip filtering
        logger.debug(
            f"Skipping outlier filtering: only {n_pairs_before} pairs "
            f"(need ≥7 for n_neighbors=4)"
        )

    # Perform linear regression with improved normalization
    # To avoid numerical inaccuracies:
    # 1. Subtract minimum from both micros and UTC
    # 2. Convert micros offset to seconds
    # 3. Normalize both to have max value of 1.0
    
    # Use global minimum if provided, otherwise use minimum from PPS data
    if global_min_micros is None:
        min_micros = min(pps_matched_micros)
    else:
        min_micros = global_min_micros
    
    min_utc = min(pps_matched_utc)
    
    # Subtract minimums
    pps_matched_micros_offset = [m - min_micros for m in pps_matched_micros]
    pps_matched_utc_offset = [u - min_utc for u in pps_matched_utc]
    
    # Convert micros to seconds
    pps_matched_micros_offset_sec = [m / 1e6 for m in pps_matched_micros_offset]
    
    # Normalize both to max value of 1.0
    max_micros_sec = max(pps_matched_micros_offset_sec)
    max_utc = max(pps_matched_utc_offset)
    
    # Avoid division by zero (shouldn't happen with valid data)
    if max_micros_sec == 0 or max_utc == 0:
        logger.error("Cannot normalize: max value is zero")
        return None
    
    pps_matched_micros_normalized = [m / max_micros_sec for m in pps_matched_micros_offset_sec]
    pps_matched_utc_normalized = [u / max_utc for u in pps_matched_utc_offset]
    
    # Perform linear regression on normalized data
    slope_norm, intercept_norm, r_value, p_value, std_err = stats.linregress(
        pps_matched_micros_normalized, pps_matched_utc_normalized
    )
    
    # Transform back to original scale
    # y_norm = slope_norm * x_norm + intercept_norm
    # (y - min_utc) / max_utc = slope_norm * ((x - min_micros)/1e6) / max_micros_sec + intercept_norm
    # y = slope_norm * max_utc * (x - min_micros) / (1e6 * max_micros_sec) + intercept_norm * max_utc + min_utc
    # y = slope_final * x + intercept_final
    # where slope_final = slope_norm * max_utc / (1e6 * max_micros_sec)
    #       intercept_final = -slope_final * min_micros + intercept_norm * max_utc + min_utc
    
    slope = slope_norm * max_utc / (1e6 * max_micros_sec)
    intercept = -slope * min_micros + intercept_norm * max_utc + min_utc

    logger.info(f"PPS regression: slope={slope:.12f}, intercept={intercept:.6f}")
    logger.info(f"  R²={r_value**2:.9f}, p-value={p_value:.2e}, std_err={std_err:.2e}")
    logger.info(f"  Used {len(pps_matched_micros)} PPS-GNSS matched pairs")
    logger.info(f"  Normalization: micros range {min_micros} to {min_micros + max_micros_sec*1e6:.0f} µs")
    logger.info(f"  Normalization: UTC range {min_utc:.1f} to {min_utc + max_utc:.1f} s")

    r_squared = r_value ** 2
    return (slope, intercept, r_squared)


def apply_pps_regression(
    pps_list: list[PPSFix],
    gnss_list: list[GNSSReading],
    imu_list: list[IMUReading],
    slope: float,
    intercept: float,
) -> None:
    """Apply PPS regression to all data entries for synchronized UTC timestamps.

    Modifies dataclass objects in-place, adding UTC timestamp fields computed
    from the linear regression: UTC = slope × micros_unwrapped + intercept

    This provides absolute UTC timestamps (both as POSIX floats and timezone-aware
    datetime objects) for all sensor measurements, enabling precise time
    synchronization across PPS, GNSS, and IMU data streams.

    Args:
        pps_list: List of PPS fixes to update
        gnss_list: List of GNSS readings to update
        imu_list: List of IMU readings to update
        slope: Regression slope (microseconds to seconds conversion factor)
        intercept: Regression intercept (seconds)

    Note:
        Uses unwrapped micros_reading values to handle uint32_t overflow correctly.
        Falls back to raw micros_reading if unwrapped value is None.
    """
    # Apply to PPS using unwrapped values
    if pps_list:
        for pps in pps_list:
            micros_unwrapped = pps.micros_reading_unwrapped
            if micros_unwrapped is None:
                micros_unwrapped = pps.micros_reading
            pps.utc_timestamp_from_pps_regression = (
                slope * micros_unwrapped + intercept
            )
            pps.datetime_timestamp_from_pps_regression = datetime.fromtimestamp(
                pps.utc_timestamp_from_pps_regression, tz=timezone.utc
            )

    # Apply to GNSS using unwrapped values
    if gnss_list:
        for gnss in gnss_list:
            micros_unwrapped = gnss.micros_reading_unwrapped
            if micros_unwrapped is None:
                micros_unwrapped = gnss.micros_reading
            gnss.utc_timestamp_from_pps_regression = (
                slope * micros_unwrapped + intercept
            )
            gnss.datetime_timestamp_from_pps_regression = datetime.fromtimestamp(
                gnss.utc_timestamp_from_pps_regression, tz=timezone.utc
            )

    # Apply to IMU using unwrapped values
    if imu_list:
        for imu in imu_list:
            micros_unwrapped = imu.micros_reading_unwrapped
            if micros_unwrapped is None:
                micros_unwrapped = imu.micros_reading
            imu.utc_timestamp_from_pps_regression = (
                slope * micros_unwrapped + intercept
            )
            imu.datetime_timestamp_from_pps_regression = datetime.fromtimestamp(
                imu.utc_timestamp_from_pps_regression, tz=timezone.utc
            )


def parse_pps_entry(data: bytes) -> PPSFix:
    """Parse a single PPS entry.

    Args:
        data: Raw binary data for PPS entry

    Returns:
        PPSFix object

    Raises:
        AssertionError: If data size is incorrect
    """
    assert len(data) == PPS_STRUCT_SIZE, (
        f"PPS data size mismatch: expected exactly {PPS_STRUCT_SIZE} bytes, "
        f"got {len(data)} bytes"
    )
    micros_reading = struct.unpack("<I", data[:4])[0]
    return PPSFix(micros_reading=micros_reading)


def parse_gnss_entry(data: bytes) -> GNSSReading:
    """Parse a single GNSS entry.

    Two on-disk sizes are accepted:
      - GPS_STRUCT_SIZE (36): legacy layout, no altitude.
      - GPS_STRUCT_SIZE_WITH_ALTITUDE (40): adds int32 altitude_msl_mm
        (mm above mean sea level) right before fix_type.
    Selection is by len(data); the upstream segmenter picks the size from
    the header's "GNSS includes altitude_msl" marker.

    Args:
        data: Raw binary data for GNSS entry

    Returns:
        GNSSReading object with raw and physical unit values

    Raises:
        AssertionError: If data size is incorrect
    """
    assert len(data) in (GPS_STRUCT_SIZE, GPS_STRUCT_SIZE_WITH_ALTITUDE), (
        f"GNSS data size mismatch: expected {GPS_STRUCT_SIZE} (no altitude) "
        f"or {GPS_STRUCT_SIZE_WITH_ALTITUDE} (with altitude), got {len(data)} bytes"
    )

    has_altitude = len(data) == GPS_STRUCT_SIZE_WITH_ALTITUDE
    if has_altitude:
        # 37 data bytes (9 int32 + 1 uint8) + 3 padding
        values = struct.unpack("<IiiiIiiiiB", data[:37])
        altitude_msl_mm = values[8]
        fix_type = values[9]
    else:
        # 33 data bytes + 3 padding
        values = struct.unpack("<IiiiIiiiB", data[:33])
        altitude_msl_mm = 0
        fix_type = values[8]

    micros_reading = values[0]
    latitude = values[1]
    longitude = values[2]
    posix_timestamp = values[3]
    microseconds = values[4]
    ned_vel_north = values[5]
    ned_vel_east = values[6]
    ned_vel_down = values[7]

    # Convert to physical units
    latitude_dd = latitude / 1e7
    longitude_dd = longitude / 1e7
    ned_vel_north_mmps = ned_vel_north
    ned_vel_east_mmps = ned_vel_east
    ned_vel_down_mmps = ned_vel_down
    altitude_msl_m = altitude_msl_mm / 1000.0

    # Create datetime with microsecond accuracy
    datetime_utc = datetime.fromtimestamp(
        posix_timestamp + microseconds / 1e6, tz=timezone.utc
    )

    return GNSSReading(
        micros_reading=micros_reading,
        latitude=latitude,
        longitude=longitude,
        posix_timestamp=posix_timestamp,
        microseconds=microseconds,
        ned_vel_north=ned_vel_north,
        ned_vel_east=ned_vel_east,
        ned_vel_down=ned_vel_down,
        fix_type=fix_type,
        latitude_dd=latitude_dd,
        longitude_dd=longitude_dd,
        ned_vel_north_mmps=ned_vel_north_mmps,
        ned_vel_east_mmps=ned_vel_east_mmps,
        ned_vel_down_mmps=ned_vel_down_mmps,
        datetime_utc=datetime_utc,
        altitude_msl_mm=altitude_msl_mm,
        altitude_msl_m=altitude_msl_m,
    )


def parse_imu_entry(
    data: bytes,
    acc_sensitivity: float = 0.061,
    gyr_sensitivity: float = 4.375,
    mag_sensitivity: float | None = None,
) -> IMUReading:
    """Parse a single IMU entry.

    Args:
        data: Raw binary data for IMU entry. Length must be either
              IMU_STRUCT_SIZE (18, no magnetometer) or
              IMU_STRUCT_SIZE_WITH_MAG (24, with magnetometer).
        acc_sensitivity: Accelerometer sensitivity in mg/LSB
        gyr_sensitivity: Gyroscope sensitivity in mdps/LSB
        mag_sensitivity: Magnetometer sensitivity in uT/LSB. Required if
                         `data` is 24 bytes (with-mag format); ignored if
                         `data` is 18 bytes (legacy no-mag format).

    Returns:
        IMUReading object with raw and scaled values. mag_* fields stay at
        their default 0 when the legacy 18-byte struct is parsed.

    Raises:
        AssertionError: If data size is neither 18 nor 24 bytes.
    """
    assert len(data) in (IMU_STRUCT_SIZE, IMU_STRUCT_SIZE_WITH_MAG), (
        f"IMU data size mismatch: expected {IMU_STRUCT_SIZE} (no mag) or "
        f"{IMU_STRUCT_SIZE_WITH_MAG} (with mag) bytes, got {len(data)} bytes"
    )

    if len(data) == IMU_STRUCT_SIZE:
        # Legacy 18-byte struct: micros, counter, 3x acc, 3x gyr
        values = struct.unpack("<IHhhhhhh", data)
        micros_reading = values[0]
        counter = values[1]
        acc_x, acc_y, acc_z = values[2], values[3], values[4]
        gyr_x, gyr_y, gyr_z = values[5], values[6], values[7]
        mag_x = mag_y = mag_z = 0
    else:
        # New 24-byte struct: micros, counter, 3x acc, 3x gyr, 3x mag
        values = struct.unpack("<IHhhhhhhhhh", data)
        micros_reading = values[0]
        counter = values[1]
        acc_x, acc_y, acc_z = values[2], values[3], values[4]
        gyr_x, gyr_y, gyr_z = values[5], values[6], values[7]
        mag_x, mag_y, mag_z = values[8], values[9], values[10]

    acc_x_mg = acc_x * acc_sensitivity
    acc_y_mg = acc_y * acc_sensitivity
    acc_z_mg = acc_z * acc_sensitivity
    gyr_x_mdps = gyr_x * gyr_sensitivity
    gyr_y_mdps = gyr_y * gyr_sensitivity
    gyr_z_mdps = gyr_z * gyr_sensitivity

    if mag_sensitivity is None:
        mag_x_uT = mag_y_uT = mag_z_uT = 0.0
    else:
        mag_x_uT = mag_x * mag_sensitivity
        mag_y_uT = mag_y * mag_sensitivity
        mag_z_uT = mag_z * mag_sensitivity

    return IMUReading(
        micros_reading=micros_reading,
        counter=counter,
        acc_x=acc_x,
        acc_y=acc_y,
        acc_z=acc_z,
        gyr_x=gyr_x,
        gyr_y=gyr_y,
        gyr_z=gyr_z,
        acc_x_mg=acc_x_mg,
        acc_y_mg=acc_y_mg,
        acc_z_mg=acc_z_mg,
        gyr_x_mdps=gyr_x_mdps,
        gyr_y_mdps=gyr_y_mdps,
        gyr_z_mdps=gyr_z_mdps,
        mag_x=mag_x,
        mag_y=mag_y,
        mag_z=mag_z,
        mag_x_uT=mag_x_uT,
        mag_y_uT=mag_y_uT,
        mag_z_uT=mag_z_uT,
    )


def check_micros_consistency(
    pps_list: list[PPSFix],
    gnss_list: list[GNSSReading],
    imu_list: list[IMUReading],
    segment_idx: int,
    max_deviation_seconds: float = 10.0
) -> None:
    """Check that min/max micros timestamps are consistent across PPS, GNSS, and IMU.
    
    After unwrapping, the minimum and maximum micros_reading_unwrapped values
    across all three data types should be within a reasonable time range
    (default 10 seconds). This sanity check catches issues like:
    - Incorrect unwrapping
    - Mixed data from different time periods
    - Corruption causing timestamp discontinuities
    
    Args:
        pps_list: List of PPS fixes with micros_reading_unwrapped populated
        gnss_list: List of GNSS readings with micros_reading_unwrapped populated
        imu_list: List of IMU readings with micros_reading_unwrapped populated
        segment_idx: Segment index for logging
        max_deviation_seconds: Maximum allowed deviation in seconds (default: 10.0)
    
    Raises:
        ValueError: If deviation exceeds threshold
    """
    max_deviation_micros = max_deviation_seconds * 1_000_000  # Convert to microseconds
    
    # Collect min/max for each data type
    data_types_info = []
    
    if pps_list:
        pps_micros = [p.micros_reading_unwrapped for p in pps_list if p.micros_reading_unwrapped is not None]
        if pps_micros:
            data_types_info.append(("PPS", min(pps_micros), max(pps_micros), len(pps_micros)))
    
    if gnss_list:
        gnss_micros = [g.micros_reading_unwrapped for g in gnss_list if g.micros_reading_unwrapped is not None]
        if gnss_micros:
            data_types_info.append(("GNSS", min(gnss_micros), max(gnss_micros), len(gnss_micros)))
    
    if imu_list:
        imu_micros = [i.micros_reading_unwrapped for i in imu_list if i.micros_reading_unwrapped is not None]
        if imu_micros:
            data_types_info.append(("IMU", min(imu_micros), max(imu_micros), len(imu_micros)))
    
    # Need at least 2 data types to compare
    if len(data_types_info) < 2:
        logger.debug(f"Segment {segment_idx}: Only {len(data_types_info)} data type(s) present, skipping consistency check")
        return
    
    # Extract all mins and maxes
    all_mins = [info[1] for info in data_types_info]
    all_maxes = [info[2] for info in data_types_info]
    
    # Compute deviations
    min_of_mins = min(all_mins)
    max_of_mins = max(all_mins)
    min_of_maxes = min(all_maxes)
    max_of_maxes = max(all_maxes)
    
    deviation_in_mins = max_of_mins - min_of_mins
    deviation_in_maxes = max_of_maxes - min_of_maxes
    
    # Check deviations
    error_messages = []
    
    if deviation_in_mins > max_deviation_micros:
        deviation_seconds = deviation_in_mins / 1_000_000
        error_messages.append(
            f"Min timestamp deviation: {deviation_seconds:.3f}s (> {max_deviation_seconds}s threshold)"
        )
        for name, min_val, max_val, count in data_types_info:
            error_messages.append(f"  {name:5s}: min={min_val:15d} µs  (n={count})")
    
    if deviation_in_maxes > max_deviation_micros:
        deviation_seconds = deviation_in_maxes / 1_000_000
        error_messages.append(
            f"Max timestamp deviation: {deviation_seconds:.3f}s (> {max_deviation_seconds}s threshold)"
        )
        for name, min_val, max_val, count in data_types_info:
            error_messages.append(f"  {name:5s}: max={max_val:15d} µs  (n={count})")
    
    if error_messages:
        logger.error(f"❌ Segment {segment_idx}: Micros timestamp consistency check FAILED")
        for msg in error_messages:
            logger.error(f"   {msg}")
        raise ValueError(
            f"Segment {segment_idx} failed micros consistency check: "
            f"min deviation={deviation_in_mins/1_000_000:.3f}s, "
            f"max deviation={deviation_in_maxes/1_000_000:.3f}s"
        )
    
    # Log success at debug level
    logger.debug(
        f"Segment {segment_idx}: Micros consistency check passed "
        f"(min_dev={deviation_in_mins/1_000_000:.3f}s, max_dev={deviation_in_maxes/1_000_000:.3f}s)"
    )


def compute_pps_mismatch_statistics(
    pps_list: list[PPSFix],
    show_plot: bool = False
) -> float | None:
    """Compute and display PPS mismatch statistics to assess regression quality.
    
    Evaluates how well the linear regression aligns PPS events to exact UTC
    second boundaries. Each PPS pulse should occur at the start of a UTC second
    (e.g., 12:34:56.000). This function computes the deviation between the
    regression-predicted timestamp and the nearest second boundary.
    
    Displays:
    - Maximum absolute mismatch (ms)
    - Mean mismatch (ms) - should be near zero for unbiased regression
    - RMS mismatch (ms) - overall accuracy metric
    - Optional ASCII plot of mismatch vs time (requires gnuplotlib)
    
    Args:
        pps_list: List of PPS fixes with utc_timestamp_from_pps_regression populated
        show_plot: If True, display ASCII terminal plot using gnuplotlib
                  (silently skips if gnuplotlib not available)
    
    Returns:
        Maximum absolute mismatch in seconds, or None if unable to compute
    
    Note:
        Typical good results: max < 5ms, RMS < 2ms for R² > 0.999999
    """
    if not pps_list:
        logger.warning("No PPS data available for mismatch analysis")
        return None

    # Check if regression was computed
    if pps_list[0].utc_timestamp_from_pps_regression is None:
        logger.warning("PPS regression not computed, skipping mismatch analysis")
        return None

    # Compute mismatch for each PPS entry
    mismatches = []
    for pps in pps_list:
        utc_timestamp = pps.utc_timestamp_from_pps_regression
        closest_second = round(utc_timestamp)
        mismatch = utc_timestamp - closest_second
        mismatches.append(mismatch)

    mismatches_array = np.array(mismatches)

    # Compute statistics
    max_mismatch = np.max(np.abs(mismatches_array))
    mean_mismatch = np.mean(mismatches_array)
    rms_mismatch = np.sqrt(np.mean(mismatches_array ** 2))

    # Log statistics
    logger.info("")
    logger.info("PPS Mismatch Statistics (UTC regression vs closest second):")
    logger.info(f"  Max absolute mismatch: {max_mismatch * 1000:.3f} ms")
    logger.info(f"  Mean mismatch:         {mean_mismatch * 1000:.3f} ms")
    logger.info(f"  RMS mismatch:          {rms_mismatch * 1000:.3f} ms")
    
    return max_mismatch

    if show_plot:
        if not GNUPLOT_AVAILABLE:
            logger.error(
                "Cannot display plots: gnuplotlib is not available. "
                "Install with: pip install gnuplotlib"
            )
            return
        # Prepare data for plotting
        # X-axis: time since first PPS (in seconds)
        first_pps_utc = pps_list[0].utc_timestamp_from_pps_regression
        x_data = np.array([
            pps.utc_timestamp_from_pps_regression - first_pps_utc
            for pps in pps_list
        ])

        # Y-axis: mismatch in milliseconds
        y_data = mismatches_array * 1000

        # Plot using gnuplotlib
        import sys
        # Temporarily redirect stderr to stdout to ensure plot appears correctly
        old_stderr = sys.stderr
        sys.stderr = sys.stdout

        print("")  # Print blank line directly to stdout
        print("PPS Mismatch vs Time Plot:")
        sys.stdout.flush()
        gp.plot(
            x_data, y_data,
            _with='lines',
            terminal='dumb 80,24',
            unset='grid',
            xlabel='Time since first PPS (seconds)',
            ylabel='UTC mismatch (ms)',
            title='PPS UTC Mismatch (ms): Regression vs Closest Second'
        )
        print("")  # Print blank line after plot

        # Restore stderr
        sys.stderr = old_stderr


def print_summary_statistics(
    pps_list: list[PPSFix],
    gnss_list: list[GNSSReading],
    imu_list: list[IMUReading],
    unwrap_stats: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Print summary statistics about the parsed data.

    Displays:
    - Number of messages parsed for each data type
    - Unwrap and jump statistics (wraps and anomalous jumps detected)
    - File duration based on IMU timestamps (excluding anomalous entries)
    - Effective sampling rates for each data type

    Args:
        pps_list: List of parsed PPS entries
        gnss_list: List of parsed GNSS entries
        imu_list: List of parsed IMU entries
        unwrap_stats: Optional dictionary with unwrap/jump statistics containing
                     wrap and jump counts for each field of each data type
    """
    logger.info("=" * 60)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 60)

    logger.info("Number of messages parsed:")
    logger.info(f"  PPS:  {len(pps_list):6d}")
    logger.info(f"  GNSS: {len(gnss_list):6d}")
    logger.info(f"  IMU:  {len(imu_list):6d}")

    # Print unwrap statistics if available
    if unwrap_stats:
        logger.info("")
        logger.info("Unwrap and jump statistics:")
        for data_type, field_stats in unwrap_stats.items():
            logger.info(f"  {data_type}:")
            for field, counts in field_stats.items():
                logger.info(f"    {field}:")
                logger.info(f"      Wraps: {counts['wraps']}")
                logger.info(f"      Jumps: {counts['jumps']}")

    if len(imu_list) >= 2:
        # Find first and last non-jumped entries for duration calculation
        first_idx = 0
        last_idx = len(imu_list) - 1

        # Get jump indices if available
        jump_indices = set()
        if unwrap_stats and "IMU" in unwrap_stats:
            if "micros_reading" in unwrap_stats["IMU"]:
                jumps = unwrap_stats["IMU"]["micros_reading"].get("jump_indices")
                if jumps is not None:
                    jump_indices = set(jumps)

        # Walk backwards from end to find last non-jumped entry
        while last_idx > first_idx and last_idx in jump_indices:
            last_idx -= 1

        # Walk forward from start to find first non-jumped entry
        while first_idx < last_idx and first_idx in jump_indices:
            first_idx += 1

        if first_idx < last_idx:
            # Use unwrapped micros if available, otherwise use raw
            if imu_list[first_idx].micros_reading_unwrapped is not None:
                first_micros = imu_list[first_idx].micros_reading_unwrapped
                last_micros = imu_list[last_idx].micros_reading_unwrapped
            else:
                first_micros = imu_list[first_idx].micros_reading
                last_micros = imu_list[last_idx].micros_reading

            duration_us = last_micros - first_micros
            
            # Handle counter wraps (if duration is negative, assume wrap occurred)
            if duration_us < 0:
                duration_us += 2**32  # uint32_t wrap
            
            duration_s = duration_us / 1e6
            duration_min = duration_s / 60.0

            logger.info("")
            logger.info("File duration (from IMU timestamps, excluding jumps):")
            logger.info(f"  First micros: {first_micros}")
            logger.info(f"  Last micros:  {last_micros}")
            logger.info(
                f"  Duration:     {duration_s:.2f} seconds "
                f"({duration_min:.2f} min)"
            )

            if duration_s > 0:
                logger.info("")
                logger.info("Effective sampling rates:")
                if len(pps_list) > 0:
                    pps_rate = len(pps_list) / duration_s
                    logger.info(f"  PPS:  {pps_rate:.3f} Hz")
                if len(gnss_list) > 0:
                    gnss_rate = len(gnss_list) / duration_s
                    logger.info(f"  GNSS: {gnss_rate:.3f} Hz")
                if len(imu_list) > 0:
                    imu_rate = len(imu_list) / duration_s
                    logger.info(f"  IMU:  {imu_rate:.3f} Hz")
        else:
            logger.warning("All IMU entries have jumps, cannot compute duration")
    else:
        logger.warning("Not enough IMU data to compute duration")

    logger.info("=" * 60)


CORRUPTION_SCAN_BYTES = 1024


def scan_for_next_valid_marker(
    content: bytes,
    start_idx: int,
    valid_markers: tuple[bytes, ...],
    scan_bytes: int = CORRUPTION_SCAN_BYTES,
) -> int | None:
    """Scan ahead in content for the next valid data entry marker.

    Used for corruption recovery: when an unexpected byte is encountered,
    this function searches forward to find the next valid entry marker
    to resume parsing.

    Args:
        content: Full file content as bytes
        start_idx: Position to start scanning from
        valid_markers: Tuple of valid marker bytes to search for (e.g., PPS, GPS, IMU markers)
        scan_bytes: Maximum number of bytes to scan (default: 1024)

    Returns:
        Index of the next valid marker if found, None otherwise
    """
    end_idx = min(start_idx + scan_bytes, len(content))
    scan_region = content[start_idx:end_idx]

    # Find earliest occurrence of any valid marker
    earliest_pos = None
    earliest_offset = float('inf')

    for marker in valid_markers:
        pos = scan_region.find(marker)
        if pos != -1 and pos < earliest_offset:
            earliest_offset = pos
            earliest_pos = start_idx + pos

    return earliest_pos


def handle_junk_bytes(
    content: bytes,
    idx: int,
    next_byte: int,
    junk_start: int,
    entry_type: str,
    markers: tuple[bytes, ...],
    footer_marker: bytes,
    pps_list: list,
    gnss_list: list,
    imu_list: list,
) -> tuple[int, bool]:
    """Handle junk bytes after an entry.
    
    Args:
        content: Full file content
        idx: Current index after entry
        next_byte: The unexpected byte found
        junk_start: Offset where junk started
        entry_type: Type of entry ("PPS", "GNSS", "IMU")
        markers: Valid entry markers for recovery
        footer_marker: Footer marker bytes
        pps_list, gnss_list, imu_list: Current parsed entries
        
    Returns:
        Tuple of (new_idx, should_break)
        - new_idx: Updated index position
        - should_break: Whether to break from parsing loop
    """
    junk_bytes = []
    
    # Skip junk bytes until we find a valid marker
    while idx < len(content):
        b = content[idx]
        if b == ord(b'\n') or footer_marker in content[idx:idx+20]:
            # Found valid marker, stop skipping
            break
        junk_bytes.append(b)
        idx += 1
        
        # Safety: don't skip more than a reasonable amount
        if len(junk_bytes) >= CORRUPTION_SCAN_BYTES:
            # Too much junk - treat as serious corruption
            logger.warning(
                f"Unexpected byte 0x{next_byte:02x} at offset {junk_start} after {entry_type} entry "
                f"(>{CORRUPTION_SCAN_BYTES} junk bytes)"
            )
            logger.error("Scanning ahead for valid marker...")
            
            # Scan ahead for next valid entry
            next_marker_idx = scan_for_next_valid_marker(content, idx, markers)
            
            if next_marker_idx is not None:
                bytes_skipped = next_marker_idx - junk_start
                logger.info(
                    f"Recovered at offset {next_marker_idx} ({bytes_skipped} bytes skipped)"
                )
                return next_marker_idx, False
            else:
                logger.error(
                    f"Recovery failed. Parsed {len(pps_list)} PPS, "
                    f"{len(gnss_list)} GNSS, {len(imu_list)} IMU before corruption"
                )
                return idx, True
    
    # Successfully skipped small amount of junk
    if 0 < len(junk_bytes) < CORRUPTION_SCAN_BYTES:
        logger.warning(
            f"Skipped {len(junk_bytes)} junk byte(s) at offset {junk_start} after {entry_type} "
            f"(first: 0x{junk_bytes[0]:02x})"
        )
    
    return idx, False


def process_pps_entry(
    content: bytes,
    idx: int,
    pps_list: list,
    gnss_list: list,
    imu_list: list,
    pps_marker: bytes,
    gps_marker: bytes,
    imu_marker: bytes,
    footer_marker: bytes,
    pps_struct_size: int,
) -> tuple[int, bool]:
    """Process a single PPS entry.
    
    Returns:
        Tuple of (new_idx, should_break)
    """
    # Check we have enough bytes
    line_end = idx + PPS_LINE_SIZE
    if line_end > len(content):
        logger.warning(
            f"Incomplete PPS entry at offset {idx}: "
            f"need {PPS_LINE_SIZE} bytes, only {len(content) - idx} available"
        )
        logger.error(
            f"File truncated. Parsed {len(pps_list)} PPS, "
            f"{len(gnss_list)} GNSS, {len(imu_list)} IMU entries before truncation"
        )
        return idx, True
    
    # Parse entry
    idx += 4
    pps_data = content[idx : idx + pps_struct_size]
    try:
        pps_entry = parse_pps_entry(pps_data)
        pps_list.append(pps_entry)
    except (struct.error, AssertionError) as e:
        logger.warning(f"Failed to parse PPS entry at offset {idx}: {e}")
        logger.error(f"Parsing aborted (data length={len(pps_data)}, expected={pps_struct_size})")
        raise
    idx += pps_struct_size
    
    # Check next byte is valid
    if idx < len(content):
        next_byte = content[idx]
        if next_byte == ord(b'\n') or footer_marker in content[idx:idx+20]:
            return idx, False
        else:
            # Handle junk bytes
            new_idx, should_break = handle_junk_bytes(
                content, idx, next_byte, idx, "PPS",
                (pps_marker, gps_marker, imu_marker, footer_marker),
                footer_marker, pps_list, gnss_list, imu_list
            )
            return new_idx, should_break
    
    return idx, False


def process_gnss_entry(
    content: bytes,
    idx: int,
    pps_list: list,
    gnss_list: list,
    imu_list: list,
    pps_marker: bytes,
    gps_marker: bytes,
    imu_marker: bytes,
    footer_marker: bytes,
    gps_struct_size: int,
) -> tuple[int, bool]:
    """Process a single GNSS entry.
    
    Returns:
        Tuple of (new_idx, should_break)
    """
    # Check we have enough bytes. gps_struct_size is variable (36 or 40).
    gps_line_size = MARKER_SIZE + gps_struct_size
    line_end = idx + gps_line_size
    if line_end > len(content):
        logger.warning(
            f"Incomplete GNSS entry at offset {idx}: "
            f"need {gps_line_size} bytes, only {len(content) - idx} available"
        )
        logger.error(
            f"File truncated. Parsed {len(pps_list)} PPS, "
            f"{len(gnss_list)} GNSS, {len(imu_list)} IMU entries before truncation"
        )
        return idx, True
    
    # Parse entry
    idx += 4
    gnss_data = content[idx : idx + gps_struct_size]
    try:
        gnss_entry = parse_gnss_entry(gnss_data)
        gnss_list.append(gnss_entry)
    except (struct.error, AssertionError) as e:
        logger.warning(f"Failed to parse GNSS entry at offset {idx}: {e}")
        logger.error(f"Parsing aborted (data length={len(gnss_data)}, expected={gps_struct_size})")
        raise
    idx += gps_struct_size
    
    # Check next byte is valid
    if idx < len(content):
        next_byte = content[idx]
        if next_byte == ord(b'\n') or footer_marker in content[idx:idx+20]:
            return idx, False
        else:
            # Handle junk bytes
            new_idx, should_break = handle_junk_bytes(
                content, idx, next_byte, idx, "GNSS",
                (pps_marker, gps_marker, imu_marker, footer_marker),
                footer_marker, pps_list, gnss_list, imu_list
            )
            return new_idx, should_break
    
    return idx, False


def process_imu_entry(
    content: bytes,
    idx: int,
    pps_list: list,
    gnss_list: list,
    imu_list: list,
    pps_marker: bytes,
    gps_marker: bytes,
    imu_marker: bytes,
    footer_marker: bytes,
    imu_struct_size: int,
    acc_sensitivity: float,
    gyr_sensitivity: float,
    prev_imu_micros: int | None = None,
    mag_sensitivity: float | None = None,
    prev_imu_counter: int | None = None,
    clip_at_counter_discontinuity: bool = True,
) -> tuple[int, bool, bool]:
    """Process a single IMU entry.

    Args:
        imu_struct_size: 18 for legacy (no mag) records, 24 for records that
                         include magnetometer. The on-disk record size is
                         MARKER_SIZE + imu_struct_size + (2 padding if 18, 0 if 24).
        prev_imu_micros: Previous IMU micros value for jump detection (None if first in segment)
        mag_sensitivity: Mag sensitivity in uT/LSB if mag is logged (struct=24), else None.
        prev_imu_counter: Previous IMU sample counter for end-of-data detection
                          (None if first in segment).
        clip_at_counter_discontinuity: If True (default), a break in the +1 counter
                          sequence is treated as end-of-real-data: the bad entry is
                          NOT appended, idx is NOT advanced, and should_break=True is
                          returned so the caller can stop parsing. Pre-existing data
                          on a non-zero-formatted SD card after the firmware stops
                          mid-write looks like valid IMU records on resync, and this
                          is the only signal that reliably tells them apart.

    Returns:
        Tuple of (new_idx, should_break, jump_detected)
        - jump_detected: True if a micros jump was detected that should trigger segmentation
    """
    # Derive on-disk line size from the struct size. The legacy 18-byte struct
    # has 2 bytes of C alignment padding; the 24-byte mag-included struct is
    # already 4-aligned so no padding.
    imu_padding = IMU_PADDING if imu_struct_size == IMU_STRUCT_SIZE else 0
    imu_line_size = MARKER_SIZE + imu_struct_size + imu_padding

    # Check we have enough bytes
    line_end = idx + imu_line_size
    if line_end > len(content):
        logger.warning(
            f"Incomplete IMU entry at offset {idx}: "
            f"need {imu_line_size} bytes, only {len(content) - idx} available"
        )
        logger.error(
            f"File truncated. Parsed {len(pps_list)} PPS, "
            f"{len(gnss_list)} GNSS, {len(imu_list)} IMU entries before truncation"
        )
        return idx, True, False

    # Parse entry
    marker_idx = idx
    idx += 4
    imu_data = content[idx : idx + imu_struct_size]
    try:
        imu_entry = parse_imu_entry(imu_data, acc_sensitivity, gyr_sensitivity, mag_sensitivity)

        # Counter-continuity check — clip end-of-real-data.
        # The firmware stamps every IMU sample with counter = imu_isr_count++ in the ISR,
        # so consecutive on-disk entries must satisfy counter == (prev + 1) & 0xFFFF.
        # FIFO overflows at the chip level do not produce skips here (the firmware
        # only sees and counts samples it actually pops), so a discontinuity means
        # the firmware stopped writing (e.g. user unplugged the logger) and the
        # bytes we're now reading are stale data left on the SD card from a previous
        # session that wasn't zero-formatted.
        if (
            clip_at_counter_discontinuity
            and prev_imu_counter is not None
            and imu_entry.counter != ((prev_imu_counter + 1) & 0xFFFF)
        ):
            expected = (prev_imu_counter + 1) & 0xFFFF
            logger.warning(
                f"IMU counter discontinuity at offset {marker_idx}: "
                f"expected {expected}, got {imu_entry.counter}. "
                f"Treating as end-of-real-data; stopping parse."
            )
            return marker_idx, True, False

        # Check for micros jump if we have previous value
        jump_detected = False
        if prev_imu_micros is not None:
            micros_diff = imu_entry.micros_reading - prev_imu_micros
            
            # Check for negative jump (backwards in time)
            if micros_diff < 0:
                logger.warning(
                    f"IMU micros negative jump detected at offset {idx}: "
                    f"{prev_imu_micros} → {imu_entry.micros_reading} "
                    f"(diff={micros_diff} µs). Starting new segment."
                )
                jump_detected = True
            # Check for large positive jump (> 1 second = 1,000,000 µs)
            elif micros_diff > 1_000_000:
                logger.warning(
                    f"IMU micros large jump detected at offset {idx}: "
                    f"{prev_imu_micros} → {imu_entry.micros_reading} "
                    f"(diff={micros_diff} µs = {micros_diff/1e6:.3f}s). Starting new segment."
                )
                jump_detected = True
        
        imu_list.append(imu_entry)
    except (struct.error, AssertionError) as e:
        logger.warning(f"Failed to parse IMU entry at offset {idx}: {e}")
        logger.error(f"Parsing aborted (data length={len(imu_data)}, expected={imu_struct_size})")
        raise
    idx += imu_struct_size

    # Skip padding bytes (2 for legacy 18-byte struct, 0 for 24-byte mag struct).
    idx += imu_padding

    # Check next byte is valid
    if idx < len(content):
        next_byte = content[idx]
        if next_byte == ord(b'\n') or footer_marker in content[idx:idx+20]:
            return idx, False, jump_detected
        else:
            # Handle junk bytes
            new_idx, should_break = handle_junk_bytes(
                content, idx, next_byte, idx, "IMU",
                (pps_marker, gps_marker, imu_marker, footer_marker),
                footer_marker, pps_list, gnss_list, imu_list
            )
            return new_idx, should_break, jump_detected
    
    return idx, False, jump_detected


def parse_binary_content(
    content: bytes,
    header_info: dict,
    pps_marker: bytes,
    gps_marker: bytes,
    imu_marker: bytes,
    footer_marker: bytes,
    clip_at_counter_discontinuity: bool = True,
) -> list[dict[str, list]]:
    """Parse binary content and extract all PPS, GNSS, and IMU entries in segments.

    Segments are created based on two conditions:
    1. Time-based: once n_imus_per_segment IMU entries are reached (~1 minute)
    2. Jump-based: when IMU micros has a negative jump or positive jump > 1 second

    Args:
        content: Full file content as bytes
        header_info: Parsed header information
        pps_marker, gps_marker, imu_marker, footer_marker: Entry markers
        clip_at_counter_discontinuity: If True (default), parsing stops at the first
            IMU sample whose counter is not (prev + 1) & 0xFFFF. This is the
            recommended setting; see process_imu_entry() for the rationale.

    Returns:
        List of segment dicts, each containing {'pps_list': [], 'gnss_list': [], 'imu_list': []}
    """
    acc_sensitivity = header_info.get("acc_sensitivity", 0.061)
    gyr_sensitivity = header_info.get("gyr_sensitivity", 4.375)
    mag_sensitivity = header_info.get("mag_sensitivity")  # None when header has no mag line
    imu_odr = header_info.get("imu_odr", 6667.0)

    # Calculate segment size: 1 minute of IMU samples
    n_imus_per_segment = round(imu_odr * 60)
    logger.info(f"Segment size: {n_imus_per_segment} IMU samples (≈1 minute at {imu_odr} Hz)")
    logger.info("Additional segmentation on IMU micros jumps: negative or > 1 second")

    pps_struct_size = PPS_STRUCT_SIZE
    # Pick GNSS struct size based on whether the firmware logged altitude.
    # Detection key: "GNSS includes altitude_msl" header line → 40-byte struct; else legacy 36.
    if header_info.get("gnss_has_altitude", False):
        gps_struct_size = GPS_STRUCT_SIZE_WITH_ALTITUDE
        logger.info("Altitude present in GNSS header — using 40-byte GNSS struct")
    else:
        gps_struct_size = GPS_STRUCT_SIZE
        logger.info("No altitude in GNSS header — using legacy 36-byte GNSS struct")
    # Pick IMU struct size based on whether the firmware logged the magnetometer.
    # Detection key: "Mag sensitivity" header line → 24-byte struct; else legacy 18.
    if mag_sensitivity is not None:
        imu_struct_size = IMU_STRUCT_SIZE_WITH_MAG
        logger.info(
            f"Magnetometer present in header (sensitivity={mag_sensitivity} uT/LSB) — "
            f"using 24-byte IMU struct"
        )
    else:
        imu_struct_size = IMU_STRUCT_SIZE
        logger.info("No magnetometer in header — using legacy 18-byte IMU struct")
    
    # Initialize first segment
    segments = []
    current_segment = {'pps_list': [], 'gnss_list': [], 'imu_list': []}
    segments.append(current_segment)
    segment_imu_count = 0
    prev_imu_micros = None  # Track previous IMU micros for jump detection
    prev_imu_counter = None  # Track previous IMU counter for end-of-data clipping

    counter_clip_idx: int | None = None  # Set when parsing stops on counter discontinuity
    counter_clip_samples: int = 0

    start_offset = 0
    idx = 0
    while idx < len(content):
        # Check if we need to start a new segment based on time
        if segment_imu_count >= n_imus_per_segment:
            logger.info(
                f"Starting segment {len(segments)} at byte {idx} "
                f"(segment {len(segments)-1} had {segment_imu_count} IMUs, "
                f"{len(current_segment['pps_list'])} PPS, {len(current_segment['gnss_list'])} GNSS) - TIME THRESHOLD"
            )
            current_segment = {'pps_list': [], 'gnss_list': [], 'imu_list': []}
            segments.append(current_segment)
            segment_imu_count = 0
            prev_imu_micros = None  # Reset for new segment
            prev_imu_counter = None  # Reset for new segment
        
        if content[idx : idx + 4] == pps_marker:
            idx, should_break = process_pps_entry(
                content, idx, current_segment['pps_list'], current_segment['gnss_list'], current_segment['imu_list'],
                pps_marker, gps_marker, imu_marker, footer_marker,
                pps_struct_size
            )
            if should_break:
                break
                
        elif content[idx : idx + 4] == gps_marker:
            idx, should_break = process_gnss_entry(
                content, idx, current_segment['pps_list'], current_segment['gnss_list'], current_segment['imu_list'],
                pps_marker, gps_marker, imu_marker, footer_marker,
                gps_struct_size
            )
            if should_break:
                break
                
        elif content[idx : idx + 4] == imu_marker:
            entry_start_idx = idx
            idx, should_break, jump_detected = process_imu_entry(
                content, idx, current_segment['pps_list'], current_segment['gnss_list'], current_segment['imu_list'],
                pps_marker, gps_marker, imu_marker, footer_marker,
                imu_struct_size, acc_sensitivity, gyr_sensitivity,
                prev_imu_micros, mag_sensitivity,
                prev_imu_counter, clip_at_counter_discontinuity,
            )
            if should_break:
                # If process_imu_entry stopped at a counter discontinuity it did
                # not advance idx past the marker, so idx still points at the
                # offending entry. That's our clip point.
                if idx == entry_start_idx:
                    counter_clip_idx = idx
                    counter_clip_samples = sum(len(s['imu_list']) for s in segments)
                break

            # Check if jump was detected and we should start a new segment
            if jump_detected and len(current_segment['imu_list']) > 0:
                # Move the current IMU entry (which has the jump) to a new segment
                jumped_imu_entry = current_segment['imu_list'].pop()

                logger.info(
                    f"Starting segment {len(segments)} at byte {idx} "
                    f"(segment {len(segments)-1} had {segment_imu_count} IMUs, "
                    f"{len(current_segment['pps_list'])} PPS, {len(current_segment['gnss_list'])} GNSS) - MICROS JUMP"
                )

                # Start new segment with the jumped entry
                current_segment = {'pps_list': [], 'gnss_list': [], 'imu_list': [jumped_imu_entry]}
                segments.append(current_segment)
                segment_imu_count = 1
                prev_imu_micros = jumped_imu_entry.micros_reading
                prev_imu_counter = jumped_imu_entry.counter
            else:
                # Normal processing
                segment_imu_count += 1
                if len(current_segment['imu_list']) > 0:
                    prev_imu_micros = current_segment['imu_list'][-1].micros_reading
                    prev_imu_counter = current_segment['imu_list'][-1].counter
                
        elif footer_marker in content[idx : idx + len(footer_marker) + 10]:
            logger.info("Found footer marker, stopping parsing")
            break
        else:
            idx += 1
    
    # Check if file ended properly
    footer_found = footer_marker in content[max(0, idx - 100) : idx + 100]
    
    # Calculate totals across all segments
    total_pps = sum(len(seg['pps_list']) for seg in segments)
    total_gnss = sum(len(seg['gnss_list']) for seg in segments)
    total_imu = sum(len(seg['imu_list']) for seg in segments)
    
    if counter_clip_idx is not None:
        discarded = len(content) - counter_clip_idx
        logger.warning(
            f"Clipped at byte {counter_clip_idx} after {counter_clip_samples} valid "
            f"IMU samples — IMU counter discontinuity = end of real data. "
            f"{discarded:,} trailing bytes ({discarded/len(content)*100:.1f}% of file) "
            f"discarded as stale (pre-existing data on a non-zero-formatted SD card)."
        )
    elif not footer_found and idx >= len(content):
        logger.warning(f"Missing footer at end of file (byte {len(content)})")
        logger.error(
            f"File incomplete. Parsed {total_pps} PPS, "
            f"{total_gnss} GNSS, {total_imu} IMU entries before EOF"
        )
    elif not footer_found and idx < len(content):
        remaining_bytes = len(content) - idx
        logger.warning(
            f"Parsing stopped at byte {idx} ({remaining_bytes} bytes remaining, "
            f"{remaining_bytes / len(content) * 100:.1f}% of file unprocessed)"
        )
        logger.error(
            f"Unrecoverable corruption. Parsed {total_pps} PPS, "
            f"{total_gnss} GNSS, {total_imu} IMU before corruption"
        )
    
    # Log final parse statistics
    bytes_parsed = idx - start_offset
    total_entries = total_pps + total_gnss + total_imu
    
    logger.info(f"Created {len(segments)} segments")
    logger.info(f"Total: {total_pps} PPS, {total_gnss} GNSS, {total_imu} IMU entries")
    logger.info(f"Processed {bytes_parsed:,} bytes ({bytes_parsed / len(content) * 100:.1f}% of file)")
    if total_entries > 0:
        logger.info(f"Average {bytes_parsed / total_entries:.1f} bytes per entry")
    
    return segments



def save_decoded_data(
    segments: list[dict[str, list]],
    output_dir: Path,
    base_name: str,
    header_info: dict[str, Any],
    header_text: str,
    unwrap_stats: dict | None = None,
) -> dict[str, Path]:
    """Save decoded data to compressed numpy archive with segment naming.
    
    Args:
        segments: List of segment dicts, each with pps_list, gnss_list, imu_list
        output_dir: Directory to save files
        base_name: Base name for output file
        header_info: Dictionary of parsed header values
        header_text: Full header text string
        unwrap_stats: Optional unwrap statistics to include in return
        
    Returns:
        Dictionary with keys:
        - "file": Path to compressed .npz file
        - "unwrap_stats": Unwrap statistics (if provided)
    """
    output_files = {}
    
    # Prepare header data for storage
    header_string_array = np.array([header_text], dtype=object)
    
    save_dict = {
        "header_string": header_string_array,
        "number_of_segments": np.array([len(segments)]),
    }
    
    # Add individual header fields as separate arrays (without segment suffix)
    if "acc_sensitivity" in header_info:
        save_dict["acc_sensitivity"] = np.array([header_info["acc_sensitivity"]])
    if "gyr_sensitivity" in header_info:
        save_dict["gyr_sensitivity"] = np.array([header_info["gyr_sensitivity"]])
    if "imu_odr" in header_info:
        save_dict["imu_odr"] = np.array([header_info["imu_odr"]])
    if "gnss_rate" in header_info:
        save_dict["gnss_rate"] = np.array([header_info["gnss_rate"]])
    if "firmware_commit" in header_info:
        save_dict["firmware_commit"] = np.array([header_info["firmware_commit"]], dtype=object)
    
    # Save each segment with _segmentXXX naming
    for seg_idx, segment in enumerate(segments):
        seg_suffix = f"_segment{seg_idx:03d}"
        
        # Convert segment lists to arrays
        pps_array = np.array(segment['pps_list'], dtype=object)
        gnss_array = np.array(segment['gnss_list'], dtype=object)
        imu_array = np.array(segment['imu_list'], dtype=object)
        
        save_dict[f"pps{seg_suffix}"] = pps_array
        save_dict[f"gnss{seg_suffix}"] = gnss_array
        save_dict[f"imu{seg_suffix}"] = imu_array
    
    # Save as single compressed file
    npz_file = output_dir / f"{base_name}.npz"
    np.savez_compressed(npz_file, **save_dict)
    output_files["file"] = npz_file
    logger.info(f"Saved {len(segments)} segments to {npz_file} (compressed)")
    
    if unwrap_stats is not None:
        output_files["unwrap_stats"] = unwrap_stats
    
    return output_files


def load_and_combine_segments(npz_file: Path) -> dict[str, Any]:
    """Load segmented NPZ file and combine segments into single arrays.
    
    Args:
        npz_file: Path to the segmented .npz file
        
    Returns:
        Dictionary with combined arrays and header info:
        - 'pps': Combined PPS array
        - 'gnss': Combined GNSS array
        - 'imu': Combined IMU array
        - 'header_string': Header text
        - 'acc_sensitivity', 'gyr_sensitivity', etc.: Header fields
        - 'number_of_segments': Number of segments in original file
    """
    logger.info(f"Loading segmented data from {npz_file}")
    
    data = np.load(npz_file, allow_pickle=True)
    result = {}
    
    # Check if this is a segmented file (new format) or old format
    if 'number_of_segments' in data:
        # New segmented format
        number_of_segments = int(data['number_of_segments'][0])
        result['number_of_segments'] = number_of_segments
        logger.info(f"File contains {number_of_segments} segments")
        
        # Combine segments
        pps_segments = []
        gnss_segments = []
        imu_segments = []
        
        for seg_idx in range(number_of_segments):
            seg_suffix = f"_segment{seg_idx:03d}"
            pps_segments.append(data[f"pps{seg_suffix}"])
            gnss_segments.append(data[f"gnss{seg_suffix}"])
            imu_segments.append(data[f"imu{seg_suffix}"])
        
        # Concatenate all segments
        result['pps'] = np.concatenate(pps_segments) if pps_segments else np.array([])
        result['gnss'] = np.concatenate(gnss_segments) if gnss_segments else np.array([])
        result['imu'] = np.concatenate(imu_segments) if imu_segments else np.array([])
    else:
        # Old non-segmented format
        logger.info("Legacy format (no segments)")
        result['number_of_segments'] = 1
        result['pps'] = data.get('pps', np.array([]))
        result['gnss'] = data.get('gnss', np.array([]))
        result['imu'] = data.get('imu', np.array([]))
    
    logger.info(
        f"Combined: {len(result['pps'])} PPS, "
        f"{len(result['gnss'])} GNSS, {len(result['imu'])} IMU entries"
    )
    
    # Copy header info (non-segment data)
    for key in data.keys():
        # Skip segment arrays (these have _segmentXXX suffix)
        if '_segment' in key:
            continue
        # Skip data arrays we've already processed
        if key in ['pps', 'gnss', 'imu', 'number_of_segments']:
            continue
        # Copy all other keys (header info)
        result[key] = data[key]
    
    return result


def load_data_as_arrays(npz_file: Path) -> dict[str, np.ndarray]:
    """Load decoded data and extract all fields as individual numpy arrays.
    
    This is the recommended way for end users to load decoded data. It provides
    easy access to all sensor readings and timestamps as numpy arrays, ready for
    analysis and plotting.
    
    Args:
        npz_file: Path to the decoded .npz file
        
    Returns:
        Dictionary mapping field names to numpy arrays. Keys include:
        
        **Header Information:**
        - 'number_of_segments': Number of data segments (int)
        - 'firmware_commit': Firmware version string
        - 'acc_sensitivity': Accelerometer sensitivity (mg/LSB)
        - 'gyr_sensitivity': Gyroscope sensitivity (mdps/LSB)
        - 'imu_odr': IMU output data rate (Hz)
        - 'gnss_rate': GNSS update rate (Hz)
        - 'header_string': Full header text
        
        **PPS Data (N_pps entries):**
        - 'pps_micros': MCU microsecond timestamps
        - 'pps_micros_unwrapped': Unwrapped MCU timestamps
        - 'pps_utc': UTC timestamps from regression (seconds since epoch)
        
        **GNSS Data (N_gnss entries):**
        - 'gnss_micros': MCU microsecond timestamps
        - 'gnss_micros_unwrapped': Unwrapped MCU timestamps
        - 'gnss_latitude': Latitude (decimal degrees)
        - 'gnss_longitude': Longitude (decimal degrees)
        - 'gnss_vel_north': North velocity (mm/s)
        - 'gnss_vel_east': East velocity (mm/s)
        - 'gnss_vel_down': Down velocity (mm/s)
        - 'gnss_fix_type': GPS fix type (0=none, 2=2D, 3=3D)
        - 'gnss_posix': GNSS receiver POSIX timestamp
        - 'gnss_utc': UTC timestamps from regression (seconds since epoch)
        - 'gnss_latitude_outlier': Outlier flags for latitude
        - 'gnss_longitude_outlier': Outlier flags for longitude
        - 'gnss_vel_north_outlier': Outlier flags for north velocity
        - 'gnss_vel_east_outlier': Outlier flags for east velocity
        - 'gnss_vel_down_outlier': Outlier flags for down velocity
        
        **IMU Data (N_imu entries):**
        - 'imu_micros': MCU microsecond timestamps
        - 'imu_micros_unwrapped': Unwrapped MCU timestamps
        - 'imu_counter': Sample counter (may wrap at 65536)
        - 'imu_counter_unwrapped': Unwrapped sample counter
        - 'imu_acc_x': X acceleration (milli-g)
        - 'imu_acc_y': Y acceleration (milli-g)
        - 'imu_acc_z': Z acceleration (milli-g)
        - 'imu_gyr_x': X angular velocity (millidegrees/s)
        - 'imu_gyr_y': Y angular velocity (millidegrees/s)
        - 'imu_gyr_z': Z angular velocity (millidegrees/s)
        - 'imu_utc': UTC timestamps from regression (seconds since epoch)
        - 'imu_acc_x_outlier': Outlier flags for X acceleration
        - 'imu_acc_y_outlier': Outlier flags for Y acceleration
        - 'imu_acc_z_outlier': Outlier flags for Z acceleration
        - 'imu_gyr_x_outlier': Outlier flags for X gyroscope
        - 'imu_gyr_y_outlier': Outlier flags for Y gyroscope
        - 'imu_gyr_z_outlier': Outlier flags for Z gyroscope
        
    Example:
        >>> from pathlib import Path
        >>> from decoder import decode_file, load_data_as_arrays
        >>> import matplotlib.pyplot as plt
        >>> 
        >>> # Decode the file
        >>> result = decode_file(Path("DATA_BOOT_0000_TIME_20260204T193000.dat"))
        >>> 
        >>> # Load as arrays
        >>> data = load_data_as_arrays(result['file'])
        >>> 
        >>> # Access header info
        >>> print(f"IMU rate: {data['imu_odr']} Hz")
        >>> print(f"Firmware: {data['firmware_commit']}")
        >>> 
        >>> # Plot acceleration
        >>> plt.figure(figsize=(12, 6))
        >>> plt.plot(data['imu_utc'], data['imu_acc_x'], label='X')
        >>> plt.plot(data['imu_utc'], data['imu_acc_y'], label='Y')
        >>> plt.plot(data['imu_utc'], data['imu_acc_z'], label='Z')
        >>> plt.xlabel('UTC Time (s)')
        >>> plt.ylabel('Acceleration (mg)')
        >>> plt.legend()
        >>> plt.show()
        >>> 
        >>> # Plot GPS track
        >>> plt.figure(figsize=(10, 8))
        >>> plt.plot(data['gnss_longitude'], data['gnss_latitude'])
        >>> plt.xlabel('Longitude (°)')
        >>> plt.ylabel('Latitude (°)')
        >>> plt.title('GPS Track')
        >>> plt.show()
    """
    # First load and combine segments
    combined = load_and_combine_segments(npz_file)
    
    result = {}
    
    # Copy header information (scalars)
    scalar_keys = ['number_of_segments', 'firmware_commit', 'acc_sensitivity', 
                   'gyr_sensitivity', 'imu_odr', 'gnss_rate', 'header_string']
    for key in scalar_keys:
        if key in combined:
            if key == 'number_of_segments':
                result[key] = combined[key]
            else:
                result[key] = combined[key][0] if len(combined[key]) > 0 else None
    
    # Extract PPS arrays
    pps_data = combined['pps']
    if len(pps_data) > 0:
        result['pps_micros'] = np.array([p.micros_reading for p in pps_data])
        result['pps_micros_unwrapped'] = np.array([
            p.micros_reading_unwrapped if p.micros_reading_unwrapped is not None else np.nan
            for p in pps_data
        ])
        result['pps_utc'] = np.array([
            p.utc_timestamp_from_pps_regression if p.utc_timestamp_from_pps_regression is not None else np.nan
            for p in pps_data
        ])
    else:
        result['pps_micros'] = np.array([])
        result['pps_micros_unwrapped'] = np.array([])
        result['pps_utc'] = np.array([])
    
    # Extract GNSS arrays
    gnss_data = combined['gnss']
    if len(gnss_data) > 0:
        result['gnss_micros'] = np.array([g.micros_reading for g in gnss_data])
        result['gnss_micros_unwrapped'] = np.array([
            g.micros_reading_unwrapped if g.micros_reading_unwrapped is not None else np.nan
            for g in gnss_data
        ])
        result['gnss_latitude'] = np.array([g.latitude_dd for g in gnss_data])
        result['gnss_longitude'] = np.array([g.longitude_dd for g in gnss_data])
        result['gnss_vel_north'] = np.array([g.ned_vel_north_mmps for g in gnss_data])
        result['gnss_vel_east'] = np.array([g.ned_vel_east_mmps for g in gnss_data])
        result['gnss_vel_down'] = np.array([g.ned_vel_down_mmps for g in gnss_data])
        # Altitude above mean sea level (m). 0.0 for recordings from firmware
        # versions that didn't log altitude — check header['gnss_has_altitude'].
        result['gnss_altitude_msl'] = np.array([g.altitude_msl_m for g in gnss_data])
        result['gnss_fix_type'] = np.array([g.fix_type for g in gnss_data])
        result['gnss_posix'] = np.array([g.posix_timestamp + g.microseconds * 1e-6 for g in gnss_data])
        result['gnss_utc'] = np.array([
            g.utc_timestamp_from_pps_regression if g.utc_timestamp_from_pps_regression is not None else np.nan
            for g in gnss_data
        ])
        # Outlier flags
        result['gnss_latitude_outlier'] = np.array([g.latitude_dd_stdchecked for g in gnss_data], dtype=bool)
        result['gnss_longitude_outlier'] = np.array([g.longitude_dd_stdchecked for g in gnss_data], dtype=bool)
        result['gnss_vel_north_outlier'] = np.array([g.ned_vel_north_mmps_stdchecked for g in gnss_data], dtype=bool)
        result['gnss_vel_east_outlier'] = np.array([g.ned_vel_east_mmps_stdchecked for g in gnss_data], dtype=bool)
        result['gnss_vel_down_outlier'] = np.array([g.ned_vel_down_mmps_stdchecked for g in gnss_data], dtype=bool)
        result['gnss_altitude_msl_outlier'] = np.array([g.altitude_msl_m_stdchecked for g in gnss_data], dtype=bool)
    else:
        result['gnss_micros'] = np.array([])
        result['gnss_micros_unwrapped'] = np.array([])
        result['gnss_latitude'] = np.array([])
        result['gnss_longitude'] = np.array([])
        result['gnss_vel_north'] = np.array([])
        result['gnss_vel_east'] = np.array([])
        result['gnss_vel_down'] = np.array([])
        result['gnss_altitude_msl'] = np.array([])
        result['gnss_fix_type'] = np.array([])
        result['gnss_posix'] = np.array([])
        result['gnss_utc'] = np.array([])
        result['gnss_latitude_outlier'] = np.array([], dtype=bool)
        result['gnss_longitude_outlier'] = np.array([], dtype=bool)
        result['gnss_vel_north_outlier'] = np.array([], dtype=bool)
        result['gnss_vel_east_outlier'] = np.array([], dtype=bool)
        result['gnss_vel_down_outlier'] = np.array([], dtype=bool)
        result['gnss_altitude_msl_outlier'] = np.array([], dtype=bool)
    
    # Extract IMU arrays
    imu_data = combined['imu']
    if len(imu_data) > 0:
        result['imu_micros'] = np.array([i.micros_reading for i in imu_data])
        result['imu_micros_unwrapped'] = np.array([
            i.micros_reading_unwrapped if i.micros_reading_unwrapped is not None else np.nan
            for i in imu_data
        ])
        result['imu_counter'] = np.array([i.counter for i in imu_data])
        result['imu_counter_unwrapped'] = np.array([
            i.counter_unwrapped if i.counter_unwrapped is not None else np.nan
            for i in imu_data
        ])
        result['imu_acc_x'] = np.array([i.acc_x_mg for i in imu_data])
        result['imu_acc_y'] = np.array([i.acc_y_mg for i in imu_data])
        result['imu_acc_z'] = np.array([i.acc_z_mg for i in imu_data])
        result['imu_gyr_x'] = np.array([i.gyr_x_mdps for i in imu_data])
        result['imu_gyr_y'] = np.array([i.gyr_y_mdps for i in imu_data])
        result['imu_gyr_z'] = np.array([i.gyr_z_mdps for i in imu_data])
        # Magnetometer (AK09916). Zero arrays when the firmware didn't log mag.
        result['imu_mag_x'] = np.array([i.mag_x_uT for i in imu_data])
        result['imu_mag_y'] = np.array([i.mag_y_uT for i in imu_data])
        result['imu_mag_z'] = np.array([i.mag_z_uT for i in imu_data])
        result['imu_utc'] = np.array([
            i.utc_timestamp_from_pps_regression if i.utc_timestamp_from_pps_regression is not None else np.nan
            for i in imu_data
        ])
        # Outlier flags
        result['imu_acc_x_outlier'] = np.array([i.acc_x_mg_stdchecked for i in imu_data], dtype=bool)
        result['imu_acc_y_outlier'] = np.array([i.acc_y_mg_stdchecked for i in imu_data], dtype=bool)
        result['imu_acc_z_outlier'] = np.array([i.acc_z_mg_stdchecked for i in imu_data], dtype=bool)
        result['imu_gyr_x_outlier'] = np.array([i.gyr_x_mdps_stdchecked for i in imu_data], dtype=bool)
        result['imu_gyr_y_outlier'] = np.array([i.gyr_y_mdps_stdchecked for i in imu_data], dtype=bool)
        result['imu_gyr_z_outlier'] = np.array([i.gyr_z_mdps_stdchecked for i in imu_data], dtype=bool)
        result['imu_mag_x_outlier'] = np.array([i.mag_x_uT_stdchecked for i in imu_data], dtype=bool)
        result['imu_mag_y_outlier'] = np.array([i.mag_y_uT_stdchecked for i in imu_data], dtype=bool)
        result['imu_mag_z_outlier'] = np.array([i.mag_z_uT_stdchecked for i in imu_data], dtype=bool)
    else:
        result['imu_micros'] = np.array([])
        result['imu_micros_unwrapped'] = np.array([])
        result['imu_counter'] = np.array([])
        result['imu_counter_unwrapped'] = np.array([])
        result['imu_acc_x'] = np.array([])
        result['imu_acc_y'] = np.array([])
        result['imu_acc_z'] = np.array([])
        result['imu_gyr_x'] = np.array([])
        result['imu_gyr_y'] = np.array([])
        result['imu_gyr_z'] = np.array([])
        result['imu_mag_x'] = np.array([])
        result['imu_mag_y'] = np.array([])
        result['imu_mag_z'] = np.array([])
        result['imu_utc'] = np.array([])
        result['imu_acc_x_outlier'] = np.array([], dtype=bool)
        result['imu_acc_y_outlier'] = np.array([], dtype=bool)
        result['imu_acc_z_outlier'] = np.array([], dtype=bool)
        result['imu_gyr_x_outlier'] = np.array([], dtype=bool)
        result['imu_gyr_y_outlier'] = np.array([], dtype=bool)
        result['imu_gyr_z_outlier'] = np.array([], dtype=bool)
        result['imu_mag_x_outlier'] = np.array([], dtype=bool)
        result['imu_mag_y_outlier'] = np.array([], dtype=bool)
        result['imu_mag_z_outlier'] = np.array([], dtype=bool)
    
    return result


def decode_file(
    input_file: Path,
    output_dir: Path | None = None,
    show_plots: bool = False,
    pps_marker: bytes = PPS_MARKER,
    gps_marker: bytes = GPS_MARKER,
    imu_marker: bytes = IMU_MARKER,
    footer_marker: bytes = FOOTER_MARKER,
    pps_struct_size: int = PPS_STRUCT_SIZE,
    gps_struct_size: int = GPS_STRUCT_SIZE,
    imu_struct_size: int = IMU_STRUCT_SIZE,
    allow_no_pps: bool = False,
    clip_at_counter_discontinuity: bool = True,
) -> dict[str, Path]:
    """Decode a single data file and save to compressed numpy archive with segments.

    This function performs the complete decoding pipeline:
    1. Parses file header to extract sensor sensitivities
    2. Scans binary file for PPS, GNSS, and IMU entries in 1-minute segments
    3. For each segment independently:
       - Unwraps potentially wrapping counters and detects anomalies
       - Computes linear regression from PPS+GNSS to get UTC timestamps
       - Applies regression to all entries for synchronized timestamps
    4. Saves decoded data to compressed .npz file with segment naming

    Args:
        input_file: Path to input data file
        output_dir: Directory to save output files (defaults to same as input)
        show_plots: If True, display ASCII plots (e.g., PPS mismatch plot)
        pps_marker, gps_marker, imu_marker, footer_marker: Entry markers
        pps_struct_size: Size of PPS struct in bytes (default: 4)
        gps_struct_size: Size of GPS struct in bytes (default: 36)
        imu_struct_size: Size of IMU struct in bytes (default: 18)

    Returns:
        Dictionary with keys:
        - "file": Path to compressed .npz file
        - "unwrap_stats": Unwrap statistics with wrap/jump counts per segment

    Raises:
        AssertionError: If binary data structure doesn't match expected format
        struct.error: If binary unpacking fails
    """
    logger.info(f"Decoding file: {input_file}")

    if output_dir is None:
        output_dir = input_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    header_info, header_text = parse_header(input_file, markers=(pps_marker, gps_marker, imu_marker))

    with open(input_file, "rb") as f:
        content = f.read()

    # Parse binary content into segments
    segments = parse_binary_content(
        content, header_info, pps_marker, gps_marker, imu_marker, footer_marker,
        clip_at_counter_discontinuity=clip_at_counter_discontinuity,
    )

    # Process each segment independently with unwrap offset carryover
    all_unwrap_stats = {}
    
    # Initialize unwrap offsets and last raw values (carry over between segments)
    pps_micros_offset = 0
    pps_micros_prev_raw = None
    gnss_micros_offset = 0
    gnss_micros_prev_raw = None
    imu_micros_offset = 0
    imu_micros_prev_raw = None
    imu_counter_offset = 0
    imu_counter_prev_raw = None
    
    for seg_idx, segment in enumerate(segments):
        logger.info(f"Processing segment {seg_idx}...")
        
        pps_list = segment['pps_list']
        gnss_list = segment['gnss_list']
        imu_list = segment['imu_list']
        
        # Skip empty segments
        if not pps_list and not gnss_list and not imu_list:
            logger.warning(f"Segment {seg_idx} is empty, skipping")
            continue
        
        # Find minimum micros reading for this segment
        all_micros = (
            [p.micros_reading for p in pps_list] +
            [g.micros_reading for g in gnss_list] +
            [i.micros_reading for i in imu_list]
        )
        segment_min_micros = min(all_micros) if all_micros else 0
        
        # Unwrap potentially wrapping arrays and detect jumps (per segment with offset carryover)
        segment_unwrap_stats = {}

        # Process PPS micros_reading
        if pps_list:
            pps_micros = np.array([p.micros_reading for p in pps_list])
            pps_micros_unwrapped, pps_micros_wraps, pps_micros_jumps, pps_micros_offset, pps_micros_prev_raw = unwrap_array(
                pps_micros, max_value=2**32, initial_offset=pps_micros_offset, prev_raw_value=pps_micros_prev_raw
            )
            for i, pps in enumerate(pps_list):
                pps.micros_reading_unwrapped = int(pps_micros_unwrapped[i])

            segment_unwrap_stats["PPS"] = {
                "micros_reading": {
                    "wraps": 0 if pps_micros_wraps is None else len(pps_micros_wraps),
                    "jumps": 0 if pps_micros_jumps is None else len(pps_micros_jumps),
                    "wrap_indices": pps_micros_wraps,
                    "jump_indices": pps_micros_jumps,
                }
            }

        # Process GNSS micros_reading
        if gnss_list:
            gnss_micros = np.array([g.micros_reading for g in gnss_list])
            gnss_micros_unwrapped, gnss_micros_wraps, gnss_micros_jumps, gnss_micros_offset, gnss_micros_prev_raw = unwrap_array(
                gnss_micros, max_value=2**32, initial_offset=gnss_micros_offset, prev_raw_value=gnss_micros_prev_raw
            )
            for i, gnss in enumerate(gnss_list):
                gnss.micros_reading_unwrapped = int(gnss_micros_unwrapped[i])

            segment_unwrap_stats["GNSS"] = {
                "micros_reading": {
                    "wraps": 0 if gnss_micros_wraps is None else len(gnss_micros_wraps),
                    "jumps": 0 if gnss_micros_jumps is None else len(gnss_micros_jumps),
                    "wrap_indices": gnss_micros_wraps,
                    "jump_indices": gnss_micros_jumps,
                }
            }

        # Process IMU micros_reading and counter
        if imu_list:
            imu_micros = np.array([i.micros_reading for i in imu_list])
            imu_micros_unwrapped, imu_micros_wraps, imu_micros_jumps, imu_micros_offset, imu_micros_prev_raw = unwrap_array(
                imu_micros, max_value=2**32, initial_offset=imu_micros_offset, prev_raw_value=imu_micros_prev_raw
            )

            imu_counter = np.array([i.counter for i in imu_list])
            imu_counter_unwrapped, imu_counter_wraps, imu_counter_jumps, imu_counter_offset, imu_counter_prev_raw = unwrap_array(
                imu_counter, max_value=2**16, jump_threshold=1, initial_offset=imu_counter_offset, prev_raw_value=imu_counter_prev_raw
            )

            for i, imu in enumerate(imu_list):
                imu.micros_reading_unwrapped = int(imu_micros_unwrapped[i])
                imu.counter_unwrapped = int(imu_counter_unwrapped[i])

            segment_unwrap_stats["IMU"] = {
                "micros_reading": {
                    "wraps": 0 if imu_micros_wraps is None else len(imu_micros_wraps),
                    "jumps": 0 if imu_micros_jumps is None else len(imu_micros_jumps),
                    "wrap_indices": imu_micros_wraps,
                    "jump_indices": imu_micros_jumps,
                },
                "counter": {
                    "wraps": 0 if imu_counter_wraps is None else len(imu_counter_wraps),
                    "jumps": 0 if imu_counter_jumps is None else len(imu_counter_jumps),
                    "wrap_indices": imu_counter_wraps,
                    "jump_indices": imu_counter_jumps,
                },
            }
        
        # Store unwrap stats for this segment
        all_unwrap_stats[f"segment_{seg_idx:03d}"] = segment_unwrap_stats
        
        # Don't carry forward offsets/prev_raw from segments with insufficient data
        # This prevents anomalous single-entry segments from contaminating next segment's unwrapping
        # IMPORTANT: When resetting prev_raw, also reset offset to maintain consistency
        MIN_ENTRIES_FOR_CARRYOVER = 10
        
        if len(pps_list) < MIN_ENTRIES_FOR_CARRYOVER:
            pps_micros_prev_raw = None  # Reset for next segment
            pps_micros_offset = 0  # Reset offset too
        if len(gnss_list) < MIN_ENTRIES_FOR_CARRYOVER:
            gnss_micros_prev_raw = None  # Reset for next segment  
            gnss_micros_offset = 0  # Reset offset too
        if len(imu_list) < MIN_ENTRIES_FOR_CARRYOVER:
            imu_micros_prev_raw = None  # Reset for next segment
            imu_micros_offset = 0  # Reset offset too
            imu_counter_prev_raw = None  # Reset for next segment
            imu_counter_offset = 0  # Reset offset too

        # Sanity check: verify micros timestamps are consistent across data types
        try:
            check_micros_consistency(pps_list, gnss_list, imu_list, seg_idx)
        except ValueError as e:
            logger.error(f"Segment {seg_idx} failed consistency check: {e}")
            logger.error(f"This segment will be marked as invalid")
            segment['regression_valid'] = False
            # Don't carry forward offsets from failed segments
            pps_micros_prev_raw = None
            pps_micros_offset = 0
            gnss_micros_prev_raw = None
            gnss_micros_offset = 0
            imu_micros_prev_raw = None
            imu_micros_offset = 0
            imu_counter_prev_raw = None
            imu_counter_offset = 0
            continue

        # Compute PPS regression for this segment
        logger.info(f"Computing PPS to UTC timestamp regression for segment {seg_idx}...")
        regression = compute_pps_regression(pps_list, gnss_list, segment_min_micros)
        
        # Track regression quality for filtering
        segment['regression_valid'] = False
        
        if regression is None:
            logger.warning(f"Skipping PPS regression for segment {seg_idx} due to insufficient data")
        else:
            slope, intercept, r_squared = regression
            apply_pps_regression(pps_list, gnss_list, imu_list, slope, intercept)
            
            # Check PPS mismatch statistics
            max_mismatch = None
            if pps_list:
                max_mismatch = compute_pps_mismatch_statistics(pps_list)
            
            # Validate regression quality
            # Thresholds: R² ≥ 0.99 and max mismatch ≤ 200ms
            R2_THRESHOLD = 0.99
            MAX_MISMATCH_THRESHOLD = 0.200  # 200ms in seconds
            
            regression_good = True
            if r_squared < R2_THRESHOLD:
                logger.error(
                    f"⚠️  Poor regression quality in segment {seg_idx}: "
                    f"R²={r_squared:.6f} < {R2_THRESHOLD} threshold"
                )
                regression_good = False
            
            if max_mismatch is not None and max_mismatch > MAX_MISMATCH_THRESHOLD:
                logger.error(
                    f"⚠️  Excessive PPS mismatch in segment {seg_idx}: "
                    f"max={max_mismatch*1000:.1f}ms > {MAX_MISMATCH_THRESHOLD*1000:.0f}ms threshold"
                )
                regression_good = False
            
            if not regression_good:
                logger.error(
                    f"❌ Segment {seg_idx} has poor GPS synchronization and will be discarded"
                )
            else:
                segment['regression_valid'] = True

        # Print summary statistics for this segment
        logger.info(
            f"Segment {seg_idx}: {len(pps_list)} PPS, "
            f"{len(gnss_list)} GNSS, {len(imu_list)} IMU entries"
        )
        
        # Apply outlier detection to physical variables
        # IMU acceleration and gyroscope
        if imu_list and len(imu_list) > 4:
            acc_x_values = np.array([i.acc_x_mg for i in imu_list])
            acc_y_values = np.array([i.acc_y_mg for i in imu_list])
            acc_z_values = np.array([i.acc_z_mg for i in imu_list])
            gyr_x_values = np.array([i.gyr_x_mdps for i in imu_list])
            gyr_y_values = np.array([i.gyr_y_mdps for i in imu_list])
            gyr_z_values = np.array([i.gyr_z_mdps for i in imu_list])
            
            acc_x_outliers = detect_outliers_stdcheck(acc_x_values)
            acc_y_outliers = detect_outliers_stdcheck(acc_y_values)
            acc_z_outliers = detect_outliers_stdcheck(acc_z_values)
            gyr_x_outliers = detect_outliers_stdcheck(gyr_x_values)
            gyr_y_outliers = detect_outliers_stdcheck(gyr_y_values)
            gyr_z_outliers = detect_outliers_stdcheck(gyr_z_values)
            
            # Flag outliers in the data structures
            apply_outlier_flags(imu_list, 'acc_x_mg_stdchecked', acc_x_outliers)
            apply_outlier_flags(imu_list, 'acc_y_mg_stdchecked', acc_y_outliers)
            apply_outlier_flags(imu_list, 'acc_z_mg_stdchecked', acc_z_outliers)
            apply_outlier_flags(imu_list, 'gyr_x_mdps_stdchecked', gyr_x_outliers)
            apply_outlier_flags(imu_list, 'gyr_y_mdps_stdchecked', gyr_y_outliers)
            apply_outlier_flags(imu_list, 'gyr_z_mdps_stdchecked', gyr_z_outliers)
            
            n_acc_outliers = len(acc_x_outliers) + len(acc_y_outliers) + len(acc_z_outliers)
            n_gyr_outliers = len(gyr_x_outliers) + len(gyr_y_outliers) + len(gyr_z_outliers)
            if n_acc_outliers > 0 or n_gyr_outliers > 0:
                logger.info(
                    f"Segment {seg_idx} IMU outliers detected: "
                    f"{n_acc_outliers} acceleration, {n_gyr_outliers} gyroscope"
                )
        
        # GNSS position and velocity
        if gnss_list and len(gnss_list) > 4:
            lat_values = np.array([g.latitude_dd for g in gnss_list])
            lon_values = np.array([g.longitude_dd for g in gnss_list])
            vel_n_values = np.array([g.ned_vel_north_mmps for g in gnss_list])
            vel_e_values = np.array([g.ned_vel_east_mmps for g in gnss_list])
            vel_d_values = np.array([g.ned_vel_down_mmps for g in gnss_list])
            
            lat_outliers = detect_outliers_stdcheck(lat_values)
            lon_outliers = detect_outliers_stdcheck(lon_values)
            vel_n_outliers = detect_outliers_stdcheck(vel_n_values)
            vel_e_outliers = detect_outliers_stdcheck(vel_e_values)
            vel_d_outliers = detect_outliers_stdcheck(vel_d_values)
            
            # Flag outliers in the data structures
            apply_outlier_flags(gnss_list, 'latitude_dd_stdchecked', lat_outliers)
            apply_outlier_flags(gnss_list, 'longitude_dd_stdchecked', lon_outliers)
            apply_outlier_flags(gnss_list, 'ned_vel_north_mmps_stdchecked', vel_n_outliers)
            apply_outlier_flags(gnss_list, 'ned_vel_east_mmps_stdchecked', vel_e_outliers)
            apply_outlier_flags(gnss_list, 'ned_vel_down_mmps_stdchecked', vel_d_outliers)
            
            n_pos_outliers = len(lat_outliers) + len(lon_outliers)
            n_vel_outliers = len(vel_n_outliers) + len(vel_e_outliers) + len(vel_d_outliers)
            if n_pos_outliers > 0 or n_vel_outliers > 0:
                logger.info(
                    f"Segment {seg_idx} GNSS outliers detected: "
                    f"{n_pos_outliers} position, {n_vel_outliers} velocity"
                )

    # Filter out segments that are too small for meaningful GPS synchronization
    # OR have poor regression quality (bad R² or high mismatch)
    # Only applies when: (1) file has GNSS data, AND (2) there are multiple segments
    # Small segments in single-segment files or non-GNSS files are kept
    has_any_gnss = any(len(seg['gnss_list']) > 0 for seg in segments)
    has_multiple_segments = len(segments) > 1
    
    valid_segments = []
    skipped_segments = []
    bad_regression_segments = []
    
    for seg_idx, segment in enumerate(segments):
        pps_count = len(segment['pps_list'])
        gnss_count = len(segment['gnss_list'])
        imu_count = len(segment['imu_list'])
        
        # Check if segment has poor regression quality
        has_bad_regression = not segment.get('regression_valid', False)
        
        # Only filter small segments if:
        # - File has GNSS data (need GPS sync)
        # - File has multiple segments (one small segment at end is problematic)
        should_filter_small = has_any_gnss and has_multiple_segments
        
        # Filter segments with bad regression quality only if file has:
        # - GNSS data (need GPS sync)
        # - Multiple segments (single segment files are kept even with bad regression)
        should_filter_regression = has_any_gnss and has_bad_regression and pps_count >= 2 and has_multiple_segments
        
        if should_filter_small and (pps_count < 2 or gnss_count < 1) and not allow_no_pps:
            skipped_segments.append(seg_idx)
            logger.warning(
                f"Skipping segment {seg_idx} (insufficient for GPS sync): "
                f"{pps_count} PPS, {gnss_count} GNSS, {imu_count} IMU entries"
            )
        elif should_filter_regression and not allow_no_pps:
            bad_regression_segments.append(seg_idx)
            logger.warning(
                f"Skipping segment {seg_idx} (poor GPS sync quality): "
                f"{pps_count} PPS, {gnss_count} GNSS, {imu_count} IMU entries"
            )
        else:
            valid_segments.append(segment)
    
    if skipped_segments:
        logger.info(f"Skipped {len(skipped_segments)} small segment(s): {skipped_segments}")
    if bad_regression_segments:
        logger.info(f"Skipped {len(bad_regression_segments)} bad regression segment(s): {bad_regression_segments}")
    
    # Use valid segments for saving
    segments = valid_segments

    # Print overall summary statistics
    total_pps = sum(len(seg['pps_list']) for seg in segments)
    total_gnss = sum(len(seg['gnss_list']) for seg in segments)
    total_imu = sum(len(seg['imu_list']) for seg in segments)
    logger.info(
        f"Total across all segments: {total_pps} PPS, "
        f"{total_gnss} GNSS, {total_imu} IMU entries"
    )

    # Save results and return file paths with unwrap stats
    return save_decoded_data(
        segments, output_dir, input_file.stem,
        header_info, header_text, all_unwrap_stats
    )
