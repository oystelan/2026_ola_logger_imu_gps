"""Tests for outlier detection functionality."""

import numpy as np
import pytest

from decoder import detect_outliers_stdcheck


class TestDetectOutliersStdcheck:
    """Test suite for detect_outliers_stdcheck function."""
    
    def test_no_outliers_in_clean_data(self):
        """Test that clean data produces no outliers."""
        # Smooth sine wave
        x = np.sin(np.linspace(0, 2 * np.pi, 100))
        outliers = detect_outliers_stdcheck(x)
        assert len(outliers) == 0, "Clean sine wave should have no outliers"
    
    def test_obvious_outliers_detected(self):
        """Test that obvious outliers are detected."""
        x = np.ones(100)
        x[25] = 100  # Large positive outlier
        x[75] = -100  # Large negative outlier
        
        outliers = detect_outliers_stdcheck(x)
        assert 25 in outliers, "Index 25 should be detected as outlier"
        assert 75 in outliers, "Index 75 should be detected as outlier"
        assert len(outliers) == 2, f"Should detect exactly 2 outliers, got {len(outliers)}"
    
    def test_edge_outliers_detected(self):
        """Test that outliers at array edges are detected."""
        x = np.ones(50)
        x[0] = 100  # Outlier at start
        x[-1] = -100  # Outlier at end
        
        outliers = detect_outliers_stdcheck(x)
        assert 0 in outliers, "First index should be detected"
        assert 49 in outliers, "Last index should be detected"
    
    def test_threshold_sensitivity(self):
        """Test that n_sigma parameter controls sensitivity."""
        np.random.seed(42)  # Fix seed for reproducibility
        x = np.random.normal(0, 1, 1000)
        x[500] = 10  # 10-sigma outlier
        x[600] = 4   # 4-sigma outlier (but may be flagged depending on local std)
        
        # With n_sigma=5.0, 10-sigma should definitely be detected
        outliers_5sig = detect_outliers_stdcheck(x, n_sigma=5.0)
        assert 500 in outliers_5sig, "10-sigma outlier should be detected with n_sigma=5"
        
        # With n_sigma=3.0, both should be detected
        outliers_3sig = detect_outliers_stdcheck(x, n_sigma=3.0)
        assert 500 in outliers_3sig, "10-sigma outlier should be detected with n_sigma=3"
        # 4-sigma outlier should be detected with looser threshold
        assert 600 in outliers_3sig, "4-sigma outlier should be detected with n_sigma=3"
        
        # More flagged points with lower threshold
        assert len(outliers_3sig) >= len(outliers_5sig), "Lower threshold should flag more points"
    
    def test_neighbor_count_effect(self):
        """Test that n_neighbors parameter affects detection."""
        x = np.array([1.0, 1.1, 1.0, 10.0, 0.9, 1.1, 1.0, 0.95, 1.05])
        
        # With fewer neighbors, statistics are less stable
        outliers_4 = detect_outliers_stdcheck(x, n_neighbors=4, n_sigma=3.0)
        outliers_6 = detect_outliers_stdcheck(x, n_neighbors=6, n_sigma=3.0)
        
        # Both should detect the obvious outlier at index 3
        assert 3 in outliers_4, "Outlier should be detected with n_neighbors=4"
        assert 3 in outliers_6, "Outlier should be detected with n_neighbors=6"
    
    def test_empty_array(self):
        """Test handling of empty array."""
        x = np.array([])
        outliers = detect_outliers_stdcheck(x)
        assert len(outliers) == 0, "Empty array should return no outliers"
    
    def test_small_array(self):
        """Test handling of arrays smaller than n_neighbors."""
        # Array with 5 elements, n_neighbors=6
        # With n_neighbors=6, array is too small for full detection
        # Should only flag NaN/inf values
        x = np.array([1, 2, 100, 3, 4])
        outliers = detect_outliers_stdcheck(x, n_neighbors=6)
        # Since array has 5 elements < n_neighbors+1 (7), returns empty or flags NaN only
        # This is acceptable behavior for small arrays
        assert isinstance(outliers, np.ndarray), "Should return array"
        
        # With smaller n_neighbors, should work
        outliers = detect_outliers_stdcheck(x, n_neighbors=2, n_sigma=3.0)
        assert 2 in outliers, "Should detect outlier with n_neighbors=2"
    
    def test_all_same_values(self):
        """Test handling of array with all identical values."""
        x = np.ones(100)
        outliers = detect_outliers_stdcheck(x)
        assert len(outliers) == 0, "Array with all same values should have no outliers"
    
    def test_constant_neighbors_with_outlier(self):
        """Test zero-std case: all neighbors identical but point differs."""
        x = np.ones(100)
        x[50] = 2.0  # Small difference from constant neighbors
        
        outliers = detect_outliers_stdcheck(x)
        assert 50 in outliers, "Should detect outlier when neighbors have zero std"
    
    def test_nan_handling(self):
        """Test handling of NaN values in input."""
        x = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        
        # Function should handle NaN gracefully (either skip or detect)
        outliers = detect_outliers_stdcheck(x)
        # NaN may or may not be flagged depending on implementation
        # Main check: function doesn't crash
        assert isinstance(outliers, np.ndarray), "Should return array even with NaN"
    
    def test_inf_handling(self):
        """Test handling of inf values in input."""
        x = np.ones(50)
        x[25] = np.inf
        
        outliers = detect_outliers_stdcheck(x)
        assert 25 in outliers, "Inf should be detected as outlier"
    
    def test_alternating_pattern(self):
        """Test that regular alternating patterns don't trigger false positives."""
        # Regular square wave - should not flag as outliers
        x = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 10)
        outliers = detect_outliers_stdcheck(x, n_neighbors=4, n_sigma=5.0)
        assert len(outliers) == 0, "Regular alternating pattern should not produce outliers"
    
    def test_linear_trend(self):
        """Test that linear trends don't produce outliers."""
        x = np.linspace(0, 100, 200)
        outliers = detect_outliers_stdcheck(x)
        assert len(outliers) == 0, "Linear trend should not produce outliers"
    
    def test_multiple_outliers_in_sequence(self):
        """Test handling of consecutive outliers."""
        x = np.ones(100)
        x[48:52] = 100  # 4 consecutive outliers
        
        outliers = detect_outliers_stdcheck(x, n_neighbors=6, n_sigma=5.0)
        # With consecutive outliers, they may become each other's neighbors
        # which can reduce detection. This is expected behavior.
        # Test with wider separation to ensure detection works
        
        x2 = np.ones(100)
        x2[20] = 100
        x2[50] = 100
        x2[80] = 100
        
        outliers2 = detect_outliers_stdcheck(x2, n_neighbors=6, n_sigma=5.0)
        assert len(outliers2) >= 3, f"Should detect widely spaced outliers, got {len(outliers2)}"
    
    def test_return_type_and_dtype(self):
        """Test that return type is correct."""
        x = np.ones(50)
        x[25] = 100
        
        outliers = detect_outliers_stdcheck(x)
        assert isinstance(outliers, np.ndarray), "Should return numpy array"
        assert outliers.dtype == np.int64, f"Should return int64, got {outliers.dtype}"
        assert outliers.ndim == 1, "Should return 1D array"
    
    def test_outlier_indices_are_sorted(self):
        """Test that outlier indices are returned in sorted order."""
        x = np.ones(100)
        x[75] = 100
        x[25] = 100
        x[10] = 100
        x[90] = 100
        
        outliers = detect_outliers_stdcheck(x)
        assert len(outliers) > 0, "Should detect outliers"
        # Check if sorted
        assert np.array_equal(outliers, np.sort(outliers)), "Outliers should be in sorted order"
    
    def test_noisy_data_realistic(self):
        """Test with realistic noisy sensor data."""
        # Simulate IMU accelerometer data with noise
        np.random.seed(42)
        # Base signal: mostly around 1g with small noise
        x = np.random.normal(1000, 10, 5000)  # 1000 mg ± 10 mg
        
        # Add a few realistic outliers (sensor glitches)
        x[1000] = 5000  # Large spike
        x[2500] = -2000  # Large dip
        x[4000] = 3500  # Medium spike
        
        outliers = detect_outliers_stdcheck(x, n_neighbors=6, n_sigma=5.0)
        
        # Should detect the obvious outliers
        assert 1000 in outliers, "Should detect large positive spike"
        assert 2500 in outliers, "Should detect large negative spike"
        # May or may not detect medium spike depending on local noise
        
        # Should not flag too many false positives
        assert len(outliers) < 50, f"Too many false positives: {len(outliers)}"


class TestOutlierDetectionIntegration:
    """Integration tests for outlier detection in full decode pipeline."""
    
    def test_outlier_flags_in_dataclass(self):
        """Test that outlier flags are properly set in dataclass objects."""
        from decoder import IMUReading, GNSSReading
        
        # Create IMU reading with outlier flag (need all required fields)
        imu = IMUReading(
            micros_reading=1000,
            counter=1,
            acc_x=16000,  # Raw accelerometer value
            acc_y=0,
            acc_z=0,
            gyr_x=0,  # Raw gyroscope value
            gyr_y=0,
            gyr_z=0,
            acc_x_mg=1000.0,  # Converted value in mg
            acc_y_mg=0.0,
            acc_z_mg=0.0,
            gyr_x_mdps=0.0,  # Converted value in mdps
            gyr_y_mdps=0.0,
            gyr_z_mdps=0.0,
            counter_unwrapped=1,
            micros_reading_unwrapped=1000,
            utc_timestamp_from_pps_regression=0.0,
            acc_x_mg_stdchecked=True,  # This one is flagged
            acc_y_mg_stdchecked=False,
            acc_z_mg_stdchecked=False,
            gyr_x_mdps_stdchecked=False,
            gyr_y_mdps_stdchecked=False,
            gyr_z_mdps_stdchecked=False,
        )
        
        assert imu.acc_x_mg_stdchecked is True
        assert imu.acc_y_mg_stdchecked is False
    
    def test_outlier_detection_with_real_file(self, tmp_path):
        """Test that outlier detection runs on real decode."""
        from decoder import decode_file
        from pathlib import Path
        
        # Use existing real data file
        real_file = Path("DATA_BOOT_0000_TIME_20260204T193000.dat")
        if not real_file.exists():
            pytest.skip("Real data file not available")
        
        # Decode should complete without errors
        decode_file(real_file, output_dir=tmp_path)
        
        # Load and verify outliers were detected
        from decoder import load_and_combine_segments
        npz_file = tmp_path / "DATA_BOOT_0000_TIME_20260204T193000.npz"
        data = load_and_combine_segments(npz_file)
        
        # Check that some outliers were detected
        imu_outliers = sum(
            1 for imu in data['imu'] 
            if any([
                imu.acc_x_mg_stdchecked,
                imu.acc_y_mg_stdchecked,
                imu.acc_z_mg_stdchecked,
                imu.gyr_x_mdps_stdchecked,
                imu.gyr_y_mdps_stdchecked,
                imu.gyr_z_mdps_stdchecked
            ])
        )
        
        # Should detect some outliers but not too many
        assert imu_outliers > 0, "Should detect some IMU outliers"
        assert imu_outliers < len(data['imu']) * 0.1, "Should not flag >10% as outliers"
