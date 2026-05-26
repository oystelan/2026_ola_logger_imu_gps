"""Tests for the decoder_cli module."""

from pathlib import Path
from unittest.mock import patch

import matplotlib
import pytest

# Use non-interactive backend for tests
matplotlib.use("Agg")

from decoder_cli import (
    plot_gnss_coordinates,
    plot_imu_acceleration,
    plot_imu_gyroscope,
    plot_pps_mismatch,
    plot_time_differences,
)


def test_cli_help():
    """Test that CLI help works."""
    from click.testing import CliRunner
    from decoder_cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Decode OLA logger data file" in result.output
    assert "--path" in result.output


def test_cli_missing_file():
    """Test that CLI fails gracefully with missing file."""
    from click.testing import CliRunner
    from decoder_cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["-p", "nonexistent.dat"])
    assert result.exit_code != 0


def test_cli_with_real_data():
    """Test CLI with real data file if it exists."""
    from click.testing import CliRunner
    from decoder_cli import main

    test_file = Path("DATA_BOOT_0000_TIME_20260204T193000.dat")
    if not test_file.exists():
        pytest.skip("Real data file not found")

    runner = CliRunner()
    # Mock plt.show() to prevent blocking
    with patch("decoder_cli.plt.show"):
        result = runner.invoke(main, ["-p", str(test_file)])

    assert result.exit_code == 0

    # Clean up generated file
    output_file = test_file.parent / f"{test_file.stem}.npz"
    if output_file.exists():
        output_file.unlink()


def test_plot_functions_with_empty_data(tmp_path):
    """Test that plot functions handle empty data gracefully."""
    import numpy as np

    empty_data = np.array([])

    # These should not crash, just log warnings
    plot_imu_acceleration(empty_data)
    plot_imu_gyroscope(empty_data)
    plot_gnss_coordinates(empty_data)
    plot_time_differences(empty_data, empty_data)
    plot_pps_mismatch(empty_data)
