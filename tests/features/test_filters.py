import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd

from databallpy.features.filters import _filter_data, filter_tracking_data, _savgol_with_nan_compat
import warnings

import pytest
from numpy.testing import assert_allclose

from scipy.signal import savgol_filter



class TestFilters(unittest.TestCase):
    def test_filter_data_input_types(self):
        # Test that filter_data returns a numpy array
        array = np.array([1, 2, 3])

        # Test that filter_data raises an error for non-numpy arrays
        with self.assertRaises(TypeError):
            _filter_data([1, 2, 3])

        # Test that filter_data raises an error for invalid filter types
        with self.assertRaises(ValueError):
            _filter_data(array, filter_type="invalid")

        # Test that filter_data raises an error for non-integer window lengths
        with self.assertRaises(TypeError):
            _filter_data(array, window_length=3.5)

        # Test that filter_data raises an error for non-integer polyorders
        with self.assertRaises(TypeError):
            _filter_data(array, filter_type="savitzky_golay", polyorder=2.5)

        # Test that filter_data raises an error for short arrays
        with self.assertRaises(ValueError):
            _filter_data(np.array([1, 2]), window_length=3)

    def test_filter_data_ma(self):
        arr = np.array([1, 2, 3, 4, 5])
        expected_output = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        self.assertTrue(
            np.allclose(
                _filter_data(arr, filter_type="moving_average", window_length=2),
                expected_output,
            )
        )

    def test_filter_data_sg(self):
        # Test that filter_data returns the expected output for a Savitzky-Golay filter
        arr = np.array([1, 3, 3, 4, 5])
        expected_output = np.array([1.33, 2.33, 3.33, 4.0, 5.0])
        self.assertTrue(
            np.allclose(
                _filter_data(
                    arr, filter_type="savitzky_golay", window_length=3, polyorder=1
                ),
                expected_output,
            )
        )

    def test_filter_tracking_data_input_types(self):
        tracking_data = pd.DataFrame(
            {
                "home_1_x": [10, 20, -30, 40, np.nan, 60],
                "home_1_y": [5, 12, -20, 30, np.nan, 60],
                "ball_x": [10, 20, -30, 40, np.nan, 60],
                "ball_y": [5, 12, -20, 30, np.nan, 60],
                "away_13_x": [10, 20, -30, 40, np.nan, 60],
                "away_13_y": [5, 12, -20, 30, np.nan, 60],
            }
        )

        with self.assertRaises(TypeError):
            filter_tracking_data("tracking_data", column_ids="home_1")
        with self.assertRaises(ValueError):
            filter_tracking_data(tracking_data, column_ids=[])
        with self.assertRaises(TypeError):
            filter_tracking_data(tracking_data, column_ids="home_1", inplace="True")
        with self.assertRaises(TypeError):
            filter_tracking_data(tracking_data, column_ids="home_1", window_length=3.5)
        with self.assertRaises(TypeError):
            filter_tracking_data(tracking_data, column_ids="home_1", polyorder=2.5)
        with self.assertRaises(ValueError):
            filter_tracking_data(
                tracking_data, column_ids="home_1", filter_type="invalid"
            )

    @patch(
        "databallpy.features.filters.savgol_filter",
        side_effect=TypeError("Mocked ValueError"),
    )
    @patch("numpy.convolve", side_effect=ValueError("Mocked ValueError"))
    def test_filter_data_error(self, mock_savgol_filter, mock_convolve):
        arr = np.array([1, 2, 3, 8, 5])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _filter_data(arr, "savitzky_golay", window_length=3)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, UserWarning))
            self.assertIn(
                "An unexpected error occurred while filtering the data",
                str(w[-1].message),
            )
            np.testing.assert_array_equal(result, arr)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _filter_data(arr, "moving_average", window_length=3)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, UserWarning))
            self.assertIn(
                "An unexpected error occurred while filtering the data",
                str(w[-1].message),
            )
            np.testing.assert_array_equal(result, arr)

    def test_filter_tracking_data_ma_inplace(self):
        tracking_data = pd.DataFrame(
            {
                "home_1_x": [10, 20, -30, 40, np.nan, 60],
                "home_1_y": [5, 12, -20, 30, np.nan, 60],
                "ball_x": [10, 20, -30, 40, np.nan, 60],
                "ball_y": [5, 12, -20, 30, np.nan, 60],
                "away_13_x": [10, 20, -30, 40, np.nan, 60],
                "away_13_y": [5, 12, -20, 30, np.nan, 60],
            }
        )

        filter_tracking_data(
            tracking_data,
            column_ids=["ball"],
            filter_type="moving_average",
            window_length=2,
            inplace=True,
        )

        expected_output = pd.DataFrame(
            {
                "home_1_x": [10, 20, -30, 40, np.nan, 60],
                "home_1_y": [5, 12, -20, 30, np.nan, 60],
                "ball_x": [5, 15, -5, 5, np.nan, np.nan],
                "ball_y": [2.5, 8.5, -4, 5, np.nan, np.nan],
                "away_13_x": [10, 20, -30, 40, np.nan, 60],
                "away_13_y": [5, 12, -20, 30, np.nan, 60],
            }
        )
        pd.testing.assert_frame_equal(tracking_data, expected_output)

    def test_filter_tracking_data_sg_not_inplace(self):
        tracking_data = pd.DataFrame(
            {
                "home_1_x": [10, 20, -30, 40, np.nan, 60],
                "home_1_y": [5, 12, -20, 30, np.nan, 60],
                "ball_x": [10, 20, -30, 40, np.nan, 60],
                "ball_y": [5, 12, -20, 30, np.nan, 60],
                "away_13_x": [10, 20, -30, 40, np.nan, 60],
                "away_13_y": [5, 12, -20, 30, np.nan, 60],
            }
        )

        filtered_data = filter_tracking_data(
            tracking_data,
            column_ids=["ball"],
            filter_type="savitzky_golay",
            window_length=3,
            polyorder=1,
            inplace=False,
        )

        assert filtered_data is not tracking_data

        expected_output = pd.DataFrame(
            {
                "home_1_x": [10, 20, -30, 40, np.nan, 60],
                "home_1_y": [5, 12, -20, 30, np.nan, 60],
                "ball_x": [20, 0, 10, 20., np.nan, 60.],
                "ball_y": [11.5, -1, 7.33, 18.33, np.nan, 60.],
                "away_13_x": [10, 20, -30, 40, np.nan, 60],
                "away_13_y": [5, 12, -20, 30, np.nan, 60],
            }
        )

        pd.testing.assert_frame_equal(filtered_data, expected_output)



class TestSavgolWithNanCompat:
    def test_no_nans_behaves_like_plain_savgol(self) -> None:
        """When there are no NaNs, the wrapper should behave like savgol_filter.

        It should:
        - not modify NaNs (there are none),
        - not emit warnings,
        - produce the same output (up to the .round(2) in the implementation).
        """
        rng = np.random.default_rng(42)
        arr = rng.normal(size=20)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("error")  # fail on unexpected warnings
            result = _savgol_with_nan_compat(arr, window_length=5, polyorder=2)

        assert len(w) == 0, "No warnings expected when there are no NaNs"

        expected = savgol_filter(arr.astype(float), 5, 2, mode="interp").round(2)
        assert result.shape == arr.shape
        assert_allclose(result, expected, rtol=0, atol=0)  # exact after rounding

    def test_with_nans_preserves_nan_positions(self) -> None:
        """NaNs should be preserved in their original positions after filtering.

        Finite values should be smoothed; NaN positions remain NaN.
        """
        arr = np.array([1.0, -1.0, np.nan, 4.0, 5.0, np.nan, 7.0], dtype=float)

        result = _savgol_with_nan_compat(arr, window_length=5, polyorder=2)

        assert result.shape == arr.shape

        # NaN mask should be identical
        assert np.array_equal(np.isnan(result), np.isnan(arr)), \
            "NaN positions must be preserved"

        # At least one finite value should be changed by the smoothing
        finite_mask = np.isfinite(arr)
        assert np.any(result[finite_mask] != arr[finite_mask])

    def test_with_nans_does_not_raise_or_warn_when_enough_valid_samples(self) -> None:
        """If there are enough finite samples, the function should not warn or error."""
        arr = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, np.nan, 8.0], dtype=float)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _savgol_with_nan_compat(arr, window_length=5, polyorder=2)

        # No warnings for normal NaN handling when enough valid data exists
        assert len(w) == 0, "No warnings expected when there are enough finite samples"
        assert result.shape == arr.shape

    def test_too_few_valid_samples_returns_original_and_warns(self) -> None:
        """When there are not enough finite samples, original data should be returned.

        The function should emit a warning but not raise.
        """
        # Only 2 valid points, but window_length=5 and polyorder=2 → not enough
        arr = np.array([np.nan, 1.0, np.nan, np.nan, 2.0, np.nan], dtype=float)

        with pytest.warns(UserWarning, match="Not enough finite samples"):
            result = _savgol_with_nan_compat(arr, window_length=5, polyorder=2)

        # Should be exactly the original array (including NaNs)
        assert_allclose(result, arr, equal_nan=True)

    def test_all_nans_returns_original_and_warns(self) -> None:
        """All-NaN input should just be returned, with a warning."""
        arr = np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)

        with pytest.warns(UserWarning, match="Not enough finite samples"):
            result = _savgol_with_nan_compat(arr, window_length=3, polyorder=1)

        assert_allclose(result, arr, equal_nan=True)

    def test_mode_argument_is_respected(self) -> None:
        """Ensure that the 'mode' argument is passed through to savgol_filter."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

        res_interp = _savgol_with_nan_compat(arr, window_length=5, polyorder=2, mode="interp")
        res_mirror = _savgol_with_nan_compat(arr, window_length=5, polyorder=2, mode="mirror")

        # Different mode should generally yield different results
        assert not np.allclose(res_interp, res_mirror)

    def test_input_not_modified_in_place(self) -> None:
        """Verify that the input array is not modified in place."""
        arr = np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=float)
        arr_copy = arr.copy()

        _ = _savgol_with_nan_compat(arr, window_length=5, polyorder=2)

        assert_allclose(arr, arr_copy, equal_nan=True)