import unittest

import numpy as np
import pandas as pd

from databallpy.features.filters import _filter_data, filter_tracking_data


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
        expected_output = np.array([1.333333333, 2.3333333333, 3.33333333, 4.0, 5.0])
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
                "ball_x": [20, 0, 10, np.nan, np.nan, np.nan],
                "ball_y": [11.5, -1, 7.333333, np.nan, np.nan, np.nan],
                "away_13_x": [10, 20, -30, 40, np.nan, 60],
                "away_13_y": [5, 12, -20, 30, np.nan, 60],
            }
        )

        pd.testing.assert_frame_equal(filtered_data, expected_output)
