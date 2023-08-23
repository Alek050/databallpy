import unittest

import numpy as np

from databallpy.utils.filters import filter_data


class TestFilters(unittest.TestCase):
    def test_filter_data_input_types(self):
        # Test that filter_data returns a numpy array
        array = np.array([1, 2, 3])

        # Test that filter_data raises an error for non-numpy arrays
        with self.assertRaises(TypeError):
            filter_data([1, 2, 3])

        # Test that filter_data raises an error for invalid filter types
        with self.assertRaises(ValueError):
            filter_data(array, filter_type="invalid")

        # Test that filter_data raises an error for non-integer window lengths
        with self.assertRaises(TypeError):
            filter_data(array, window_length=3.5)

        # Test that filter_data raises an error for non-integer polyorders
        with self.assertRaises(TypeError):
            filter_data(array, filter_type="savitzky_golay", polyorder=2.5)

        # Test that filter_data raises an error for short arrays
        with self.assertRaises(ValueError):
            filter_data(np.array([1, 2]), window_length=3)

    def test_filter_data_ma(self):
        arr = np.array([1, 2, 3, 4, 5])
        expected_output = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        self.assertTrue(
            np.allclose(
                filter_data(arr, filter_type="moving_average", window_length=2),
                expected_output,
            )
        )

    def test_filter_data_sg(self):
        # Test that filter_data returns the expected output for a Savitzky-Golay filter
        arr = np.array([1, 3, 3, 4, 5])
        expected_output = np.array([1.333333333, 2.3333333333, 3.33333333, 4.0, 5.0])
        self.assertTrue(
            np.allclose(
                filter_data(
                    arr, filter_type="savitzky_golay", window_length=3, polyorder=1
                ),
                expected_output,
            )
        )
