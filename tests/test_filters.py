import unittest

import numpy as np
import pandas as pd

from databallpy.filters import filter_data


class TestFilters(unittest.TestCase):
    def setUp(self):
        self.input = pd.DataFrame(
            {
                "col_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "col_2": [5, 6, -1, -9, 10, 11, 12, -50, 1, 3],
            }
        )

    def test_moving_average_filter(self):
        expected_output_1 = pd.DataFrame(
            {
                "col_1": [np.nan, np.nan, np.nan, np.nan, 3, 4, 5, 6, 7, 8],
                "col_2": [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    2.2,
                    3.4,
                    4.6,
                    -5.2,
                    -3.2,
                    -4.6,
                ],
            }
        )

        output_1 = filter_data(self.input, ["col_1", "col_2"], kind="moving_average")
        pd.testing.assert_frame_equal(expected_output_1, output_1)

    def test_savitzky_golay_filter(self):
        expected_output_2 = pd.DataFrame(
            {
                "col_1": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "col_2": [
                    8.2,
                    0.2,
                    -2.8,
                    -2.742857,
                    4.6,
                    17.942857,
                    -8.485714,
                    -21.028571,
                    -15.514286,
                    6.428571,
                ],
            }
        )

        output_2 = filter_data(self.input, ["col_1", "col_2"], kind="savitzky_golay")
        pd.testing.assert_frame_equal(expected_output_2, output_2)

    def test_single_column_filter(self):
        expected_output_3 = pd.DataFrame(
            {
                "col_1": [np.nan, np.nan, np.nan, np.nan, 3, 4, 5, 6, 7, 8],
                "col_2": [5, 6, -1, -9, 10, 11, 12, -50, 1, 3],
            }
        )

        output_3 = filter_data(self.input, ["col_1"], kind="moving_average")
        pd.testing.assert_frame_equal(expected_output_3, output_3)

    def test_different_window_moving_average(self):
        expected_output_4 = pd.DataFrame(
            {
                "col_1": [np.nan, np.nan, np.nan, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
                "col_2": [
                    np.nan,
                    np.nan,
                    np.nan,
                    0.25,
                    1.50,
                    2.75,
                    6,
                    -4.25,
                    -6.5,
                    -8.5,
                ],
            }
        )

        output_4 = filter_data(
            self.input, ["col_1", "col_2"], kind="moving_average", window_length=4
        )
        pd.testing.assert_frame_equal(expected_output_4, output_4)

    def test_filters_wrong_input(self):

        with self.assertRaises(TypeError) as cm:
            filter_data({"key": "value"}, ["key"], kind="moving_average")
        self.assertEqual(
            str(cm.exception), "df should be a pd.DataFrame, not a <class 'dict'>"
        )

        with self.assertRaises(TypeError) as cm:
            filter_data(self.input, "col_1", kind="moving_average")
        self.assertEqual(
            str(cm.exception), "input_columns should be a list, not a <class 'str'>"
        )

        with self.assertRaises(TypeError) as cm:
            filter_data(self.input, ["col_1", 2], kind="moving_average")
        self.assertEqual(
            str(cm.exception), "All elements of input_columns should be strings"
        )

        with self.assertRaises(TypeError) as cm:
            filter_data(self.input, ["col_1", "col_2"], kind=5)
        self.assertEqual(
            str(cm.exception), "kind should be a string, not a <class 'int'>"
        )

        with self.assertRaises(TypeError) as cm:
            filter_data(self.input, ["col_1", "col_2"], kind="magic_filter")
        self.assertEqual(
            str(cm.exception),
            "kind should be one of moving_average, savitzky_golay, not magic_filter",
        )

        with self.assertRaises(TypeError) as cm:
            filter_data(
                self.input, ["col_1", "col_2"], kind="moving_average", window_length="5"
            )
        self.assertEqual(
            str(cm.exception), "window_length should be an int, not a <class 'str'>"
        )
