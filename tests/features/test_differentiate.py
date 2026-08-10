import unittest

import numpy as np
import pandas as pd

from databallpy.features.differentiate import _differentiate
from databallpy.utils.warnings import DataBallPyWarning


class TestDifferentiate(unittest.TestCase):
    def setUp(self):
        self.input = pd.DataFrame(
            {
                "home_1_x": [10, 20, -30, 40, np.nan, 60],
                "home_1_y": [5, 12, -20, 30, np.nan, 60],
                "home_1_vx": [1, 2, 5, 1, np.nan, -3.0],
                "home_1_vy": [1, 2, -5, 1, np.nan, 1.0],
                "home_1_velocity": [
                    np.sqrt(2),
                    np.sqrt(8),
                    np.sqrt(25),
                    np.sqrt(2),
                    np.nan,
                    np.sqrt(10),
                ],
            }
        )
        self.framerate = 1

    def test_differentiate_sg_filter(self):
        input = self.input.copy()
        output1 = _differentiate(
            input,
            new_name="velocity",
            metric="",
            frame_rate=self.framerate,
            filter_type="savitzky_golay",
            window=2,
            max_val=np.nan,
            poly_order=1,
            column_ids=["home_1"],
            inplace=False,
            allow_overwrite=True,
        )

        expected_output = pd.DataFrame(
            {
                "home_1_x": [10, 20, -30, 40, np.nan, 60],
                "home_1_y": [5, 12, -20, 30, np.nan, 60],
                "home_1_vx": [10.0, -5.0, 10.0, np.nan, 10.0, np.nan],
                "home_1_vy": [7.0, -1.75, 10.5, np.nan, 15.0, np.nan],
                "home_1_velocity": [
                    np.sqrt(149),
                    np.sqrt(25 + 1.75**2),
                    np.sqrt(100 + 10.5**2),
                    np.nan,
                    np.sqrt(100 + 225),
                    np.nan,
                ],
            }
        )
        pd.testing.assert_frame_equal(output1, expected_output)

        with self.assertWarns(DataBallPyWarning):
            output2 = _differentiate(
                input,
                new_name="velocity",
                metric="",
                frame_rate=self.framerate,
                filter_type="savitzky_golay",
                window=2,
                max_val=np.nan,
                poly_order=1,
                column_ids=["home_1"],
                inplace=False,
                allow_overwrite=False,
            )
        pd.testing.assert_frame_equal(output2, input)

    def test_differentiate_ma_filter(self):
        output = _differentiate(
            self.input.copy(),
            new_name="velocity",
            metric="",
            frame_rate=self.framerate,
            filter_type="moving_average",
            window=2,
            max_val=51,
            poly_order=1,
            column_ids=None,
            allow_overwrite=True,
        )

        expected_output = pd.DataFrame(
            {
                "home_1_x": [10, 20, -30, 40, np.nan, 60],
                "home_1_y": [5, 12, -20, 30, np.nan, 60],
                "home_1_vx": [5.0, -5.0, -5.0, np.nan, np.nan, np.nan],
                "home_1_vy": [3.5, -2.75, -1.75, np.nan, np.nan, np.nan],
                "home_1_velocity": [
                    np.sqrt(25 + 3.5**2),
                    np.sqrt(25 + 2.75**2),
                    np.sqrt(25 + 1.75**2),
                    np.nan,
                    np.nan,
                    np.nan,
                ],
            }
        )
        pd.testing.assert_frame_equal(output, expected_output)

    def test_differentiate_wrong_input(self):
        with self.assertRaises(KeyError):
            _differentiate(
                self.input.copy(),
                new_name="velocity",
                metric="a",
                frame_rate=self.framerate,
                filter_type="savitzky_golay",
                window=2,
                max_val=np.nan,
                poly_order=1,
                column_ids=["home_1"],
            )
