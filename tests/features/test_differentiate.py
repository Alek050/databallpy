import unittest

import numpy as np
import pandas as pd

from databallpy.features.differentiate import _differentiate, get_velocity
from databallpy.utils.utils import MISSING_INT


class TestDifferentiate(unittest.TestCase):
    def setUp(self):
        self.input = pd.DataFrame(
            {
                "ball_x": [10, 20, -30, 40, np.nan, 60],
                "ball_y": [5, 12, -20, 30, np.nan, 60],
            }
        )
        self.expected_output = pd.DataFrame(
            {
                "ball_x": [10, 20, -30, 40, np.nan, 60],
                "ball_y": [5, 12, -20, 30, np.nan, 60],
                "ball_vx": [np.nan, 10, -50, 70, np.nan, np.nan],
                "ball_vy": [np.nan, 7, -32, 50, np.nan, np.nan],
                "ball_velocity": [
                    np.nan,
                    np.sqrt(149),
                    np.sqrt(2500 + 1024),
                    np.sqrt(4900 + 2500),
                    np.nan,
                    np.nan,
                ],
            }
        )
        self.framerate = 1

    def test_get_velocity(self):
        input_df = self.input.copy()
        output = get_velocity(input_df, ["ball"], self.framerate)
        pd.testing.assert_frame_equal(output, self.expected_output)

        with self.assertRaises(ValueError):
            get_velocity(input_df, ["ball"], self.framerate, filter_type="test")

    def test_differentiate_sg_filter(self):

        output = _differentiate(
            self.input.copy(),
            new_name="velocity",
            metric="",
            frame_rate=self.framerate,
            filter_type="savitzky_golay",
            window=2,
            max_val=MISSING_INT,
            poly_order=1,
            column_ids=["ball"],
        )

        expected_output = pd.DataFrame(
            {
                "ball_x": [10, 20, -30, 40, np.nan, 60],
                "ball_y": [5, 12, -20, 30, np.nan, 60],
                "ball_vx": [np.nan, -20, 10, np.nan, np.nan, np.nan],
                "ball_vy": [np.nan, -12.5, 9.0, np.nan, np.nan, np.nan],
                "ball_velocity": [
                    np.nan,
                    np.sqrt(400 + 12.5**2),
                    np.sqrt(181),
                    np.nan,
                    np.nan,
                    np.nan,
                ],
            }
        )
        pd.testing.assert_frame_equal(output, expected_output)

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
        )

        expected_output = pd.DataFrame(
            {
                "ball_x": [10, 20, -30, 40, np.nan, 60],
                "ball_y": [5, 12, -20, 30, np.nan, 60],
                "ball_vx": [np.nan, np.nan, -20, 0.5, np.nan, np.nan],
                "ball_vy": [np.nan, np.nan, -12.5, 9.0, np.nan, np.nan],
                "ball_velocity": [
                    np.nan,
                    np.nan,
                    np.sqrt(400 + 12.5**2),
                    np.sqrt(81.25),
                    np.nan,
                    np.nan,
                ],
            }
        )
        pd.testing.assert_frame_equal(output, expected_output)
