import unittest

import numpy as np
import pandas as pd

from databallpy.features.differentiate import (
    _differentiate,
    add_acceleration,
    add_velocity,
)


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
        self.expected_output_vel = pd.DataFrame(
            {
                "home_1_x": [10, 20, -30, 40, np.nan, 60],
                "home_1_y": [5, 12, -20, 30, np.nan, 60],
                "home_1_vx": [10.0, -20.0, 10.0, np.nan, 10.0, np.nan],
                "home_1_vy": [7.0, -12.5, 9.0, np.nan, 15.0, np.nan],
                "home_1_velocity": [
                    np.sqrt(149),
                    np.sqrt(400 + 12.5**2),
                    np.sqrt(181),
                    np.nan,
                    np.sqrt(325),
                    np.nan,
                ],
            }
        )
        self.framerate = 1

        self.expected_output_acc = pd.DataFrame(
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
                "home_1_ax": [1.0, 2.0, -0.5, np.nan, -2.0, np.nan],
                "home_1_ay": [1.0, -3, -0.5, np.nan, 0.0, np.nan],
                "home_1_acceleration": [
                    np.sqrt(2),
                    np.sqrt(13),
                    np.sqrt(0.5),
                    np.nan,
                    np.sqrt(4),
                    np.nan,
                ],
            },
        )

    def test_get_velocity(self):
        input_df = self.input.copy()
        output = add_velocity(input_df, ["home_1"], self.framerate)
        pd.testing.assert_frame_equal(output, self.expected_output_vel)

        with self.assertRaises(ValueError):
            add_velocity(input_df, ["home_1"], self.framerate, filter_type="test")

    def test_get_acceleration(self):
        input_df = self.input.copy()
        input_df.drop(columns=["home_1_vx"], inplace=True)
        with self.assertRaises(ValueError):
            add_acceleration(input_df, ["home_1"], self.framerate)

        input_df = self.input.copy()
        output = add_acceleration(input_df, "home_1", self.framerate)
        pd.testing.assert_frame_equal(output, self.expected_output_acc)

        with self.assertRaises(ValueError):
            add_acceleration(input_df, ["home_1"], self.framerate, filter_type="wrong")

    def test_differentiate_sg_filter(self):
        output = self.input.copy()
        _differentiate(
            output,
            new_name="velocity",
            metric="",
            frame_rate=self.framerate,
            filter_type="savitzky_golay",
            window=2,
            max_val=np.nan,
            poly_order=1,
            column_ids=["home_1"],
            inplace=True,
        )

        expected_output = pd.DataFrame(
            {
                "home_1_x": [10, 20, -30, 40, np.nan, 60],
                "home_1_y": [5, 12, -20, 30, np.nan, 60],
                "home_1_vx": [10.0, -5.0, np.nan, np.nan, np.nan, np.nan],
                "home_1_vy": [7.0, -1.75, np.nan, np.nan, np.nan, np.nan],
                "home_1_velocity": [
                    np.sqrt(149),
                    np.sqrt(25 + 1.75**2),
                    np.nan,
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
