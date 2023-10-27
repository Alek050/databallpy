import unittest

import pandas as pd

from databallpy.load_data.tracking_data.utils import (
    _normalize_playing_direction_tracking,
)
from databallpy.utils.utils import MISSING_INT


class TestNormalizePlayingDirection(unittest.TestCase):
    def setUp(self) -> None:
        self.td = pd.DataFrame(
            {
                "frame": [10, 11, 12, 13, 14, 15],
                "ball_x": [0, 1, 2, 3, 4, 5],
                "ball_y": [0, -1, 2, 33, -13, 12],
                "home_1_x": [-5, -4, 22, 21, -6, 7],
                "home_1_y": [22, 23, 33, -12, -12, -12],
                "home_2_x": [-6, -14, 13, 21, 3, 7],
                "home_2_y": [-22, 23, 0, -0, -12, -1],
                "away_1_x": [8, 8, 8, 8, 8, 8],
                "away_1_y": [9, 9, 9, 9, 9, 9],
                "away_12_x": [-1, 22, 23, 46, 11, -10],
                "away_12_y": [20, -20, -24, 21, 33, -15],
            }
        )

        self.periods = pd.DataFrame(
            {
                "period_id": [1, 2, 3, 4, 5],
                "start_frame": [10, 12, 14, MISSING_INT, MISSING_INT],
                "end_frame": [11, 13, 15, MISSING_INT, MISSING_INT],
            }
        )

    def test_normalize_playing_direction(self):
        expected = pd.DataFrame(
            {
                "frame": [10, 11, 12, 13, 14, 15],
                "ball_x": [0, 1, -2, -3, 4, 5],
                "ball_y": [0, -1, -2, -33, -13, 12],
                "home_1_x": [-5, -4, -22, -21, -6, 7],
                "home_1_y": [22, 23, -33, 12, -12, -12],
                "home_2_x": [-6, -14, -13, -21, 3, 7],
                "home_2_y": [-22, 23, -0, 0, -12, -1],
                "away_1_x": [8, 8, -8, -8, 8, 8],
                "away_1_y": [9, 9, -9, -9, 9, 9],
                "away_12_x": [-1, 22, -23, -46, 11, -10],
                "away_12_y": [20, -20, 24, -21, 33, -15],
            }
        )

        res_td, res_periods = _normalize_playing_direction_tracking(self.td, self.periods)

        pd.testing.assert_frame_equal(res_td, expected)
        assert res_periods == [2]
