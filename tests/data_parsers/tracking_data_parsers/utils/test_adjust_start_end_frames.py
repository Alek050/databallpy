import unittest

import numpy as np
import pandas as pd

from databallpy.data_parsers.metadata import Metadata
from databallpy.data_parsers.tracking_data_parsers.utils._adjust_start_end_frames import (
    _adjust_start_end_frames,
    _find_new_start_frame,
)
from databallpy.utils.errors import DataBallPyError
from databallpy.utils.utils import MISSING_INT


class TestAdjustStartEndFrames(unittest.TestCase):
    def setUp(self):
        self.td = pd.DataFrame(
            {
                "frame": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                "period_id": [MISSING_INT, 1, 1, 1, 1, 1, 1, 1, 1],
                "home_1_x": [-10, -10, -10, -10, -10, -10, -10, -10, -10],
                "home_2_x": [-10, -10, -10, -10, -10, -10, -10, -10, -10],
                "home_3_x": [-10, -10, -10, -10, -10, -10, -10, -10, -10],
                "home_1_y": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "home_2_y": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "home_3_y": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "home_4_x": [-10, -10, -10, -10, -10, -10, -10, -10, -10],
                "home_5_x": [-10, -10, -10, -10, -10, -10, -10, -10, -10],
                "home_6_x": [-10, -10, -10, -10, -10, -10, -10, -10, -10],
                "home_4_y": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "home_5_y": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "home_6_y": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "home_7_x": [-10, -10, -10, -10, -10, -10, -10, -10, -10],
                "home_8_x": [-10, -10, -10, -10, -10, -10, -10, -10, -10],
                "home_9_x": [-10, -10, -10, -10, -10, -10, -10, -10, -10],
                "home_7_y": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "home_8_y": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "home_9_y": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "home_10_x": [-10, -10, -10, -10, -10, -10, -10, -10, -10],
                "home_11_x": [-10, -10, -10, -10, -10, -10, -10, -10, -10],
                "home_10_y": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "home_11_y": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "away_1_x": [10, 10, 10, 10, 10, 10, 10, 10, 10],
                "away_2_x": [10, 10, 10, 10, 10, 10, 10, 10, 10],
                "away_3_x": [10, 10, 10, 10, 10, 10, 10, 10, 10],
                "away_1_y": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "away_2_y": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "away_3_y": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "away_4_x": [10, 10, 10, 10, 10, 10, 10, 10, 10],
                "away_5_x": [10, 10, 10, 10, 10, 10, 10, 10, 10],
                "away_6_x": [10, 10, 10, 10, 10, 10, 10, 10, 10],
                "away_4_y": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "away_5_y": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "away_6_y": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "away_7_x": [10, 10, 10, 10, 10, 10, 10, 10, 10],
                "away_8_x": [10, 10, 10, 10, 10, 10, 10, 10, 10],
                "away_9_x": [10, 10, 10, 10, 10, 10, 10, 10, 10],
                "away_7_y": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "away_8_y": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "away_9_y": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "away_10_x": [10, 10, 10, 10, 10, 10, 10, 10, 10],
                "away_11_x": [10, 10, 10, 10, 10, 10, 10, 10, 10],
                "away_10_y": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "away_11_y": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "ball_x": [-10, 0, 0, 0, 0, 0, 0, 0, 0],
                "ball_y": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "ball_z": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "datetime": pd.to_datetime(
                    [
                        "2020-01-01 00:00:00",
                        "2020-01-01 00:00:01",
                        "2020-01-01 00:00:02",
                        "2020-01-01 00:00:03",
                        "2020-01-01 00:00:04",
                        "2020-01-01 00:00:05",
                        "2020-01-01 00:00:06",
                        "2020-01-01 00:00:07",
                        "2020-01-01 00:00:08",
                    ],
                    utc=True,
                ),
            }
        )
        self.home_players_x_columns = [f"home_{x}_x" for x in range(1, 12)]
        self.away_players_x_columns = [f"away_{x}_x" for x in range(1, 12)]
        self.frame_rate = 1
        self.period_id = 1

        self.metadata = Metadata(
            match_id=1,
            pitch_dimensions=[100.0, 50.0],
            periods_frames=pd.DataFrame(
                {
                    "period_id": [1, 2, 3, 4, 5],
                    "start_datetime_td": pd.to_datetime(
                        ["2020-01-01 00:00:00", "NaT", "NaT", "NaT", "NaT"], utc=True
                    ),
                    "end_datetime_td": pd.to_datetime(
                        ["2020-01-01 01:45:00", "NaT", "NaT", "NaT", "NaT"], utc=True
                    ),
                    "start_frame": [
                        2,
                        MISSING_INT,
                        MISSING_INT,
                        MISSING_INT,
                        MISSING_INT,
                    ],
                    "end_frame": [
                        9,
                        MISSING_INT,
                        MISSING_INT,
                        MISSING_INT,
                        MISSING_INT,
                    ],
                }
            ),
            frame_rate=self.frame_rate,
            home_team_id=10,
            home_team_name="Team Home",
            home_players=pd.DataFrame(
                {
                    "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                    "full_name": [
                        "Player 1",
                        "Player 2",
                        "Player 3",
                        "Player 4",
                        "Player 5",
                        "Player 6",
                        "Player 7",
                        "Player 8",
                        "Player 9",
                        "Player 10",
                        "Player 11",
                    ],
                    "shirt_num": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                }
            ),
            home_formation="1442",
            home_score=1,
            away_team_id=11,
            away_team_name="Team Away",
            away_players=pd.DataFrame(
                {
                    "id": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
                    "full_name": [
                        "Player 12",
                        "Player 13",
                        "Player 14",
                        "Player 15",
                        "Player 16",
                        "Player 17",
                        "Player 18",
                        "Player 19",
                        "Player 20",
                        "Player 21",
                        "Player 22",
                    ],
                    "shirt_num": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                }
            ),
            away_formation="1433",
            away_score=0,
            country="Netherlands",
            periods_changed_playing_direction=[],
        )

    def test_adjust_start_end_frames_no_adjustment(self):
        input_td = self.td.copy()
        input_md = self.metadata.copy()
        res_td, res_md = _adjust_start_end_frames(input_td, input_md)

        expected_td = self.td.copy()
        expected_td = expected_td.loc[1:].reset_index(drop=True)
        pd.testing.assert_frame_equal(res_td, expected_td)
        self.assertEqual(res_md, self.metadata)

    def test_adjust_start_end_frames_adjust_invalid_start(self):
        input_td = self.td.copy()
        input_td.loc[1, "ball_x"] = -10  # ball is not in the centre
        input_md = self.metadata.copy()
        res_td, res_md = _adjust_start_end_frames(input_td, input_md)

        expected_td = self.td.copy()
        expected_td = expected_td.loc[2:].reset_index(drop=True)

        expected_md = self.metadata.copy()
        expected_md.periods_frames.loc[0, "start_frame"] = 3
        expected_md.periods_frames.loc[0, "start_datetime_td"] = pd.to_datetime(
            "2020-01-01 00:00:02", utc=True
        )
        pd.testing.assert_frame_equal(res_td, expected_td)
        self.assertEqual(res_md, expected_md)

    def test_adjust_start_end_frames_adjust_no_frame_start(self):
        input_td = self.td.copy()
        input_md = self.metadata.copy()
        input_md.periods_frames.loc[0, "start_frame"] = 12  # not in td
        res_td, res_md = _adjust_start_end_frames(input_td, input_md)

        expected_td = self.td.copy()
        expected_td = expected_td.loc[1:].reset_index(drop=True)

        expected_md = self.metadata.copy()
        expected_md.periods_frames.loc[0, "start_frame"] = 2
        expected_md.periods_frames.loc[0, "start_datetime_td"] = pd.to_datetime(
            "2020-01-01 00:00:01", utc=True
        )
        pd.testing.assert_frame_equal(res_td, expected_td)
        self.assertEqual(res_md, expected_md)

    def test_adjust_start_end_frames_adjust_end(self):
        input_td = self.td.copy()
        input_md = self.metadata.copy()
        input_md.periods_frames.loc[0, "end_frame"] = 12  # not in td
        res_td, res_md = _adjust_start_end_frames(input_td, input_md)

        expected_td = self.td.copy().loc[1:].reset_index(drop=True)
        expected_md = self.metadata.copy()
        expected_md.periods_frames.loc[0, "end_frame"] = 9
        expected_md.periods_frames.loc[0, "end_datetime_td"] = pd.to_datetime(
            "2020-01-01 00:00:08", utc=True
        )

        pd.testing.assert_frame_equal(res_td, expected_td)
        self.assertEqual(res_md, expected_md)

    def test_find_new_start_frame_no_acc(self):
        td = self.td.copy()
        res_frame = _find_new_start_frame(
            td,
            self.period_id,
            self.home_players_x_columns,
            self.away_players_x_columns,
            self.frame_rate,
        )

        self.assertEqual(res_frame, 2)

    def test_find_new_start_frame_with_acc(self):
        td = self.td.copy()
        td.loc[:, "ball_x"] = [0, 0, 0, -6, -11, -11, -11, -11, -11]
        res_frame = _find_new_start_frame(
            td,
            self.period_id,
            self.home_players_x_columns,
            self.away_players_x_columns,
            3,
        )
        self.assertEqual(res_frame, 4)

    def test_find_new_start_no_period(self):
        with self.assertRaises(DataBallPyError):
            _find_new_start_frame(
                self.td,
                2,
                self.home_players_x_columns,
                self.away_players_x_columns,
                self.frame_rate,
            )

    def test_find_new_start_no_valid_options(self):
        td = self.td.copy()
        td.loc[:, "ball_x"] = [-11, -11, -11, -11, -11, -11, -11, -11, -11]
        td.loc[0, "period_id"] = 1
        res_frame = _find_new_start_frame(
            td,
            self.period_id,
            self.home_players_x_columns,
            self.away_players_x_columns,
            self.frame_rate,
        )
        self.assertEqual(res_frame, 1)

    def test_find_new_start_one_valid_options(self):
        td = self.td.copy()
        td.loc[:, self.away_players_x_columns] *= -1
        td.loc[6, self.away_players_x_columns] = 10
        res_frame = _find_new_start_frame(
            td,
            self.period_id,
            self.home_players_x_columns,
            self.away_players_x_columns,
            self.frame_rate,
        )
        self.assertEqual(res_frame, 7)

    def test_find_new_start_frame_non_values(self):
        td = self.td.copy()
        td.loc[[1, 2, 3], ["home_1_x", "home_2_x", "home_1_y", "home_2_y"]] = np.nan
        res_frame = _find_new_start_frame(
            td,
            self.period_id,
            self.home_players_x_columns,
            self.away_players_x_columns,
            self.frame_rate,
        )
        self.assertEqual(res_frame, 5)
