import unittest

import numpy as np
import pandas as pd

from databallpy.load_data.metadata import Metadata
from databallpy.load_data.tracking_data._add_ball_data_to_dict import (
    _add_ball_data_to_dict,
)
from databallpy.load_data.tracking_data._add_player_data_to_dict import (
    _add_player_data_to_dict,
)
from databallpy.load_data.tracking_data._insert_missing_rows import _insert_missing_rows
from databallpy.load_data.tracking_data.tracab import (
    _get_metadata,
    _get_player_data,
    _get_tracking_data,
    load_tracab_tracking_data,
)


class TestTracab(unittest.TestCase):
    def setUp(self):
        self.tracking_data_loc = "tests/test_data/tracab_td_test.dat"
        self.metadata_loc = "tests/test_data/tracab_metadata_test.xml"
        self.expected_metadata = Metadata(
            match_id=1908,
            pitch_dimensions=[100, 50],
            periods_frames=pd.DataFrame(
                {
                    "period": [1, 2, 3, 4, 5],
                    "start_frame": [100, 200, 300, 400, 0],
                    "end_frame": [400, 600, 900, 1200, 0],
                }
            ),
            frame_rate=25,
            home_team_id=3,
            home_team_name="TeamOne",
            home_formation=None,
            home_score=np.nan,
            home_players=pd.DataFrame(
                {
                    "id": [19367, 45849],
                    "full_name": ["Piet Schrijvers", "Jan Boskamp"],
                    "shirt_num": [1, 2],
                    "start_frame": [100, 100],
                    "end_frame": [1200, 400],
                }
            ),
            away_team_id=194,
            away_team_name="TeamTwo",
            away_formation=None,
            away_score=np.nan,
            away_players=pd.DataFrame(
                {
                    "id": [184934, 450445],
                    "full_name": ["Pepijn Blok", "Test Speler"],
                    "shirt_num": [1, 2],
                    "start_frame": [100, 100],
                    "end_frame": [1200, 400],
                }
            ),
        )

        self.expected_tracking_data = pd.DataFrame(
            {
                "timestamp": [1509993, 1509994, 1509995, 1509996, 1509997],
                "ball_x": [1.50, 1.81, 2.13, np.nan, 2.76],
                "ball_y": [-0.43, -0.49, -0.56, np.nan, -0.70],
                "ball_z": [0.07, 0.09, 0.11, np.nan, 0.15],
                "ball_status": ["Alive", "Dead", "Alive", np.nan, "Alive"],
                "ball_posession": ["Away", "Away", "Away", np.nan, "Home"],
                "home_34_x": [-13.50, -13.50, -13.50, np.nan, -13.49],
                "home_34_y": [-4.75, -4.74, -4.73, np.nan, -4.72],
                "away_17_x": [1.22, 1.21, 1.21, np.nan, 1.21],
                "away_17_y": [-13.16, -13.16, -13.17, np.nan, -13.18],
                "match_time_td": ["Break", "Break", "Break", "Break", "Break"],
            }
        )

    def test_get_metadata(self):

        meta_data = _get_metadata(self.metadata_loc)
        assert meta_data == self.expected_metadata

    def test_get_tracking_data(self):
        tracking_data = _get_tracking_data(self.tracking_data_loc, verbose=True)
        expected_tracking_data = self.expected_tracking_data.drop(
            "match_time_td", axis=1
        ).copy()
        pd.testing.assert_frame_equal(tracking_data, expected_tracking_data)

    def test_load_tracking_data(self):
        tracking_data, metadata = load_tracab_tracking_data(
            self.tracking_data_loc, self.metadata_loc, verbose=False
        )

        assert metadata == self.expected_metadata
        pd.testing.assert_frame_equal(tracking_data, self.expected_tracking_data)

    def test_get_player_data(self):
        input_players_info = [
            {
                "PlayerId": "1234",
                "FirstName": "Bart",
                "LastName": "Bakker",
                "JerseyNo": "4",
                "StartFrameCount": "1212",
                "EndFrameCount": "2323",
            },
            {
                "PlayerId": "1235",
                "FirstName": "Bram",
                "LastName": "Slager",
                "JerseyNo": "3",
                "StartFrameCount": "1218",
                "EndFrameCount": "2327",
            },
        ]

        expected_df_players = pd.DataFrame(
            {
                "id": [1234, 1235],
                "full_name": ["Bart Bakker", "Bram Slager"],
                "shirt_num": [4, 3],
                "start_frame": [1212, 1218],
                "end_frame": [2323, 2327],
            }
        )
        df_players = _get_player_data(input_players_info)
        pd.testing.assert_frame_equal(df_players, expected_df_players)
