import unittest

import numpy as np
import pandas as pd

from databallpy.load_data.metadata import Metadata
from databallpy.load_data.tracking_data.tracab import (
    _get_meta_data,
    _get_tracking_data,
    load_tracking_data_tracab,
)


class TestTracab(unittest.TestCase):
    def setUp(self):
        self.tracking_data_loc = "tests/test_data/tracab_td_test.dat"
        self.metadata_loc = "tests/test_data/tracab_metadata_test.xml"
        self.expected_metadata = Metadata(
            match_id=1908,
            pitch_dimensions=[100, 50],
            match_start_datetime=np.datetime64("2023-01-14 13:17:00"),
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
            home_score=None,
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
            away_score=None,
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
                "ball_speed": [0.296, 0.289, 0.294, np.nan, 0.294],
                "ball_status": ["Alive", "Alive", "Alive", np.nan, "Alive"],
                "ball_posession": ["A", "A", "A", np.nan, "A"],
                "away_34_x": [-13.50, -13.50, -13.50, np.nan, -13.49],
                "away_34_y": [-4.75, -4.74, -4.73, np.nan, -4.72],
                "away_34_speed": [0.002, 0.001, 0.000, np.nan, 0.000],
                "home_17_x": [1.22, 1.21, 1.21, np.nan, 1.21],
                "home_17_y": [-13.16, -13.16, -13.17, np.nan, -13.18],
                "home_17_speed": [0.002, 0.002, 0.002, np.nan, 0.002],
            }
        )

    def test_get_metadata(self):

        meta_data = _get_meta_data(self.metadata_loc)
        assert meta_data == self.expected_metadata

    def test_get_tracking_data(self):
        tracking_data = _get_tracking_data(self.tracking_data_loc, verbose=True).fillna(
            1000
        )
        assert tracking_data.equals(self.expected_tracking_data.fillna(1000))

    def test_load_tracking_data(self):
        tracking_data, metadata = load_tracking_data_tracab(
            self.tracking_data_loc, self.metadata_loc, verbose=True
        )
        tracking_data = tracking_data.fillna(1000)

        assert all(
            [
                metadata == self.expected_metadata,
                tracking_data.equals(self.expected_tracking_data.fillna(1000)),
            ]
        )
