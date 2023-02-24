import unittest

import numpy as np
import pandas as pd

from databallpy.load_data.metadata import Metadata
from databallpy.load_data.metrica_metadata import (
    _get_metadata,
    _get_td_channels,
    _update_metadata,
)


class TestMetricaMetadata(unittest.TestCase):
    def setUp(self):
        self.md_loc = "tests/test_data/metrica_metadata_test.xml"
        self.expected_metadata = Metadata(
            match_id=9999,
            pitch_dimensions=[100.0, 50.0],
            periods_frames=pd.DataFrame(
                {
                    "period": [1, 2, 3, 4, 5],
                    "start_frame": [1, 4, np.nan, np.nan, np.nan],
                    "end_frame": [3, 6, np.nan, np.nan, np.nan],
                    "start_time_td": [
                        pd.to_datetime("2019-02-21T03:30:07.000Z"),
                        pd.to_datetime("2019-02-21T03:30:08.500Z"),
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    "end_time_td": [
                        pd.to_datetime("2019-02-21T03:30:08.000Z"),
                        pd.to_datetime("2019-02-21T03:30:09.500Z"),
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                }
            ),
            frame_rate=2,
            home_team_id="FIFATMA",
            home_team_name="Team A",
            home_players=pd.DataFrame(
                {
                    "id": [3578, 3568],
                    "full_name": ["Player 11", "Player 1"],
                    "formation_place": [0, 1],
                    "position": ["Goalkeeper", "Right Back"],
                    "starter": [True, True],
                    "shirt_num": [11, 1],
                }
            ),
            home_formation="1100",
            home_score=0,
            away_team_id="FIFATMB",
            away_team_name="Team B",
            away_players=pd.DataFrame(
                {
                    "id": [3699, 3700],
                    "full_name": ["Player 34", "Player 35"],
                    "formation_place": [3, 1],
                    "position": ["Left Forward (2)", "Left Back"],
                    "starter": [True, False],
                    "shirt_num": [34, 35],
                }
            ),
            away_formation="0001",
            away_score=2,
        )
        self.expected_td_channels = pd.DataFrame(
            {
                "start": [1, 4],
                "end": [3, 6],
                "ids": [
                    ["home_1", "home_11", "away_34"],
                    ["home_11", "away_34", "away_35"],
                ],
            }
        )

    def test_get_metadata(self):
        expected_metadata = self.expected_metadata.copy()
        expected_metadata.home_formation = ""
        expected_metadata.away_formation = ""
        expected_metadata.home_players["starter"] = np.nan
        expected_metadata.away_players["starter"] = np.nan
        assert _get_metadata(self.md_loc) == expected_metadata

    def test_get_td_channels(self):
        input_metadata = self.expected_metadata.copy()
        input_metadata.home_formation = ""
        input_metadata.away_formation = ""
        input_metadata.home_players["starter"] = np.nan
        input_metadata.away_players["starter"] = np.nan
        res = _get_td_channels(self.md_loc, input_metadata)
        pd.testing.assert_frame_equal(res, self.expected_td_channels)

    def test_update_metadata(self):
        input_metadata = self.expected_metadata.copy()
        input_metadata.home_formation = ""
        input_metadata.away_formation = ""
        input_metadata.home_players["starter"] = np.nan
        input_metadata.away_players["starter"] = np.nan

        res = _update_metadata(self.expected_td_channels, input_metadata)

        assert res == self.expected_metadata
