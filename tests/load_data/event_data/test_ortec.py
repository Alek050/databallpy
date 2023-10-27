import unittest

import numpy as np
import pandas as pd

from databallpy.load_data.event_data.ortec import (
    _get_formation,
    get_player_info,
    load_metadata,
    load_ortec_event_data,
)
from databallpy.load_data.metadata import Metadata
from databallpy.utils.utils import MISSING_INT
from databallpy.warnings import DataBallPyWarning


class TestOrtec(unittest.TestCase):
    def setUp(self):
        self.meta_data_loc = "tests/test_data/ortec_metadata_test.json"
        periods = pd.DataFrame(
            {
                "period_id": [1, 2, 3, 4, 5],
                "start_datetime_ed": pd.to_datetime(
                    [
                        "2023-09-03 20:00:00+02:00",
                        "2023-09-03 21:00:00+02:00",
                        "NaT",
                        "NaT",
                        "NaT",
                    ]
                ),
                "end_datetime_ed": pd.to_datetime(
                    [
                        "2023-09-03 20:45:00+02:00",
                        "2023-09-03 21:45:00+02:00",
                        "NaT",
                        "NaT",
                        "NaT",
                    ]
                ),
            }
        )
        periods["start_datetime_ed"] = periods["start_datetime_ed"].dt.tz_convert(
            "Europe/Amsterdam"
        )
        periods["end_datetime_ed"] = periods["end_datetime_ed"].dt.tz_convert(
            "Europe/Amsterdam"
        )

        home_players = pd.DataFrame(
            {
                "id": [278, 108],
                "full_name": ["Team1 Player1", "Team1 Player2"],
                "position": ["left-midfield", "bench"],
                "starter": [True, False],
                "shirt_num": [27, 10],
            }
        )
        away_players = pd.DataFrame(
            {
                "id": [438, 28],
                "full_name": ["Team2 Player1", "Team2 Player2"],
                "position": ["left-centre-back", "right-centre-midfield"],
                "starter": [True, True],
                "shirt_num": [43, 2],
            }
        )
        self.expected_metadata = Metadata(
            match_id=1999,
            pitch_dimensions=[np.nan, np.nan],
            periods_frames=periods,
            frame_rate=MISSING_INT,
            home_team_id=2000,
            home_team_name="Team 1",
            home_players=home_players,
            home_score=MISSING_INT,
            home_formation="0010",
            away_team_id=2001,
            away_team_name="Team 2",
            away_players=away_players,
            away_score=MISSING_INT,
            away_formation="0110",
            country="Netherlands",
        )

    def test_load_ortec_event_data(self):
        event_data, metadata = load_ortec_event_data(
            event_data_loc=None, metadata_loc=self.meta_data_loc
        )
        assert event_data is None
        assert metadata == self.expected_metadata

        with self.assertWarns(DataBallPyWarning):
            load_ortec_event_data(
                event_data_loc="some_locaiton", metadata_loc=self.meta_data_loc
            )

    def test_load_metadata(self):
        result = load_metadata(self.meta_data_loc)
        assert result == self.expected_metadata

    def test_get_player_info(self):
        input = [
            {
                "DisplayName": "Some Name",
                "Id": 123,
                "Role": "Left-Midfield",
                "ShirtNumber": 22,
            },
            {
                "DisplayName": "Second Name",
                "Id": 124,
                "Role": "bench",
                "ShirtNumber": 32,
            },
        ]
        expected_output = pd.DataFrame(
            {
                "id": [123, 124],
                "full_name": ["Some Name", "Second Name"],
                "position": ["Left-Midfield", "bench"],
                "starter": [True, False],
                "shirt_num": [22, 32],
            }
        )

        output = get_player_info(input)
        pd.testing.assert_frame_equal(output, expected_output)

    def test_get_formation(self):
        input = pd.DataFrame(
            {
                "position": [
                    "GoalKeeper",
                    "bench",
                    "Back",
                    "Left-Forward",
                    "Center-Back",
                    "left-Midfielder",
                ]
            }
        )
        expected_output = "1211"
        output = _get_formation(input)
        assert output == expected_output
