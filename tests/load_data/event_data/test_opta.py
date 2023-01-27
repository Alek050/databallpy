import unittest

import numpy as np
import pandas as pd

from databallpy.load_data.event_data.opta import (
    _get_player_info,
    _load_event_data,
    _load_metadata,
    load_opta_event_data,
)
from databallpy.load_data.metadata import Metadata


class TestOpta(unittest.TestCase):
    def setUp(self):
        self.f7_loc = "tests/data/f7_test.xml"
        self.f24_loc = "tests/data/f24_test.xml"
        self.expected_metadata = Metadata(
            match_id=1908,
            pitch_dimensions=[np.nan, np.nan],
            match_start_datetime=np.datetime64("20230122111832"),
            periods_frames=pd.DataFrame(),
            frame_rate=np.nan,
            home_team_id=318,
            home_team_name="sc Heerenveen",
            home_formation="4231",
            home_score=3,
            home_players=pd.DataFrame(
                {
                    "player_id": [181078, 510654],
                    "player_name": ["Anas Tahiri", "Timo Zaal"],
                    "formation_place": [4, 0],
                    "position": ["midfielder", "midfielder"],
                    "starter": [True, False],
                    "shirt_number": [26, 34],
                }
            ),
            away_team_id=425,
            away_team_name="FC Groningen",
            away_formation="3412",
            away_score=1,
            away_players=pd.DataFrame(
                {
                    "player_id": [223089, 213399],
                    "player_name": ["Johan Hove", "Ramon Pascal Lundqvist"],
                    "formation_place": [8, 0],
                    "position": ["midfielder", "midfielder"],
                    "starter": [True, False],
                    "shirt_number": [8, 22],
                }
            ),
        )
        self.expected_event_data = pd.DataFrame(
            {
                "event_id": [
                    2499582269,
                    2499594199,
                    2499594195,
                    2499594225,
                    2499594243,
                    2499594271,
                    2499594279,
                    2499594285,
                    2499594291,
                ],
                "type_id": [34, 32, 32, 1, 1, 100, 43, 3, 7],
                "event": [
                    "team set up",
                    "start",
                    "start",
                    "pass",
                    "pass",
                    None,
                    "deleted event",
                    "take on",
                    "tackle",
                ],
                "period_id": [16, 1, 1, 1, 1, 1, 2, 2, 2],
                "minutes": [0, 0, 0, 0, 0, 0, 30, 30, 31],
                "seconds": [0, 0, 0, 1, 4, 6, 9, 10, 10],
                "player_id": [
                    np.nan,
                    np.nan,
                    np.nan,
                    181078,
                    510654,
                    510654,
                    223089,
                    510654,
                    223089,
                ],
                "team_id": [425, 318, 425, 318, 318, 318, 425, 318, 425],
                "outcome": [1, 1, 1, 1, 0, 0, 1, 0, 1],
                # field dimensions should be [10, 10] to scale down all values by
                # a factor of 10
                "start_x": [5.0, -5.0, 5.0, -0.03, -1.84, -1.9, 5.0, 1.57, 1.57],
                "start_y": [5.0, -5.0, 5.0, 0.01, -0.93, -0.57, 5.0, -2.68, -2.68],
                "datetime": np.array(
                    [
                        "2023-01-22T10:28:32.117",
                        "2023-01-22T11:18:32.152",
                        "2023-01-22T11:18:32.152",
                        "2023-01-22T11:18:33.637",
                        "2023-01-22T11:18:36.207",
                        "2023-01-22T11:18:39.109",
                        "2023-01-22T11:18:41.615",
                        "2023-01-22T11:18:43.119",
                        "2023-01-22T11:18:43.120",
                    ],
                    dtype="datetime64",
                ),
                "player_name": [
                    np.nan,
                    np.nan,
                    np.nan,
                    "Anas Tahiri",
                    "Timo Zaal",
                    "Timo Zaal",
                    "Johan Hove",
                    "Timo Zaal",
                    "Johan Hove",
                ],
            }
        )

    def test_load_opta_event_data(self):

        metadata, event_data = load_opta_event_data(
            self.f7_loc, self.f24_loc, pitch_dimensions=[10, 10]
        )
        pd.testing.assert_frame_equal(event_data, self.expected_event_data)
        assert metadata.match_id == self.expected_metadata.match_id

    def test_load_metadata(self):

        metadata = _load_metadata(self.f7_loc)
        assert metadata == self.expected_metadata

    def test_get_player_info(self):
        player_data = [
            {
                "PlayerRef": "s123",
                "Formation_Place": "0",
                "Position": "Substitute",
                "SubPosition": "Midfielder",
                "Status": "Substitute",
                "ShirtNumber": "33",
            },
            {
                "PlayerRef": "s234",
                "Formation_Place": "1",
                "Position": "GoalKeeper",
                "Status": "Start",
                "ShirtNumber": "2",
            },
        ]
        player_names = {"123": "Sven Kerhoffs", "234": "Niels Smits"}

        expected_result = pd.DataFrame(
            {
                "player_id": [123, 234],
                "player_name": ["Sven Kerhoffs", "Niels Smits"],
                "formation_place": [0, 1],
                "position": ["midfielder", "goalkeeper"],
                "starter": [False, True],
                "shirt_number": [33, 2],
            }
        )

        result = _get_player_info(player_data, player_names)
        pd.testing.assert_frame_equal(result, expected_result)

    def test_load_event_data(self):
        event_data = _load_event_data(self.f24_loc)

        # player name is added in other function later in the pipeline
        expected_event_data = self.expected_event_data.copy().drop(
            "player_name", axis=1
        )

        # away team coordinates are changed later on in the pipeling
        expected_event_data.loc[[0, 2, 6, 8], ["start_x", "start_y"]] *= -1

        # scaling of pitch dimension is done later on in the pipeling
        expected_event_data.loc[:, ["start_x", "start_y"]] = (
            expected_event_data.loc[:, ["start_x", "start_y"]] + 5
        ) * 10

        pd.testing.assert_frame_equal(event_data, expected_event_data)
