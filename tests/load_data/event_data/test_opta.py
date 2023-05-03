import unittest

import pandas as pd

from databallpy.load_data.event_data.opta import (
    _get_player_info,
    _load_event_data,
    _load_metadata,
    load_opta_event_data,
)
from tests.expected_outcomes import ED_OPTA, MD_OPTA


class TestOpta(unittest.TestCase):
    def setUp(self):
        self.f7_loc = "tests/test_data/f7_test.xml"
        self.f24_loc = "tests/test_data/f24_test.xml"

    def test_load_opta_event_data(self):

        event_data, metadata = load_opta_event_data(
            self.f7_loc, self.f24_loc, pitch_dimensions=[10.0, 10.0]
        )
        pd.testing.assert_frame_equal(event_data, ED_OPTA)
        assert metadata == MD_OPTA

    def test_load_metadata(self):

        metadata = _load_metadata(self.f7_loc, [10.0, 10.0])
        assert metadata == MD_OPTA

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
                "id": [123, 234],
                "full_name": ["Sven Kerhoffs", "Niels Smits"],
                "formation_place": [0, 1],
                "position": ["midfielder", "goalkeeper"],
                "starter": [False, True],
                "shirt_num": [33, 2],
            }
        )

        result = _get_player_info(player_data, player_names)
        pd.testing.assert_frame_equal(result, expected_result)

    def test_load_event_data(self):
        event_data = _load_event_data(self.f24_loc, country="Netherlands")

        # player name is added in other function later in the pipeline
        expected_event_data = ED_OPTA.copy().drop("player_name", axis=1)

        # away team coordinates are changed later on in the pipeling
        expected_event_data.loc[[0, 2, 6, 8], ["start_x", "start_y"]] *= -1

        # scaling of pitch dimension is done later on in the pipeling
        expected_event_data.loc[:, ["start_x", "start_y"]] = (
            expected_event_data.loc[:, ["start_x", "start_y"]] + 5
        ) * 10

        pd.testing.assert_frame_equal(event_data, expected_event_data)
