import unittest

import pandas as pd

from databallpy.load_data.event_data.opta import (
    _get_player_info,
    _load_event_data,
    _load_metadata,
    load_opta_event_data,
)
from tests.expected_outcomes import ED_OPTA, MD_OPTA, SHOT_EVENTS_OPTA


class TestOpta(unittest.TestCase):
    def setUp(self):
        self.f7_loc = "tests/test_data/f7_test.xml"
        self.f7_loc_no_timestamps = "tests/test_data/f7_test_no_timestamps.xml"
        self.f7_loc_multiple_matches = "tests/test_data/f7_test_multiple_matches.xml"
        self.f24_loc = "tests/test_data/f24_test.xml"

    def test_load_opta_event_data(self):

        event_data, metadata, dbp_events = load_opta_event_data(
            self.f7_loc, self.f24_loc, pitch_dimensions=[10.0, 10.0]
        )
        pd.testing.assert_frame_equal(event_data, ED_OPTA)
        assert metadata == MD_OPTA

        # SHOT_EVENTS_OPTA is scaled to a pitch of [106, 68],
        # while here [10, 10] is expected.
        shot_events_opta = SHOT_EVENTS_OPTA.copy()
        for shot_event in shot_events_opta.values():
            shot_event.start_x = shot_event.start_x / 106 * 10
            shot_event.start_y = shot_event.start_y / 68 * 10
        assert dbp_events == {"shot_events": SHOT_EVENTS_OPTA}

    def test_load_metadata(self):

        metadata = _load_metadata(self.f7_loc, [10.0, 10.0])
        assert metadata == MD_OPTA

    def test_load_metadata_multiple_matches(self):

        metadata = _load_metadata(self.f7_loc_multiple_matches, [10.0, 10.0])
        # the second match metadata is dropped
        assert metadata == MD_OPTA

    def test_load_metadata_no_timestamps(self):
        with self.assertRaises(ValueError):
            _load_metadata(self.f7_loc_no_timestamps, [10.0, 10.0])

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
        event_data, dbp_events = _load_event_data(
            self.f24_loc, away_team_id=194, country="Netherlands"
        )

        # player name is added in other function later in the pipeline
        expected_event_data = ED_OPTA.copy().drop("player_name", axis=1)

        # away team coordinates are changed later on in the pipeline
        expected_event_data.loc[[0, 2, 6, 8, 10], ["start_x", "start_y"]] *= -1

        # scaling of pitch dimension is done later on in the pipeline
        expected_event_data.loc[:, ["start_x", "start_y"]] = (
            expected_event_data.loc[:, ["start_x", "start_y"]] + 5
        ) * 10

        pd.testing.assert_frame_equal(event_data, expected_event_data)

        assert dbp_events == {"shot_events": SHOT_EVENTS_OPTA}
