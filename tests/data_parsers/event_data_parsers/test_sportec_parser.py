import os
import unittest
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from databallpy.data_parsers.event_data_parsers.sportec_parser import (
    _get_sportec_event_data,
    _handle_play_event,
    _handle_shot_event,
    _handle_tackling_game_event,
    _initialize_search_variables,
    load_sportec_event_data,
    load_sportec_open_event_data,
)
from tests.expected_outcomes import (
    SPORTEC_DATABALLPY_EVENTS,
    SPORTEC_EVENT_DATA,
    SPORTEC_METADATA_ED,
)


class TestSportecParser(unittest.TestCase):
    def setUp(self) -> None:
        self.expected_ed = SPORTEC_EVENT_DATA.copy()
        self.expected_md = SPORTEC_METADATA_ED.copy()
        self.ed_loc = os.path.join(
            os.getcwd(), "tests", "test_data", "sportec_ed_test.xml"
        )
        self.md_loc = os.path.join(
            os.getcwd(), "tests", "test_data", "sportec_metadata.xml"
        )
        with open(self.ed_loc, "r") as file:
            lines = file.read()
        self.soup = BeautifulSoup(lines, "xml")
        self.dbp_events = SPORTEC_DATABALLPY_EVENTS

    def test_load_sportec_event_data(self):
        with self.assertRaises(FileNotFoundError):
            load_sportec_event_data(event_data_loc="wrong", metadata_loc=self.md_loc)
        with self.assertRaises(FileNotFoundError):
            load_sportec_event_data(event_data_loc=self.ed_loc, metadata_loc="wrong2")

        res_ed, res_md, res_dbp_events = load_sportec_event_data(
            self.ed_loc, self.md_loc
        )

        exp_metadata = self.expected_md.copy()
        pd.testing.assert_frame_equal(res_ed, self.expected_ed)
        self.assertEqual(res_md, exp_metadata)
        self.assertDictEqual(res_dbp_events, self.dbp_events)

    def test_get_sportec_event_data(self):
        res_ed, res_dbp_events = _get_sportec_event_data(
            self.ed_loc, self.expected_md.copy()
        )
        pd.testing.assert_frame_equal(res_ed, self.expected_ed)
        self.assertDictEqual(res_dbp_events, self.dbp_events)

    def test_initialize_search_variables(self):
        pitch_center, period_start_times, swap_period = _initialize_search_variables(
            self.soup, self.expected_md.home_team_id
        )

        self.assertListEqual(pitch_center, [52.5, 34.0])
        assert (
            period_start_times
            == pd.to_datetime(
                ["2022-11-11T18:31:12.000+01:00", "2022-11-11T19:31:09.000+01:00"]
            ).tz_convert("Europe/Berlin")
        ).all()
        self.assertEqual(swap_period, 1)

    def test_handle_shot_event(self):
        event = self.soup.find("Event", {"EventId": "13"}).find_next()
        base_kwargs = {
            "set_piece": "no_set_piece",
            "datetime": pd.to_datetime("2022-11-11T18:37:36.200+01:00"),
            "period_id": 1,
            "minutes": 6,
            "seconds": 24.2,
            "event_id": 13,
            "start_x": -98.41 + 52.5,
            "start_y": -36.55 + 34.0,
            "sportec_event": "SavedShot",
            "player_id": "B-3",
            "team_id": "Team2",
        }
        res_kwargs, dbp_shot_event = _handle_shot_event(
            event, self.expected_md, base_kwargs
        )

        expected_kwargs = base_kwargs.copy()
        expected_kwargs["team_side"] = "away"
        expected_kwargs["pitch_size"] = [105.0, 68.0]
        expected_kwargs["jersey"] = 5
        expected_kwargs["sportec_event"] = "SavedShot"
        expected_kwargs["databallpy_event"] = "shot"
        expected_kwargs["related_event_id"] = None
        expected_kwargs["body_part"] = "head"
        expected_kwargs["possession_type"] = "free_kick"
        expected_kwargs["outcome"] = False
        expected_kwargs["outcome_str"] = "miss_on_target"

        self.assertDictEqual(res_kwargs, expected_kwargs)
        self.assertEqual(dbp_shot_event, self.dbp_events["shot_events"][13])

    def test_handle_play_event(self):
        event = self.soup.find("Event", {"EventId": "12"}).find_next().find_next()
        base_kwargs = {
            "set_piece": "kick_off",
            "datetime": pd.to_datetime("2022-11-11T18:31:12.000+01:00"),
            "period_id": 1,
            "minutes": 0,
            "seconds": 0,
            "event_id": 12,
            "start_x": 0.0,
            "start_y": 0.0,
            "sportec_event": "Pass",
            "player_id": "B-1",
            "team_id": "Team2",
        }
        res_kwargs, dbp_play_event = _handle_play_event(
            event, self.expected_md, base_kwargs
        )

        expected_kwargs = base_kwargs.copy()
        expected_kwargs["team_side"] = "away"
        expected_kwargs["pitch_size"] = [105.0, 68.0]
        expected_kwargs["jersey"] = 28
        expected_kwargs["sportec_event"] = "Pass"
        expected_kwargs["databallpy_event"] = "pass"
        expected_kwargs["related_event_id"] = None
        expected_kwargs["body_part"] = "unspecified"
        expected_kwargs["possession_type"] = "unspecified"
        expected_kwargs["outcome"] = True
        expected_kwargs["outcome_str"] = "unspecified"
        expected_kwargs["end_x"] = np.nan
        expected_kwargs["end_y"] = np.nan
        expected_kwargs["pass_type"] = "unspecified"
        expected_kwargs["receiver_player_id"] = "B-3"

        self.assertDictEqual(res_kwargs, expected_kwargs)
        self.assertEqual(dbp_play_event, self.dbp_events["pass_events"][12])

    def test_tackling_game_event(self):
        event = self.soup.find("Event", {"EventId": "17"}).find_next()
        base_kwargs = {
            "set_piece": "no_set_piece",
            "datetime": pd.to_datetime("2022-11-11T20:09:28.600+01:00"),
            "period_id": 2,
            "minutes": 83,
            "seconds": 19.6,
            "event_id": 17,
            "start_x": 15.19 - 52.5,
            "start_y": 4.39 - 34.0,
            "sportec_event": "TacklingGame",
            "player_id": "A-5",
            "team_id": "Team1",
        }
        res_kwargs, dbp_dribble_event = _handle_tackling_game_event(
            event, self.expected_md, base_kwargs
        )

        expected_kwargs = base_kwargs.copy()
        expected_kwargs["team_side"] = "home"
        expected_kwargs["pitch_size"] = [105.0, 68.0]
        expected_kwargs["jersey"] = 1
        expected_kwargs["sportec_event"] = "dribbledAround"
        expected_kwargs["databallpy_event"] = "dribble"
        expected_kwargs["outcome"] = True
        expected_kwargs["possession_type"] = "open_play"
        expected_kwargs["duel_type"] = "unspecified"
        expected_kwargs["with_opponent"] = True
        expected_kwargs["related_event_id"] = None
        expected_kwargs["databallpy_event"] = "dribble"

        self.assertDictEqual(res_kwargs, expected_kwargs)
        self.assertEqual(dbp_dribble_event, self.dbp_events["dribble_events"][17])

    @patch("databallpy.data_parsers.event_data_parsers.sportec_parser.requests.get")
    @patch("databallpy.data_parsers.event_data_parsers.sportec_parser.os.makedirs")
    @patch("databallpy.data_parsers.event_data_parsers.sportec_parser.os.path.exists")
    @patch(
        "databallpy.data_parsers.event_data_parsers.sportec_parser.open",
        new_callable=mock_open,
    )
    @patch(
        "databallpy.data_parsers.event_data_parsers.sportec_parser.load_sportec_event_data"
    )
    def test_load_sportec_open_event_data(
        self,
        mock_load_sportec_event_data,
        mock_open,
        mock_exists,
        mock_makedirs,
        mock_requests_get,
    ):
        # Setup mock responses
        mock_exists.side_effect = [
            False,
            False,
            True,
            True,
        ]  # metadata.xml does not exist, event_data.xml does not exist, then both exist
        mock_requests_get.return_value = MagicMock(content=b"mock content")
        mock_load_sportec_event_data.return_value = (
            pd.DataFrame(),
            "mock_metadata",
            {"mock_key": {"mock_subkey": "mock_value"}},
        )

        game_id = "J03WMX"
        expected_metadata_path = os.path.join(
            os.getcwd(), "datasets", "IDSSE", game_id, "metadata.xml"
        )
        expected_event_data_path = os.path.join(
            os.getcwd(), "datasets", "IDSSE", game_id, "event_data.xml"
        )

        # Call the function
        result = load_sportec_open_event_data(game_id)

        # Verify the function calls
        mock_makedirs.assert_called_once_with(
            os.path.join(os.getcwd(), "datasets", "IDSSE", game_id), exist_ok=True
        )
        self.assertEqual(mock_requests_get.call_count, 2)
        mock_open.assert_any_call(expected_metadata_path, "wb")
        mock_open.assert_any_call(expected_event_data_path, "wb")
        mock_load_sportec_event_data.assert_called_once_with(
            expected_event_data_path, expected_metadata_path
        )

        # Verify the return value
        pd.testing.assert_frame_equal(result[0], pd.DataFrame())
        self.assertEqual(result[1], "mock_metadata")
        self.assertDictEqual(result[2], {"mock_key": {"mock_subkey": "mock_value"}})
