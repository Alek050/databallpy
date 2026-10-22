import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from databallpy.data_parsers.event_data_parsers.sportec_parser import (
    _get_sportec_event_data,
    _get_sportec_spadl_action,
    _handle_play_event,
    _handle_shot_event,
    _handle_tackling_game_event,
    _initialize_search_variables,
    _update_sportec_spadl_passes,
    load_sportec_event_data,
    load_sportec_open_event_data,
)
from databallpy.schemas import EventData
from databallpy.utils.constants import SPADL_ACTION_TYPES
from tests.expected_outcomes import (
    SPORTEC_DATABALLPY_EVENTS,
    SPORTEC_EVENT_DATA,
    SPORTEC_METADATA_ED,
)

SPORTEC_EVENT_DATA = pd.DataFrame(SPORTEC_EVENT_DATA.drop(columns=["team_name"]).copy())


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
        for key in self.dbp_events.keys():
            self.assertDictEqual(res_dbp_events[key], self.dbp_events[key])

    def test_get_sportec_event_data(self):
        res_ed, res_dbp_events = _get_sportec_event_data(
            self.ed_loc, self.expected_md.copy()
        )
        pd.testing.assert_frame_equal(res_ed, self.expected_ed)
        for key in self.dbp_events.keys():
            self.assertDictEqual(res_dbp_events[key], self.dbp_events[key])

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
            "event_id": 1,
            "start_x": -98.41 + 52.5,
            "start_y": -36.55 + 34.0,
            "original_event": "SavedShot",
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
        expected_kwargs["original_event"] = "SavedShot"
        expected_kwargs["databallpy_event"] = "shot"
        expected_kwargs["related_event_id"] = None
        expected_kwargs["body_part"] = "head"
        expected_kwargs["possession_type"] = "free_kick"
        expected_kwargs["outcome"] = False
        expected_kwargs["outcome_str"] = "miss_on_target"

        self.assertDictEqual(res_kwargs, expected_kwargs)
        self.assertEqual(dbp_shot_event, self.dbp_events["shot_events"][1])

    def test_handle_play_event(self):
        event = self.soup.find("Event", {"EventId": "12"}).find_next().find_next()
        base_kwargs = {
            "set_piece": "kick_off",
            "datetime": pd.to_datetime("2022-11-11T18:31:12.000+01:00"),
            "period_id": 1,
            "minutes": 0,
            "seconds": 0,
            "event_id": 0,
            "start_x": 0.0,
            "start_y": 0.0,
            "original_event": "Pass",
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
        expected_kwargs["original_event"] = "Pass"
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
        self.assertEqual(dbp_play_event, self.dbp_events["pass_events"][0])

    def test_tackling_game_event(self):
        event = self.soup.find("Event", {"EventId": "17"}).find_next()
        base_kwargs = {
            "set_piece": "no_set_piece",
            "datetime": pd.to_datetime("2022-11-11T20:09:28.600+01:00"),
            "period_id": 2,
            "minutes": 83,
            "seconds": 19.6,
            "event_id": 4,
            "start_x": 15.19 - 52.5,
            "start_y": 4.39 - 34.0,
            "original_event": "TacklingGame",
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
        expected_kwargs["original_event"] = "dribbledAround"
        expected_kwargs["databallpy_event"] = "dribble"
        expected_kwargs["outcome"] = True
        expected_kwargs["possession_type"] = "open_play"
        expected_kwargs["duel_type"] = "unspecified"
        expected_kwargs["with_opponent"] = True
        expected_kwargs["related_event_id"] = None
        expected_kwargs["databallpy_event"] = "dribble"

        self.assertDictEqual(res_kwargs, expected_kwargs)
        self.assertEqual(dbp_dribble_event, self.dbp_events["dribble_events"][4])

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
        cache_path = Path().parent / "IDSSE" / game_id

        expected_metadata_path = cache_path / "metadata.xml"
        expected_event_data_path = cache_path / "event_data.xml"

        # Call the function
        result = load_sportec_open_event_data(game_id, cache_path=cache_path)

        # Verify the function calls
        mock_makedirs.assert_called_once_with(cache_path, exist_ok=True)
        self.assertEqual(mock_requests_get.call_count, 2)
        mock_open.assert_any_call(expected_metadata_path, "wb")
        mock_open.assert_any_call(expected_event_data_path, "wb")
        mock_load_sportec_event_data.assert_called_once_with(
            str(expected_event_data_path), str(expected_metadata_path)
        )

        # Verify the return value
        pd.testing.assert_frame_equal(result[0], pd.DataFrame())
        self.assertEqual(result[1], "mock_metadata")
        self.assertDictEqual(result[2], {"mock_key": {"mock_subkey": "mock_value"}})


class TestSportecSpadl(unittest.TestCase):
    def setUp(self) -> None:
        self.ed_loc = os.path.join(
            os.getcwd(), "tests", "test_data", "sportec_ed_test.xml"
        )
        self.md_loc = os.path.join(
            os.getcwd(), "tests", "test_data", "sportec_metadata.xml"
        )

    def _action(self, xml, set_piece="no_set_piece"):
        return _get_sportec_spadl_action(ET.fromstring(xml), set_piece)

    def test_get_sportec_spadl_action_play(self):
        cases = [
            (
                '<Play Evaluation="successfullyCompleted"><Pass/></Play>',
                "no_set_piece",
                ("pass", "success", "foot"),
            ),
            (
                '<Play Evaluation="successful"><Cross/></Play>',
                "no_set_piece",
                ("cross", "success", "foot"),
            ),
            (
                '<Play Evaluation="unsuccessful"><Pass/></Play>',
                "kick_off",
                ("pass", "fail", "foot"),
            ),
            (
                '<Play Evaluation="unsuccessful"><Pass/></Play>',
                "throw_in",
                ("throw_in", "fail", "other"),
            ),
            (
                '<Play Evaluation="successful"><Pass/></Play>',
                "goal_kick",
                ("goalkick", "success", "foot"),
            ),
            (
                '<Play Evaluation="successful"><Cross/></Play>',
                "corner_kick",
                ("corner_crossed", "success", "foot"),
            ),
            (
                '<Play Evaluation="successful" Height="flat"><Pass/></Play>',
                "corner_kick",
                ("corner_short", "success", "foot"),
            ),
            (
                '<Play Evaluation="successful" Height="high"><Pass/></Play>',
                "free_kick",
                ("freekick_crossed", "success", "foot"),
            ),
            (
                '<Play Evaluation="successful"><Pass/></Play>',
                "free_kick",
                ("freekick_short", "success", "foot"),
            ),
        ]
        for xml, set_piece, expected in cases:
            with self.subTest(xml=xml, set_piece=set_piece):
                self.assertEqual(self._action(xml, set_piece), expected)

    def test_get_sportec_spadl_action_shot(self):
        cases = [
            (
                '<ShotAtGoal TypeOfShot="head"><SuccessfulShot/></ShotAtGoal>',
                "no_set_piece",
                ("shot", "success", "head"),
            ),
            (
                '<ShotAtGoal TypeOfShot="leftLeg"><SavedShot/></ShotAtGoal>',
                "free_kick",
                ("shot_freekick", "fail", "foot_left"),
            ),
            (
                '<ShotAtGoal TypeOfShot="rightLeg"><ShotWide/></ShotAtGoal>',
                "penalty",
                ("shot_penalty", "fail", "foot_right"),
            ),
            (
                "<ShotAtGoal><OwnGoal/></ShotAtGoal>",
                "no_set_piece",
                ("bad_touch", "owngoal", "foot"),
            ),
        ]
        for xml, set_piece, expected in cases:
            with self.subTest(xml=xml, set_piece=set_piece):
                self.assertEqual(self._action(xml, set_piece), expected)

    def test_get_sportec_spadl_action_other_events(self):
        cases = [
            (
                '<TacklingGame WinnerResult="dribbledAround" '
                'DribbleEvaluation="successful"/>',
                ("take_on", "success", "foot"),
            ),
            (
                '<TacklingGame WinnerResult="dribbledAround" '
                'DribbleEvaluation="unsuccessful"/>',
                ("take_on", "fail", "foot"),
            ),
            (
                '<TacklingGame WinnerResult="ballcontactSucceeded" Type="ground" '
                'WinnerRole="withoutBallControl"/>',
                ("tackle", "success", "foot"),
            ),
            (
                '<TacklingGame WinnerResult="ballcontactSucceeded" Type="air" '
                'WinnerRole="withoutBallControl"/>',
                (None, None, None),
            ),
            (
                '<BallClaiming Type="InterceptedBall"/>',
                ("interception", "success", "foot"),
            ),
            ('<BallClaiming Type="BallClaimed"/>', (None, None, None)),
            ('<Foul Fouler="A-1" TeamFouler="Team1"/>', ("foul", "fail", "foot")),
            ('<OtherBallAction Player="B-2"/>', (None, None, None)),
        ]
        for xml, expected in cases:
            with self.subTest(xml=xml):
                self.assertEqual(self._action(xml), expected)

    def test_update_sportec_spadl_passes(self):
        event_data = pd.DataFrame(
            {
                "action_type": ["pass", "pass", "throw_in", "shot", "pass"],
                "action_result": ["success", "success", "fail", "fail", "success"],
                "end_x": [np.nan] * 5,
                "end_y": [np.nan] * 5,
                "source_x": [0.0, 1.0, 2.0, 3.0, 4.0],
                "source_y": [0.0, -1.0, -2.0, -3.0, -4.0],
            }
        )
        res = _update_sportec_spadl_passes(event_data.copy())

        self.assertListEqual(
            list(res.columns), ["action_type", "action_result", "end_x", "end_y"]
        )
        # the pass before the throw in went out of play
        self.assertListEqual(
            res["action_result"].tolist(),
            ["success", "fail", "fail", "fail", "success"],
        )
        # completed passes end at the start of the next event
        np.testing.assert_array_equal(
            res["end_x"], [1.0, np.nan, np.nan, np.nan, np.nan]
        )
        np.testing.assert_array_equal(
            res["end_y"], [-1.0, np.nan, np.nan, np.nan, np.nan]
        )

    def test_get_sportec_event_data_foul(self):
        with open(self.ed_loc, "r") as file:
            lines = file.read()
        # sportec fouls have no position
        lines = lines.replace(
            "</PutDataRequest>",
            '<Event EventId="19" EventTime="2022-11-11T20:11:18.500+01:00" '
            'MatchId="Match-1"><Foul Fouler="B-2" TeamFouler="Team2" Fouled="A-3" '
            'TeamFouled="Team1"/></Event></PutDataRequest>',
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            loc = os.path.join(tmp_dir, "sportec_foul.xml")
            with open(loc, "w") as file:
                file.write(lines)
            res_ed, _ = _get_sportec_event_data(loc, SPORTEC_METADATA_ED.copy())

        self.assertEqual(len(res_ed), 7)
        foul = res_ed.iloc[-1]
        self.assertEqual(foul["original_event_id"], 19)
        self.assertTrue(np.isnan(foul["start_x"]))
        self.assertTrue(np.isnan(foul["end_x"]))
        self.assertEqual(foul["player_id"], "B-2")
        self.assertEqual(foul["team_id"], "Team2")
        self.assertEqual(foul["action_type"], "foul")
        self.assertEqual(foul["action_result"], "fail")

    def test_get_sportec_event_data_caution(self):
        with open(self.ed_loc, "r") as file:
            lines = file.read()
        # cautions have no position and set the result of the foul
        lines = lines.replace(
            "</PutDataRequest>",
            '<Event EventId="19" EventTime="2022-11-11T20:11:18.500+01:00" '
            'MatchId="Match-1"><Foul Fouler="B-2" TeamFouler="Team2" Fouled="A-3" '
            'TeamFouled="Team1"/></Event>'
            '<Event EventId="20" EventTime="2022-11-11T20:11:30.000+01:00" '
            'MatchId="Match-1"><Caution CardColor="yellowRed" Team="Team2" '
            'Player="B-2"/></Event></PutDataRequest>',
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            loc = os.path.join(tmp_dir, "sportec_caution.xml")
            with open(loc, "w") as file:
                file.write(lines)
            res_ed, _ = _get_sportec_event_data(loc, SPORTEC_METADATA_ED.copy())

        # the caution itself is not kept in the event data
        self.assertEqual(len(res_ed), 7)
        foul = res_ed.iloc[-1]
        self.assertEqual(foul["action_type"], "foul")
        self.assertEqual(foul["action_result"], "red_card")

    def test_get_sportec_spadl_action_caution(self):
        for color, result in [("yellow", "yellow_card"), ("red", "red_card")]:
            event = ET.fromstring(f'<Caution CardColor="{color}" Player="A-1"/>')
            self.assertEqual(
                _get_sportec_spadl_action(event, "no_set_piece"), (None, result, None)
            )

    def test_to_spadl(self):
        event_data, metadata, _ = load_sportec_event_data(self.ed_loc, self.md_loc)
        res = EventData(event_data, provider="sportec").to_spadl(
            home_team_id=metadata.home_team_id,
            pitch_dimensions=metadata.pitch_dimensions,
            coordinate_system="spadl",
        )
        self.assertListEqual(
            [SPADL_ACTION_TYPES[i] for i in res["type_id"]],
            ["pass", "shot", "pass", "take_on"],
        )
        self.assertListEqual(res["player_id"].tolist(), ["B-1", "B-3", "A-1", "A-5"])
