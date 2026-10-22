import json
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from databallpy.data_parsers.event_data_parsers.metrica_event_data_parser import (
    _get_databallpy_events,
    _get_event_data,
    _get_metrica_spadl_action,
    _update_metrica_spadl_actions,
    load_metrica_event_data,
    load_metrica_open_event_data,
)
from databallpy.schemas import EventData
from databallpy.utils.constants import SPADL_ACTION_TYPES, SPADL_RESULTS
from tests.expected_outcomes import (
    DRIBBLE_INSTANCES_METRICA,
    ED_METRICA,
    MD_METRICA_ED,
    PASS_INSTANCES_METRICA,
    SHOT_INSTANCES_METRICA,
)
from tests.mocks import ED_METRICA_RAW, MD_METRICA_RAW

ED_METRICA = pd.DataFrame(ED_METRICA.copy())


class TestMetricaEventDataParser(unittest.TestCase):
    def setUp(self):
        self.md_loc = "tests/test_data/metrica_metadata_test.xml"
        self.ed_loc = "tests/test_data/metrica_event_data_test.json"

    def test_load_metrica_event_data(self):
        ed, md, dbpe = load_metrica_event_data(self.ed_loc, self.md_loc)
        assert md == MD_METRICA_ED
        pd.testing.assert_frame_equal(ed, ED_METRICA)
        assert dbpe["shot_events"] == SHOT_INSTANCES_METRICA
        assert dbpe["pass_events"] == PASS_INSTANCES_METRICA
        assert dbpe["dribble_events"] == DRIBBLE_INSTANCES_METRICA

        with self.assertRaises(TypeError):
            load_metrica_event_data(22, self.md_loc)

    def test_load_metrica_event_data_errors(self):
        with self.assertRaises(FileNotFoundError):
            load_metrica_event_data(self.ed_loc[3:], self.md_loc)
        with self.assertRaises(FileNotFoundError):
            load_metrica_event_data(self.ed_loc, self.md_loc[1:])

    def test_get_event_data(self):
        expected_event_data = ED_METRICA.copy()
        for col in ["end_x", "start_x"]:
            expected_event_data[col] = (expected_event_data[col] + 50) / 100.0
        for col in ["end_y", "start_y"]:
            expected_event_data[col] = (25 - expected_event_data[col]) / 50.0

        expected_event_data.drop(["datetime"], axis=1, inplace=True)
        ed, cards = _get_event_data(self.ed_loc)
        pd.testing.assert_frame_equal(ed, expected_event_data)
        assert cards == {}

    def test_get_event_data_take_on(self):
        with open(self.ed_loc, "r", encoding="utf-8") as f:
            events_json = json.load(f)
        carry = events_json["data"][2]

        def challenge(team_id, player_id, result):
            return {
                **carry,
                "type": {"name": "CHALLENGE", "id": 9},
                "subtypes": [{"name": "DRIBBLE"}, {"name": result}],
                "from": {"name": f"Player {player_id}", "id": f"P{player_id}"},
                "team": {"name": team_id, "id": team_id},
                "to": None,
                "end": {"x": None, "y": None},
            }

        # the carry is followed by a dribble challenge of the attacker and defender
        events_json["data"][3:3] = [
            challenge(carry["team"]["id"], carry["from"]["id"][1:], "WON"),
            challenge("FIFATMB", "3700", "LOST"),
        ]
        ed, _ = _get_event_data(json.dumps(events_json))

        self.assertEqual(ed.loc[2, "original_event"], "carry")
        self.assertListEqual(
            ed.loc[3, ["action_type", "action_result", "action_bodypart"]].tolist(),
            ["take_on", "success", "foot"],
        )
        self.assertTrue(
            ed.loc[4, ["action_type", "action_result", "action_bodypart"]].isna().all()
        )

    @patch(
        "requests.get",
        side_effect=[Mock(text=ED_METRICA_RAW), Mock(text=MD_METRICA_RAW)],
    )
    def test_load_open_metrica_event_data(self, _):
        ed, md, dbpe = load_metrica_open_event_data()
        assert md == MD_METRICA_ED
        pd.testing.assert_frame_equal(ed, ED_METRICA)
        assert dbpe["shot_events"] == SHOT_INSTANCES_METRICA
        assert dbpe["pass_events"] == PASS_INSTANCES_METRICA
        assert dbpe["dribble_events"] == DRIBBLE_INSTANCES_METRICA

    def test_get_databallpy_events(self):
        players = pd.concat(
            [MD_METRICA_ED.home_players, MD_METRICA_ED.away_players], ignore_index=True
        )
        res_dbpe = _get_databallpy_events(
            ED_METRICA, pitch_dimensions=(106, 68), home_team_id=1, all_players=players
        )
        shot_events = res_dbpe["shot_events"]
        expected_shot_events = {}
        for event_id, event in shot_events.items():
            expected_shot_events[event_id] = event.copy()
            expected_shot_events[event_id].pitch_size = (106, 68)

        pass_events = res_dbpe["pass_events"]
        expected_pass_events = {}
        for event_id, event in pass_events.items():
            expected_pass_events[event_id] = event.copy()
            expected_pass_events[event_id].pitch_size = (106, 68)
        dribble_events = res_dbpe["dribble_events"]
        expected_dribble_events = {}
        for event_id, event in dribble_events.items():
            expected_dribble_events[event_id] = event.copy()
            expected_dribble_events[event_id].pitch_size = (106, 68)

        assert res_dbpe["shot_events"] == expected_shot_events
        assert res_dbpe["pass_events"] == expected_pass_events
        assert res_dbpe["dribble_events"] == expected_dribble_events

    def test_get_metrica_spadl_action(self):
        def event(event_type, subtypes=None, end=(0.6, 0.5)):
            if isinstance(subtypes, list):
                subtypes = [{"name": name} for name in subtypes]
            return {
                "type": {"name": event_type},
                "subtypes": subtypes,
                "start": {"x": 0.5, "y": 0.5},
                "end": {"x": end[0], "y": end[1]},
            }

        cases = [
            (event("PASS"), None, ("pass", "success", "foot")),
            (event("PASS", ["HEAD"]), None, ("pass", "success", "head")),
            (event("PASS", ["OFFSIDE"]), None, ("pass", "offside", "foot")),
            (event("PASS", ["CROSS"]), None, ("cross", "success", "foot")),
            (event("PASS", ["GOAL KICK"]), None, ("goalkick", "success", "foot")),
            (event("PASS", ["DEEP BALL"]), None, ("pass", "success", "foot")),
            (
                event("PASS"),
                {"name": "THROW IN"},
                ("throw_in", "success", "other"),
            ),
            (
                event("PASS", ["CROSS"]),
                {"name": "CORNER KICK"},
                ("corner_crossed", "success", "foot"),
            ),
            (
                event("PASS"),
                {"name": "CORNER KICK"},
                ("corner_short", "success", "foot"),
            ),
            (
                event("PASS"),
                [{"name": "FREE KICK"}, {"name": "INDIRECT"}],
                ("freekick_short", "success", "foot"),
            ),
            (
                event("PASS", ["CROSS"]),
                {"name": "FREE KICK"},
                ("freekick_crossed", "success", "foot"),
            ),
            (event("PASS"), {"name": "KICK OFF"}, ("pass", "success", "foot")),
            (event("BALL LOST"), None, ("pass", "fail", "foot")),
            (event("BALL LOST", ["CROSS"]), None, ("cross", "fail", "foot")),
            (
                event("BALL LOST", ["CLEARANCE", "HEAD"]),
                None,
                ("clearance", "success", "head"),
            ),
            (
                event("BALL OUT", ["GOAL KICK"]),
                None,
                ("goalkick", "fail", "foot"),
            ),
            (
                event("BALL OUT"),
                {"name": "THROW IN"},
                ("throw_in", "fail", "other"),
            ),
            (event("BALL LOST", ["REFEREE HIT"]), None, ("pass", "fail", "foot")),
            (
                event("BALL LOST", ["THEFT"], end=(None, None)),
                None,
                ("bad_touch", "fail", "foot"),
            ),
            (event("BALL LOST", ["FORCED"]), None, ("bad_touch", "fail", "foot")),
            (
                event("BALL LOST", ["HAND BALL"], end=(None, None)),
                None,
                ("foul", "fail", "other"),
            ),
            (
                event("BALL LOST", ["OFFSIDE"], end=(None, None)),
                None,
                (None, None, None),
            ),
            (
                event("BALL LOST", ["END HALF"], end=(0.5, 0.5)),
                None,
                (None, None, None),
            ),
            (
                event("SHOT", ["ON TARGET", "GOAL"]),
                None,
                ("shot", "success", "foot"),
            ),
            (
                event("SHOT", ["HEAD", "OFF TARGET", "OUT"]),
                None,
                ("shot", "fail", "head"),
            ),
            (
                event("SHOT", {"name": "BLOCKED"}),
                [{"name": "FREE KICK"}, {"name": "DIRECT"}],
                ("shot_freekick", "fail", "foot"),
            ),
            (
                event("SHOT", ["GOAL"]),
                {"name": "PENALTY"},
                ("shot_penalty", "success", "foot"),
            ),
            (event("CARRY"), None, ("dribble", "success", "foot")),
            (event("CARRY", end=(0.5, 0.5)), None, (None, None, None)),
            (
                event("CHALLENGE", ["TACKLE", "WON"]),
                None,
                ("tackle", "success", "foot"),
            ),
            (
                event("CHALLENGE", ["TACKLE", "LOST"]),
                None,
                ("tackle", "fail", "foot"),
            ),
            (
                event("CHALLENGE", ["GROUND", "FAULT", "LOST"]),
                None,
                ("foul", "fail", "foot"),
            ),
            (
                event("CHALLENGE", ["TACKLE", "FAULT", "WON"]),
                None,
                (None, None, None),
            ),
            (
                event("CHALLENGE", ["DRIBBLE", "WON"]),
                None,
                ("take_on", "success", "foot"),
            ),
            (
                event("CHALLENGE", ["DRIBBLE", "LOST"]),
                None,
                ("take_on", "fail", "foot"),
            ),
            (
                event("CHALLENGE", ["DRIBBLE", "FAULT", "LOST"]),
                None,
                ("foul", "fail", "foot"),
            ),
            (event("CHALLENGE", ["AERIAL", "WON"]), None, (None, None, None)),
            (event("CHALLENGE", ["GROUND", "LOST"]), None, (None, None, None)),
            (
                event("RECOVERY", ["INTERCEPTION"], end=(None, None)),
                None,
                ("interception", "success", "foot"),
            ),
            (
                event("RECOVERY", ["SAVED"], end=(None, None)),
                None,
                ("keeper_save", "success", "other"),
            ),
            (event("RECOVERY", end=(None, None)), None, ("dribble", "success", "foot")),
            (
                event("RECOVERY", ["THEFT"], end=(None, None)),
                None,
                (None, None, None),
            ),
            (event("SET PIECE", ["THROW IN"]), None, (None, None, None)),
            (event("FAULT RECEIVED"), None, (None, None, None)),
            (event("CARD", ["YELLOW"]), None, (None, None, None)),
        ]
        for metrica_event, set_piece, expected in cases:
            with self.subTest(event=metrica_event, set_piece=set_piece):
                self.assertEqual(
                    _get_metrica_spadl_action(metrica_event, set_piece), expected
                )

    def test_update_metrica_spadl_actions(self):
        start = pd.to_datetime("2019-02-21T03:30:00", utc=True)
        event_data = pd.DataFrame(
            {
                "original_event": [
                    "recovery",
                    "pass",
                    "recovery",
                    "pass",
                    "challenge",
                    "fault received",
                    "card",
                    "challenge",
                    "card",
                ],
                "player_id": [1, 1, 2, 2, 3, 4, 3, 5, 5],
                "period_id": [1] * 9,
                "datetime": [
                    start + pd.Timedelta(seconds=s)
                    for s in [0, 2, 4, 5, 10, 10, 25, 30, 95]
                ],
                "start_x": [0.0, 10.0, 20.0, 21.0, 30.0, 30.0, np.nan, 40.0, np.nan],
                "start_y": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, np.nan, 0.0, np.nan],
                "end_x": [
                    np.nan,
                    20.0,
                    np.nan,
                    30.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                "end_y": [
                    np.nan,
                    0.0,
                    np.nan,
                    0.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                "action_type": [
                    "dribble",
                    "pass",
                    "dribble",
                    "pass",
                    "foul",
                    None,
                    None,
                    "foul",
                    None,
                ],
                "action_result": [
                    "success",
                    "success",
                    "success",
                    "success",
                    "fail",
                    None,
                    None,
                    "fail",
                    None,
                ],
                "action_bodypart": [
                    "foot",
                    "foot",
                    "foot",
                    "foot",
                    "foot",
                    None,
                    None,
                    "foot",
                    None,
                ],
            }
        )
        res = _update_metrica_spadl_actions(
            event_data.copy(), {6: "yellow_card", 8: "red_card"}
        )

        # recovery that moved the ball becomes a dribble to the next action
        self.assertEqual(res.loc[0, "action_type"], "dribble")
        self.assertEqual(res.loc[0, ["end_x", "end_y"]].tolist(), [10.0, 0.0])
        # recovery without movement is no action
        self.assertTrue(res.loc[2, ["action_type", "action_result"]].isna().all())
        # card after the foul of the same player sets the result
        self.assertEqual(res.loc[4, "action_result"], "yellow_card")
        # card more than 60 seconds after the foul is ignored
        self.assertEqual(res.loc[7, "action_result"], "fail")
        self.assertTrue(res.loc[[6, 8], "action_type"].isna().all())

    def test_to_spadl(self):
        event_data, metadata, _ = load_metrica_event_data(self.ed_loc, self.md_loc)
        # the second period of the test data starts in the first minute
        event_data["period_id"] = 1
        res = EventData(event_data, provider="metrica").to_spadl(
            home_team_id=metadata.home_team_id,
            pitch_dimensions=metadata.pitch_dimensions,
            coordinate_system="spadl",
        )
        res = res[res["original_event_id"].notna()].reset_index(drop=True)
        self.assertListEqual(
            [SPADL_ACTION_TYPES[i] for i in res["type_id"]],
            ["pass", "dribble", "shot", "shot", "shot", "tackle", "foul"],
        )
        self.assertListEqual(
            [SPADL_RESULTS[i] for i in res["result_id"]],
            ["success", "success", "fail", "success", "success", "fail", "fail"],
        )
