import json
import os
import tempfile
import unittest

import pandas as pd

from databallpy.data_parsers.event_data_parsers.statsbomb_parser import (
    _get_cell_offsets,
    _get_dribble_event,
    _get_pass_event,
    _get_player_info,
    _get_seconds,
    _get_shot_event,
    _get_statsbomb_spadl_action,
    _load_event_data,
    _load_metadata,
    _location_to_meters,
    load_statsbomb_event_data,
)
from databallpy.schemas import EventData
from databallpy.utils.constants import SPADL_ACTION_TYPES, SPADL_RESULTS
from tests.event_extras_testing import (
    assert_event_extras,
    assert_event_extras_equal,
    to_event_extras,
)
from tests.expected_outcomes import (
    DRIBBLE_EVENTS_STATSBOMB,
    ED_STATSBOMB,
    MD_STATSBOMB,
    PASS_EVENTS_STATSBOMB,
    SHOT_EVENTS_STATSBOMB,
)
from tests.pandas_testing import assert_frame_equal, to_list

ED_STATSBOMB = pd.DataFrame(ED_STATSBOMB.copy())
EXTRAS_STATSBOMB = to_event_extras(
    SHOT_EVENTS_STATSBOMB, PASS_EVENTS_STATSBOMB, DRIBBLE_EVENTS_STATSBOMB
)


class TestStatsbombParser(unittest.TestCase):
    def setUp(self):
        self.event_loc = "tests/test_data/statsbomb_event_test.json"
        self.match_loc = "tests/test_data/statsbomb_match_test.json"
        self.lineup_loc = "tests/test_data/statsbomb_lineup_test.json"
        self.pitch_dimensions = (105.0, 68.0)

    def test_load_statsbomb_event_data_errors(self):
        with self.assertRaises(ValueError):
            load_statsbomb_event_data(
                events_loc=self.event_loc,
                match_loc=self.match_loc,
                lineup_loc="statsbomb_lineup_test.xml",
                pitch_dimensions=(120.0, 80.0),
            )
        with self.assertRaises(TypeError):
            load_statsbomb_event_data(
                events_loc=self.event_loc,
                match_loc=3,
                lineup_loc=self.lineup_loc,
                pitch_dimensions=(120.0, 80.0),
            )
        with self.assertRaises(FileNotFoundError):
            load_statsbomb_event_data(
                events_loc="file_does_not_exist.json",
                match_loc=self.match_loc,
                lineup_loc=self.lineup_loc,
                pitch_dimensions=(120.0, 80.0),
            )
        with self.assertRaises(ValueError):
            load_statsbomb_event_data(
                events_loc=self.event_loc,
                match_loc=self.match_loc,
                lineup_loc=self.lineup_loc,
                pitch_dimensions=(120.0),
            )

    def test_load_metadata(self):
        expected_outcome = MD_STATSBOMB.copy()
        expected_outcome.home_formation = ""
        expected_outcome.away_formation = ""
        result = _load_metadata(
            match_loc=self.match_loc,
            lineup_loc=self.lineup_loc,
            pitch_dimensions=(105.0, 68.0),
        )

        assert expected_outcome == result

    def test_get_player_info(self):
        expected_outcome = MD_STATSBOMB.home_players
        with open(self.lineup_loc, "r", encoding="utf-8") as f:
            input = json.load(f)[0]["lineup"]
        result = _get_player_info(input)
        assert_frame_equal(expected_outcome, result)

    def test_load_event_data(self):
        expected_outcome_event_data = ED_STATSBOMB.copy()

        expected_outcome_metadata = MD_STATSBOMB

        metadata_input = MD_STATSBOMB.copy()
        metadata_input.home_formation = ""
        metadata_input.away_formation = ""

        outcome_event_data, outcome_dbp_events, outcome_metadata = _load_event_data(
            events_loc=self.event_loc,
            metadata=metadata_input,
            pitch_dimensions=(105.0, 68.0),
        )

        assert_event_extras(outcome_dbp_events, outcome_event_data)
        assert_event_extras_equal(outcome_dbp_events, EXTRAS_STATSBOMB)

        assert_frame_equal(outcome_event_data, expected_outcome_event_data, rtol=1e-3)
        assert outcome_metadata == expected_outcome_metadata

    def test_load_statsbomb_event_data(self):
        expected_outcome_event_data = ED_STATSBOMB
        expected_outcome_metadata = MD_STATSBOMB

        outcome_event_data, outcome_metadata, outcome_dbp_events = (
            load_statsbomb_event_data(
                events_loc=self.event_loc,
                match_loc=self.match_loc,
                lineup_loc=self.lineup_loc,
                pitch_dimensions=(105.0, 68.0),
            )
        )

        assert_event_extras(outcome_dbp_events, outcome_event_data)
        assert_event_extras_equal(outcome_dbp_events, EXTRAS_STATSBOMB)

        assert_frame_equal(expected_outcome_event_data, outcome_event_data, rtol=1e-3)
        assert expected_outcome_metadata == outcome_metadata

    def test_get_shot_event(self):
        expected_outcome = {
            "related_event_id": ["7fb36c67-4b6c-4c3d-bc52-4e1cd712e790"],
            "outcome": False,
            "body_part": "left_foot",
            "possession_type": "open_play",
            "set_piece": "no_set_piece",
            "outcome_str": "miss_off_target",
        }
        with open(self.event_loc, "r", encoding="utf-8") as f:
            input = json.load(f)[4]

        res = _get_shot_event(event=input)
        self.assertDictEqual(res, expected_outcome)

        for outcome, expected in [
            ("Saved Off Target", "miss"),
            ("Saved to Post", "miss_hit_post"),
        ]:
            input["shot"]["outcome"]["name"] = outcome
            res = _get_shot_event(event=input)
            assert res["outcome_str"] == expected

    def test_get_pass_event(self):
        expected_outcome = {
            "related_event_id": ["994ee6b7-0e8b-4168-ac79-81c68ee580f7"],
            "outcome": False,
            "body_part": "left_foot",
            "possession_type": "corner_kick",
            "set_piece": "corner_kick",
            "receiver_player_id": 5246,
            "outcome_str": "unspecified",
            "pass_type": "through_ball",
        }
        with open(self.event_loc, "r", encoding="utf-8") as f:
            input = json.load(f)[2]

        res = _get_pass_event(event=input)
        self.assertDictEqual(res, expected_outcome)

    def test_get_dribble_event(self):
        expected_outcome = {
            "related_event_id": ["4c601a79-df4f-4001-902b-6fc9788718eb"],
            "outcome": False,
            "body_part": "unspecified",
            "possession_type": "unspecified",
            "set_piece": "unspecified",
            "duel_type": "unspecified",
            "with_opponent": None,
        }
        with open(self.event_loc, "r", encoding="utf-8") as f:
            input = json.load(f)[5]

        res = _get_dribble_event(event=input)
        self.assertDictEqual(res, expected_outcome)


def _statsbomb_event(name: str, info: dict | None = None) -> dict:
    event = {"type": {"id": 0, "name": name}}
    if info is not None:
        event[name.lower().replace(" ", "_").replace("goal_keeper", "goalkeeper")] = info
    return event


class TestStatsbombSpadl(unittest.TestCase):
    def setUp(self):
        self.event_loc = "tests/test_data/statsbomb_event_test.json"
        self.match_loc = "tests/test_data/statsbomb_match_test.json"
        self.lineup_loc = "tests/test_data/statsbomb_lineup_test.json"

    def test_get_statsbomb_spadl_action(self):
        left_foot = {"body_part": {"name": "Left Foot"}}
        cases = [
            (_statsbomb_event("Pass", {}), ("pass", "success", "foot")),
            (
                _statsbomb_event(
                    "Pass", {"cross": True, "outcome": {"name": "Incomplete"}}
                ),
                ("cross", "fail", "foot"),
            ),
            (
                _statsbomb_event(
                    "Pass",
                    {"type": {"name": "Free Kick"}, "height": {"name": "High Pass"}},
                ),
                ("freekick_crossed", "success", "foot"),
            ),
            (
                _statsbomb_event("Pass", {"type": {"name": "Free Kick"}}),
                ("freekick_short", "success", "foot"),
            ),
            (
                _statsbomb_event("Pass", {"type": {"name": "Corner"}, "cross": True}),
                ("corner_crossed", "success", "foot"),
            ),
            (
                _statsbomb_event("Pass", {"type": {"name": "Corner"}, **left_foot}),
                ("corner_short", "success", "foot_left"),
            ),
            (
                _statsbomb_event(
                    "Pass",
                    {"type": {"name": "Goal Kick"}, "body_part": {"name": "Drop Kick"}},
                ),
                ("goalkick", "success", "foot"),
            ),
            (
                _statsbomb_event("Pass", {"type": {"name": "Throw-in"}}),
                ("throw_in", "success", "other"),
            ),
            (
                _statsbomb_event(
                    "Pass",
                    {
                        "outcome": {"name": "Pass Offside"},
                        "body_part": {"name": "Head"},
                    },
                ),
                ("pass", "offside", "head"),
            ),
            (
                _statsbomb_event("Pass", {"outcome": {"name": "Injury Clearance"}}),
                (None, None, None),
            ),
            (
                _statsbomb_event(
                    "Shot", {"outcome": {"name": "Goal"}, "type": {"name": "Penalty"}}
                ),
                ("shot_penalty", "success", "foot"),
            ),
            (
                _statsbomb_event(
                    "Shot",
                    {
                        "outcome": {"name": "Saved"},
                        "type": {"name": "Free Kick"},
                        "body_part": {"name": "Right Foot"},
                    },
                ),
                ("shot_freekick", "fail", "foot_right"),
            ),
            (
                _statsbomb_event(
                    "Shot",
                    {"outcome": {"name": "Off T"}, "body_part": {"name": "Other"}},
                ),
                ("shot", "fail", "other"),
            ),
            (
                _statsbomb_event("Goal Keeper", {"type": {"name": "Shot Saved"}}),
                ("keeper_save", "success", "other"),
            ),
            (
                _statsbomb_event(
                    "Goal Keeper",
                    {"type": {"name": "Smother"}, "outcome": {"name": "No Touch"}},
                ),
                ("keeper_save", "fail", "other"),
            ),
            (
                _statsbomb_event(
                    "Goal Keeper",
                    {"type": {"name": "Penalty Saved to Post"}, **left_foot},
                ),
                ("keeper_save", "success", "foot_left"),
            ),
            (
                _statsbomb_event(
                    "Goal Keeper",
                    {
                        "type": {"name": "Keeper Sweeper"},
                        "outcome": {"name": "Clear"},
                        **left_foot,
                    },
                ),
                ("clearance", "success", "foot_left"),
            ),
            (
                _statsbomb_event(
                    "Goal Keeper",
                    {"type": {"name": "Keeper Sweeper"}, "outcome": {"name": "Claim"}},
                ),
                ("keeper_claim", "success", "other"),
            ),
            (
                _statsbomb_event(
                    "Goal Keeper",
                    {"type": {"name": "Punch"}, "outcome": {"name": "In Play Danger"}},
                ),
                ("keeper_punch", "fail", "other"),
            ),
            (
                _statsbomb_event("Goal Keeper", {"type": {"name": "Shot Faced"}}),
                (None, None, None),
            ),
            (_statsbomb_event("Foul Committed"), ("foul", "fail", "foot")),
            (
                _statsbomb_event("Foul Committed", {"card": {"name": "Yellow Card"}}),
                ("foul", "yellow_card", "foot"),
            ),
            (
                _statsbomb_event("Foul Committed", {"card": {"name": "Second Yellow"}}),
                ("foul", "red_card", "foot"),
            ),
            (
                _statsbomb_event("Foul Committed", {"card": {"name": "Red Card"}}),
                ("foul", "red_card", "foot"),
            ),
            (
                _statsbomb_event(
                    "Duel", {"type": {"name": "Tackle"}, "outcome": {"name": "Won"}}
                ),
                ("tackle", "success", "foot"),
            ),
            (
                _statsbomb_event(
                    "Duel",
                    {"type": {"name": "Tackle"}, "outcome": {"name": "Lost In Play"}},
                ),
                ("tackle", "fail", "foot"),
            ),
            (
                _statsbomb_event("Duel", {"type": {"name": "Aerial Lost"}}),
                (None, None, None),
            ),
            (
                _statsbomb_event("Interception", {"outcome": {"name": "Lost Out"}}),
                ("interception", "fail", "foot"),
            ),
            (
                _statsbomb_event("Interception", {"outcome": {"name": "Won"}}),
                ("interception", "success", "foot"),
            ),
            (
                _statsbomb_event("Dribble", {"outcome": {"name": "Incomplete"}}),
                ("take_on", "fail", "foot"),
            ),
            (
                _statsbomb_event("Dribble", {"outcome": {"name": "Complete"}}),
                ("take_on", "success", "foot"),
            ),
            (
                _statsbomb_event("Clearance", {"body_part": {"name": "Head"}}),
                ("clearance", "success", "head"),
            ),
            (_statsbomb_event("Carry", {}), ("dribble", "success", "foot")),
            (_statsbomb_event("Own Goal Against"), ("bad_touch", "owngoal", "other")),
            (_statsbomb_event("Miscontrol"), ("bad_touch", "fail", "foot")),
            (_statsbomb_event("Ball Receipt*"), (None, None, None)),
        ]
        for event, expected in cases:
            with self.subTest(event=event):
                self.assertEqual(_get_statsbomb_spadl_action(event), expected)

    def test_get_seconds(self):
        self.assertAlmostEqual(
            _get_seconds({"second": 37, "timestamp": "00:31:37.826"}), 37.826
        )
        self.assertEqual(_get_seconds({"second": 37, "timestamp": "00:31:37"}), 37.0)
        self.assertEqual(_get_seconds({"second": 37}), 37.0)
        self.assertTrue(pd.isna(_get_seconds({"timestamp": "00:31:37.826"})))

    def test_get_cell_offsets(self):
        shot = {"type": {"name": "Shot"}, "location": [100.0, 40.0]}
        high_fidelity_shot = {"type": {"name": "Shot"}, "location": [100.3, 40.0]}
        other = {"type": {"name": "Pass"}, "location": [50.0, 40.0]}
        high_fidelity_other = {"type": {"name": "Pass"}, "location": [50.0, 40.1]}
        no_location = {"type": {"name": "Pass"}}
        cases = [
            ([shot, other, no_location], (0.5, 0.5)),
            ([high_fidelity_shot, other], (0.5, 0.05)),
            ([shot, high_fidelity_other], (0.05, 0.05)),
        ]
        for events, expected in cases:
            with self.subTest(events=events):
                self.assertEqual(_get_cell_offsets(events), expected)

    def test_location_to_meters(self):
        x, y = _location_to_meters([60.5, 40.5], 0.5, (105.0, 68.0), 0.875, 0.85)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 0.0)
        # the y of a 3D shot end location is always detailed to a tenth of a yard
        x, y = _location_to_meters([120.0, 40.05, 1.0], 0.5, (105.0, 68.0), 0.875, 0.85)
        self.assertAlmostEqual(x, 52.0625)
        self.assertAlmostEqual(y, 0.0)

    def test_load_event_data_interception_and_own_goal(self):
        with open(self.event_loc, "r", encoding="utf-8") as f:
            events_json = json.load(f)
        events_json[2]["pass"]["type"] = {"id": 64, "name": "Interception"}
        events_json.append(
            {
                "id": "own-goal-id",
                "period": 1,
                "minute": 10,
                "second": 0,
                "type": {"id": 20, "name": "Own Goal Against"},
                "team": {"id": 206, "name": "Deportivo Alavés"},
                "player": {"id": 6581, "name": "Jonathan Rodríguez Menéndez"},
                "location": [5.0, 30.0],
            }
        )
        metadata = MD_STATSBOMB.copy()
        with tempfile.TemporaryDirectory() as tmp_dir:
            events_loc = os.path.join(tmp_dir, "events.json")
            with open(events_loc, "w", encoding="utf-8") as f:
                json.dump(events_json, f)
            event_data, dbp_events, _ = _load_event_data(
                events_loc, metadata, (105.0, 68.0)
            )

        self.assertEqual(len(event_data), len(ED_STATSBOMB) + 2)
        self.assertListEqual(event_data["event_id"].tolist(), list(range(7)))
        interception, intercepting_pass = event_data.iloc[0], event_data.iloc[1]
        self.assertEqual(interception["original_event"], "interception")
        self.assertTrue(pd.isna(interception["databallpy_event"]))
        self.assertEqual(
            to_list(interception[["action_type", "action_result", "action_bodypart"]]),
            ["interception", "success", "foot"],
        )
        self.assertTrue(pd.isna(interception["end_x"]))
        for col in ["original_event_id", "player_id", "team_id", "start_x", "start_y"]:
            self.assertEqual(interception[col], intercepting_pass[col])
        self.assertEqual(intercepting_pass["databallpy_event"], "pass")
        self.assertListEqual(sorted(dbp_events["pass_events"]["event_id"]), [1, 5])

        # the own goal of the away team ends in the away goal on the right side
        own_goal = event_data.iloc[-1]
        self.assertEqual(own_goal["action_result"], "owngoal")
        self.assertEqual(own_goal["action_bodypart"], "other")
        self.assertAlmostEqual(own_goal["end_x"], 52.5)
        self.assertAlmostEqual(own_goal["end_y"], 0.0)

    def test_to_spadl(self):
        event_data, metadata, _ = load_statsbomb_event_data(
            self.event_loc, self.match_loc, self.lineup_loc
        )
        res = EventData(event_data, provider="statsbomb").to_spadl(
            home_team_id=metadata.home_team_id,
            pitch_dimensions=metadata.pitch_dimensions,
            coordinate_system="spadl",
        )
        res = res[res["original_event_id"].notna()].reset_index(drop=True)
        self.assertListEqual(
            [SPADL_ACTION_TYPES[i] for i in res["type_id"]],
            ["corner_crossed", "corner_crossed", "dribble", "shot", "take_on"],
        )
        self.assertListEqual(
            [SPADL_RESULTS[i] for i in res["result_id"]],
            ["fail", "fail", "success", "fail", "fail"],
        )
        # the home team shoots on the goal on the right side
        self.assertAlmostEqual(res.loc[3, "end_x"], 104.95625)
