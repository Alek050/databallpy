import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from databallpy.events.atomic_spadl import ATOMIC_SPADL_ACTION_COLUMNS
from databallpy.schemas import EventData
from databallpy.utils.constants import ATOMIC_SPADL_ACTION_TYPES
from databallpy.utils.get_game import get_game
from databallpy.utils.warnings import DataBallPyWarning
from tests.events.test_spadl import PITCH_DIMENSIONS, _get_event_data


def _type_names(actions: pd.DataFrame) -> list[str]:
    return [ATOMIC_SPADL_ACTION_TYPES[type_id] for type_id in actions["type_id"]]


def _set_action(event_data, idx, action_type, result="success", bodypart="foot"):
    event_data.loc[idx, ["action_type", "action_result", "action_bodypart"]] = [
        action_type,
        result,
        bodypart,
    ]


def _append_event(event_data, **values):
    row = event_data.iloc[[-1]].copy()
    row["event_id"] = event_data["event_id"].max() + 1
    row["original_event_id"] = "g"
    for col, value in values.items():
        row[col] = value
    return EventData(pd.concat([event_data, row], ignore_index=True), provider="test")


class TestToAtomicSpadl(unittest.TestCase):
    def setUp(self):
        self.event_data = _get_event_data()

    def _to_atomic(self, event_data=None, **kwargs):
        event_data = self.event_data if event_data is None else event_data
        return event_data.to_atomic_spadl(
            home_team_id=1, pitch_dimensions=PITCH_DIMENSIONS, **kwargs
        )

    def _next_type_after(self, actions, original_event_id, type_name):
        types = _type_names(actions)
        idx = next(
            i
            for i, (event_id, name) in enumerate(
                zip(actions["original_event_id"], types)
            )
            if event_id == original_event_id and name == type_name
        )
        if idx + 1 == len(types):
            return None, None
        return types[idx + 1], actions.iloc[idx + 1]

    def test_to_atomic_spadl_spadl_coordinates(self):
        res = self._to_atomic(game_id=99, coordinate_system="spadl")
        expected = pd.DataFrame(
            {
                "game_id": [99] * 7,
                "original_event_id": ["a", "a", None, "b", "b", "d", "f"],
                "event_id": pd.array([0, 0, pd.NA, 1, 1, 3, 5], dtype="Int64"),
                "action_id": [0, 1, 2, 3, 4, 5, 6],
                "period_id": [1, 1, 1, 1, 1, 1, 2],
                "time_seconds": [10.0, 11.0, 12.0, 14.0, 15.0, 16.0, 5.0],
                "team_id": [1, 1, 1, 1, 2, 2, 1],
                "player_id": [11, 12, 12, 12, 22, 22, 13],
                "x": [52.5, 63.0, 63.0, 73.5, 84.0, 21.0, 94.5],
                "y": [34.0, 34.0, 34.0, 34.0, 47.6, 47.6, 34.0],
                "dx": [10.5, 0.0, 10.5, 10.5, 0.0, 73.5, 10.5],
                "dy": [0.0, 0.0, 0.0, 13.6, 0.0, -13.6, 0.0],
                "type_id": [
                    ATOMIC_SPADL_ACTION_TYPES.index(name)
                    for name in [
                        "pass",
                        "receival",
                        "dribble",
                        "pass",
                        "interception",
                        "clearance",
                        "shot",
                    ]
                ],
                "bodypart_id": [0, 0, 0, 4, 0, 1, 1],
            }
        )
        expected[["game_id", "original_event_id", "team_id", "player_id"]] = expected[
            ["game_id", "original_event_id", "team_id", "player_id"]
        ].astype(object)

        self.assertListEqual(list(res.columns), ATOMIC_SPADL_ACTION_COLUMNS)
        pd.testing.assert_frame_equal(res, expected, check_dtype=False)

    def test_to_atomic_spadl_databallpy_coordinates(self):
        res = self._to_atomic()
        np.testing.assert_allclose(res["x"], [0.0, 10.0, 10.0, 20.0, 30.0, -30.0, 40.0])
        np.testing.assert_allclose(res["y"], [0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 0.0])
        np.testing.assert_allclose(res["dx"], [10.0, 0.0, 10.0, 10.0, 0.0, 70.0, 10.0])
        np.testing.assert_allclose(res["dy"], [0.0, 0.0, 0.0, 10.0, 0.0, -10.0, 0.0])

    def test_to_atomic_spadl_playing_direction(self):
        home = self._to_atomic()
        cols = ["x", "y", "dx", "dy"]
        is_home = home["team_id"] == 1
        for direction, to_flip in [
            ("away_oriented", pd.Series(True, index=home.index)),
            ("possession_oriented", ~is_home),
            ("out_of_possession_oriented", is_home),
        ]:
            with self.subTest(direction=direction):
                res = self._to_atomic(playing_direction=direction)
                expected = home[cols].copy()
                expected.loc[to_flip] *= -1
                pd.testing.assert_frame_equal(res[cols], expected)

        with self.subTest(direction="away_oriented, spadl coordinates"):
            home = self._to_atomic(coordinate_system="spadl")
            res = self._to_atomic(
                playing_direction="away_oriented", coordinate_system="spadl"
            )
            np.testing.assert_allclose(res["x"], 105.0 - home["x"])
            np.testing.assert_allclose(res["y"], 68.0 - home["y"])
            np.testing.assert_allclose(res["dx"], -home["dx"])
            np.testing.assert_allclose(res["dy"], -home["dy"])

    def test_to_atomic_spadl_team_oriented_deprecated(self):
        with self.assertWarns(DataBallPyWarning):
            res = self._to_atomic(playing_direction="team_oriented")
        pd.testing.assert_frame_equal(res, self._to_atomic())

    def test_to_atomic_spadl_wrong_input(self):
        with self.assertRaises(ValueError):
            self._to_atomic(playing_direction="wrong")
        with self.assertRaises(ValueError):
            self._to_atomic(coordinate_system="wrong")

    def test_to_atomic_spadl_instat(self):
        event_data = EventData(self.event_data, provider="instat")
        with self.assertRaises(NotImplementedError):
            self._to_atomic(event_data)

    def test_to_atomic_spadl_no_actions(self):
        event_data = self.event_data.copy()
        event_data["action_type"] = None
        res = self._to_atomic(event_data)
        self.assertEqual(len(res), 0)
        self.assertListEqual(list(res.columns), ATOMIC_SPADL_ACTION_COLUMNS)

    def test_to_atomic_spadl_after_pass(self):
        # pass b of team 1 (player 12) is followed by event d of team 2 (player 22)
        cases = {
            "throw_in_other_team": (2, "throw_in", "success", "out", 1, 12),
            "throw_in_same_team": (1, "throw_in", "success", "out", 1, 12),
            "goalkick_other_team": (2, "goalkick", "success", "out", 1, 12),
            "corner_other_team": (2, "corner_crossed", "success", "out", 1, 12),
            "corner_same_team": (1, "corner_short", "success", "receival", 1, 22),
            "pass_other_team": (2, "pass", "success", "interception", 2, 22),
            "pass_same_team": (1, "pass", "success", "receival", 1, 22),
            "interception": (2, "interception", "success", "interception", 2, 22),
            "tackle": (2, "tackle", "success", "tackle", 2, 22),
            "keeper_save": (2, "keeper_save", "success", "keeper_save", 2, 22),
        }
        for name, (
            team_id,
            action_type,
            result,
            expected,
            team,
            player,
        ) in cases.items():
            with self.subTest(case=name):
                event_data = self.event_data.copy()
                _set_action(event_data, 3, action_type, result)
                event_data.loc[3, "team_id"] = team_id
                res = self._to_atomic(event_data)
                next_type, next_action = self._next_type_after(res, "b", "pass")
                self.assertEqual(next_type, expected)
                self.assertEqual(next_action["team_id"], team)
                self.assertEqual(next_action["player_id"], player)
                if name.startswith("throw_in"):
                    # no dribble from the out to the throw in
                    self.assertNotIn("dribble", _type_names(res)[5:])

        with self.subTest(case="offside"):
            event_data = self.event_data.copy()
            event_data.loc[1, "action_result"] = "offside"
            next_type, next_action = self._next_type_after(
                self._to_atomic(event_data), "b", "pass"
            )
            self.assertEqual(next_type, "offside")
            self.assertEqual(next_action["player_id"], 12)

        with self.subTest(case="next_other_period"):
            event_data = self.event_data.drop(index=[3, 5])
            next_type, _ = self._next_type_after(
                self._to_atomic(event_data), "b", "pass"
            )
            self.assertIsNone(next_type)

    def test_to_atomic_spadl_dribble_after_interception(self):
        # event d of team 2 starts close to the end of pass b, which it intercepts
        event_data = self.event_data.copy()
        event_data.loc[3, ["start_x", "start_y"]] = [20.0, 10.0]
        # headers never get a dribble before them
        event_data.loc[3, "action_bodypart"] = "foot"
        res = self._to_atomic(event_data)
        self.assertListEqual(_type_names(res)[3:6], ["pass", "interception", "dribble"])
        dribble = res.iloc[5]
        self.assertEqual(dribble["team_id"], 2)
        self.assertEqual(dribble["player_id"], 22)
        self.assertTrue(pd.isna(dribble["event_id"]))
        np.testing.assert_allclose([dribble["dx"], dribble["dy"]], [-10.0, 0.0])

    def test_to_atomic_spadl_after_shot(self):
        cases = {
            "goal": ("success", None, "goal"),
            "owngoal": ("owngoal", None, "owngoal"),
            "missed": ("fail", None, None),
            "out_goalkick": ("fail", "goalkick", "out"),
            "out_corner": ("fail", "corner_crossed", "out"),
        }
        for name, (result, next_type, expected) in cases.items():
            with self.subTest(case=name):
                event_data = self.event_data.copy()
                event_data.loc[5, "action_result"] = result
                if next_type is not None:
                    event_data = _append_event(
                        event_data,
                        seconds=10.0,
                        team_id=2,
                        player_id=21,
                        action_type=next_type,
                        action_result="success",
                        action_bodypart="foot",
                    )
                res = self._to_atomic(event_data)
                res_type, res_action = self._next_type_after(res, "f", "shot")
                if next_type is not None and expected is None:
                    expected = next_type
                self.assertEqual(res_type, expected)
                if expected is not None:
                    self.assertEqual(res_action["team_id"], 1)
                    self.assertEqual(res_action["player_id"], 13)
                    self.assertEqual(res_action["original_event_id"], "f")
                    np.testing.assert_allclose(
                        [res_action["x"], res_action["dx"]], [50.0, 0.0]
                    )

        with self.subTest(case="owngoal_clearance"):
            event_data = self.event_data.copy()
            event_data.loc[3, "action_result"] = "owngoal"
            next_type, _ = self._next_type_after(
                self._to_atomic(event_data), "d", "clearance"
            )
            self.assertEqual(next_type, "owngoal")

    def test_to_atomic_spadl_cards(self):
        for result in ["yellow_card", "red_card"]:
            with self.subTest(result=result):
                event_data = self.event_data.copy()
                _set_action(event_data, 2, "foul", result)
                res = self._to_atomic(event_data)
                next_type, next_action = self._next_type_after(res, "c", "foul")
                self.assertEqual(next_type, result)
                self.assertEqual(next_action["player_id"], 21)

    def test_to_atomic_spadl_simplified_types(self):
        cases = {
            "corner_crossed": (0, "corner"),
            "corner_short": (0, "corner"),
            "freekick_crossed": (0, "freekick"),
            "freekick_short": (0, "freekick"),
            "shot_freekick": (5, "shot_freekick"),
            "shot_penalty": (5, "shot_penalty"),
        }
        for action_type, (idx, expected) in cases.items():
            with self.subTest(action_type=action_type):
                event_data = self.event_data.copy()
                _set_action(event_data, idx, action_type, "fail")
                res = self._to_atomic(event_data)
                original_event_id = event_data.loc[idx, "original_event_id"]
                is_event = (res["original_event_id"] == original_event_id).tolist()
                self.assertEqual(_type_names(res)[is_event.index(True)], expected)


class TestGameToAtomicSpadl(unittest.TestCase):
    def test_game_to_atomic_spadl(self):
        game = get_game(
            event_data_loc="tests/test_data/f24_test.xml",
            event_metadata_loc="tests/test_data/f7_test.xml",
            event_data_provider="opta",
        )
        with patch.object(
            EventData, "to_atomic_spadl", return_value="atomic_spadl"
        ) as mock:
            res = game.to_atomic_spadl(
                game_id=5,
                playing_direction="possession_oriented",
                coordinate_system="spadl",
            )

        self.assertEqual(res, "atomic_spadl")
        mock.assert_called_once_with(
            home_team_id=game.home_team_id,
            pitch_dimensions=game.pitch_dimensions,
            game_id=5,
            playing_direction="possession_oriented",
            coordinate_system="spadl",
        )
